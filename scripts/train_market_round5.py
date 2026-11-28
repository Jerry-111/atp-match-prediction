"""Separate prematch-market information experiment; final holdout stays closed.

Uses opening quotes for selection, not closing quotes or match outcomes as
features. Unmatched/invalid prices fall back exactly to frozen Round4 history.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.special import expit, logit
from threadpoolctl import threadpool_limits

from scripts.metrics import score_probabilities
from scripts.train_models import ROOT, SEED, SymmetricClassifier, mean_loss, sha256, cluster_interval
from scripts.train_round5 import accuracy, accuracy_interval, LOSS_GUARD, ACCURACY_GATE

REFERENCE = 'history_only'


def candidate_specs():
    specs = [dict(name=REFERENCE, kind='history', family='history'),
             dict(name='opening_market', kind='opening', family='opening')]
    for mode in ('probability', 'logit'):
        for weight in (.25, .5, .75):
            specs.append(dict(name=f'{mode}_blend_market{weight:g}', kind='blend', mode=mode,
                              weight=weight, family=f'{mode}_blend'))
    for design in ('market', 'fusion', 'context'):
        for c in (.01, .1, 1., 10.):
            specs.append(dict(name=f'{design}_calibrated_C{c:g}', kind='calibrated',
                              design=design, C=c, family=f'{design}_calibrated'))
    return specs


def validate_input(history, features):
    history = np.asarray(history, dtype=float)
    if history.shape != (len(features),) or not np.isfinite(history).all() or np.any((history <= 0) | (history >= 1)):
        raise ValueError('Aligned historical probabilities must be strictly between0 and1')
    required = ['market_open_logit_diff', 'market_open_available', 'market_open_overround']
    if features.empty or features.columns.duplicated().any() or not set(required).issubset(features):
        raise ValueError('Missing/duplicate/empty market inputs')
    if not np.isfinite(features[required].to_numpy(dtype=float)).all() or not features.market_open_available.isin([0, 1]).all():
        raise ValueError('Finite opening features and binary availability required')
    return history


def design_matrix(history, features, design):
    history = validate_input(history, features)
    market = features.market_open_logit_diff.to_numpy(dtype=float)
    prior = logit(history)
    values = {'market_logit_diff': market}
    if design in ('fusion', 'context'):
        values['history_logit_diff'] = prior
    if design == 'context':
        # Fixed even confidence/overround contexts; each resulting column is odd.
        close = (np.abs(market) < logit(.6)).astype(float)
        values['history_close_market_diff'] = prior * close
        values['market_history_confidence_diff'] = market * np.clip(np.abs(prior), 0, 3)
        values['market_overround_diff'] = market * np.clip(features.market_open_overround.to_numpy() - .04, -.1, .2)
    if design not in ('market', 'fusion', 'context'):
        raise ValueError('Unknown market calibration design')
    return pd.DataFrame(values, index=features.index)


class MarketModel:
    """Known-prior probability plus optional fit on valid earlier opening prices."""
    def __init__(self, spec):
        self.spec = spec

    def fit(self, history, features, outcomes):
        history = validate_input(history, features)
        y = np.asarray(outcomes)
        if y.shape != history.shape or not np.isin(y, [0, 1]).all():
            raise ValueError('Aligned binary training labels required')
        self.training_rows_ = len(y)
        self.available_training_rows_ = int(features.market_open_available.sum())
        if self.spec['kind'] == 'calibrated':
            available = features.market_open_available.eq(1).to_numpy()
            if available.sum() < 20 or len(np.unique(y[available])) != 2:
                raise ValueError('Too few valid earlier prices for calibration')
            design = design_matrix(history, features, self.spec['design'])
            fitted_spec = dict(kind='logistic', columns=list(design), params={'C': self.spec['C']})
            self.estimator_ = SymmetricClassifier(fitted_spec).fit(design[available], y[available])
        return self

    def predict_proba(self, history, features):
        history = validate_input(history, features)
        kind = self.spec['kind']
        market_logit = features.market_open_logit_diff.to_numpy(dtype=float)
        if kind == 'history':
            return history.copy()
        if kind == 'opening':
            candidate = expit(market_logit)
        elif kind == 'blend':
            weight = self.spec['weight']
            if not 0 <= weight <= 1:
                raise ValueError('Convex weight must lie in[0,1]')
            if self.spec['mode'] == 'probability':
                candidate = weight * expit(market_logit) + (1 - weight) * history
            elif self.spec['mode'] == 'logit':
                candidate = expit(weight * market_logit + (1 - weight) * logit(history))
            else:
                raise ValueError('Unknown blend mode')
        elif kind == 'calibrated':
            candidate = self.estimator_.predict_proba(design_matrix(history, features, self.spec['design']))
        else:
            raise ValueError('Unknown market model')
        available = features.market_open_available.eq(1).to_numpy()
        result = history.copy()
        result[available] = np.clip(candidate[available], 1e-8, 1 - 1e-8)
        return result


def select_candidate(y, predictions, specs, mask):
    y, mask = np.asarray(y), np.asarray(mask, dtype=bool)
    if y.shape != mask.shape or not mask.any():
        raise ValueError('Nonempty aligned selection mask required')
    limit = mean_loss(y[mask], predictions[REFERENCE][mask]) + LOSS_GUARD
    eligible = [s['name'] for s in specs if mean_loss(y[mask], predictions[s['name']][mask]) <= limit]
    key = lambda n: (-accuracy(y[mask], predictions[n][mask]), mean_loss(y[mask], predictions[n][mask]), n)
    best = min(eligible, key=key)
    families = {}
    for family in sorted({s['family'] for s in specs}):
        names = [s['name'] for s in specs if s['family'] == family and s['name'] in eligible]
        if names:
            families[family] = min(names, key=key)
    return best, families


def market_masks(rows, fold):
    dates = pd.to_datetime(rows.tourney_date)
    if fold == '2021H2':
        valid = rows.season.eq(2021) & dates.ge('2021-07-01')
    elif fold == '2022':
        valid = rows.season.eq(2022)
    else:
        raise ValueError('Unknown market selection fold')
    if not valid.any():
        raise ValueError('Market validation fold is empty')
    cutoff = dates[valid].min() - pd.Timedelta(days=21)
    train = rows.season.ge(2021) & dates.le(cutoff) & ~valid
    if not train.any():
        raise ValueError('Market training fold is empty')
    return train, valid


def market_gate(y, predictions, specs, folds):
    y = np.asarray(y)
    early, gate = np.asarray(folds) == '2021H2', np.asarray(folds) == '2022'
    if not early.any() or not gate.any():
        raise ValueError('Need2021H2selection and2022gate')
    selected, _ = select_candidate(y, predictions, specs, early)
    ref, candidate = predictions[REFERENCE][gate], predictions[selected][gate]
    gain = accuracy(y[gate], candidate) - accuracy(y[gate], ref)
    ref_loss, candidate_loss = mean_loss(y[gate], ref), mean_loss(y[gate], candidate)
    passed = gain >= ACCURACY_GATE - 1e-12 and candidate_loss <= ref_loss + LOSS_GUARD + 1e-12
    return dict(procedure='adopt_market_information' if passed else 'retain_history', early_selected=selected,
        reference_2022_accuracy=accuracy(y[gate], ref), candidate_2022_accuracy=accuracy(y[gate], candidate),
        accuracy_improvement=gain, required_accuracy_improvement=ACCURACY_GATE,
        reference_2022_loss=ref_loss, candidate_2022_loss=candidate_loss, allowed_loss_increase=LOSS_GUARD,
        selection_fold='2021H2', gate_fold='2022')


def predict_market_bundle(bundle, history, features):
    predictions = {name: model.predict_proba(history, features) for name, model in bundle['models'].items()}
    # Closing prices are a separate descriptive benchmark, never a selected input.
    if not features.market_close_available.isin([0, 1]).all():
        raise ValueError('Closing availability must be binary')
    available = features.market_close_available.eq(1).to_numpy()
    closed = np.asarray(history, dtype=float).copy()
    quotes = features.market_close_logit_diff.to_numpy(dtype=float)
    if not np.isfinite(quotes).all():
        raise ValueError('Closing logit must be finite')
    closed[available] = np.clip(expit(quotes[available]), 1e-8, 1 - 1e-8)
    predictions['closing_market_descriptive'] = closed
    return pd.DataFrame(predictions, index=features.index)


def load_history_predictions(rows, old_dir):
    """Load audited real OOF for fitting, frozen validation forecasts for2023."""
    old_dir = Path(old_dir)
    audit = json.loads((old_dir / 'artifact_audit.json').read_text())
    for name in ('oof_predictions.csv', 'validation_predictions.csv', 'frozen_selection.json'):
        if sha256(old_dir / name) != audit['artifact_sha256'][name]:
            raise ValueError('Audited previous forecasts changed')
    current = json.loads((old_dir / 'frozen_selection.json').read_text())['recommendation']
    history = pd.Series(index=rows.index, dtype=float)
    oof = pd.read_csv(old_dir / 'oof_predictions.csv')
    val = pd.read_csv(old_dir / 'validation_predictions.csv')
    for subset in (oof, val):
        selected_rows = rows[rows.match_id.isin(subset.match_id)]
        if not selected_rows[['match_id', 'season', 'a_won']].reset_index(drop=True).equals(subset[['match_id', 'season', 'a_won']]):
            raise ValueError('Historical prediction alignment changed')
        history.loc[selected_rows.index] = subset[current].to_numpy()
    if not history[rows.season.between(2021, 2023)].notna().all():
        raise ValueError('Historical forecasts omit required market-era targets')
    return history, current


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'results/round5/market')
    args = parser.parse_args()
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    from scripts.market_data import build_market_features
    rows = pd.read_csv(ROOT / 'results/round3/development_targets.csv')
    history, current = load_history_predictions(rows, ROOT / 'results/round4')
    # Earlier years are excluded: there are no historical market files for them.
    keep = rows.season.between(2021, 2023)
    rows, history = rows[keep].copy(), history[keep].to_numpy()
    market, metadata = build_market_features(rows)
    specs = candidate_specs()
    registry = dict(round=5, information_regime='Opening prematch market plus frozen historical model; separate from T-21 baseline',
        seed=SEED, candidates=specs, market_metadata=market.attrs, base_model=current,
        source_manifest_sha256=sha256(ROOT / 'data/market-manifest.json'),
        selection='2021H2 chronological selection;2022accuracy/loss gate; pooled temporal refit; fixed.5threshold',
        exploratory_after_2023=True, final_test_evaluated=False)
    (out / 'search_registry.json').write_text(json.dumps(registry, indent=2) + '\n')
    pd.concat([rows[['match_id']].reset_index(drop=True), market.reset_index(drop=True)], axis=1).to_csv(out / 'features.csv', index=False)
    metadata.to_csv(out / 'linked_matches.csv', index=False)
    folds = {}
    all_valid = pd.Series(False, index=rows.index)
    for fold in ('2021H2', '2022'):
        train, valid = market_masks(rows, fold)
        folds[fold] = (train, valid)
        all_valid |= valid
    orows = rows[all_valid]
    labels = np.where(orows.season.eq(2021), '2021H2', '2022')
    y = orows.a_won.to_numpy()
    predictions, cv = {}, []
    for spec in specs:
        p = pd.Series(index=orows.index, dtype=float)
        metrics = {}
        for fold, (train, valid) in folds.items():
            model = MarketModel(spec).fit(history[train], market[train], rows.a_won[train].to_numpy())
            probability = model.predict_proba(history[valid], market[valid])
            p.loc[rows.index[valid]] = probability
            metrics[fold + '_accuracy'] = accuracy(rows.a_won[valid], probability)
            metrics[fold + '_log_loss'] = mean_loss(rows.a_won[valid].to_numpy(), probability)
        predictions[spec['name']] = p.to_numpy()
        cv.append(dict(name=spec['name'], family=spec['family'], accuracy=accuracy(y, p.to_numpy()),
                       mean_log_loss=mean_loss(y, p.to_numpy()), **metrics))
    pd.DataFrame(cv).to_csv(out / 'temporal_cv.csv', index=False)
    gate = market_gate(y, predictions, specs, labels)
    pooled, families = select_candidate(y, predictions, specs, np.ones(len(y), dtype=bool))
    choice = pooled if gate['procedure'] == 'adopt_market_information' else REFERENCE
    frozen = dict(recommendation=choice, best_candidate=pooled, selected_by_family=families, procedure_gate=gate,
        base_model=current, information_regime=registry['information_regime'], decision_threshold=.5,
        final_test_evaluated=False, exploratory_after_2023=True)
    (out / 'frozen_selection.json').write_text(json.dumps(frozen, indent=2) + '\n')
    print('Frozen independent market procedure:', choice, gate, flush=True)
    train, valid = rows.season.le(2022), rows.season.eq(2023)
    cutoff = pd.to_datetime(rows.tourney_date[valid]).min() - pd.Timedelta(days=21)
    train &= pd.to_datetime(rows.tourney_date).le(cutoff)
    # All20 tiny finalists are reported; their settings were frozen before2023.
    models = {spec['name']: MarketModel(spec).fit(history[train], market[train], rows.a_won[train].to_numpy()) for spec in specs}
    bundle = dict(models=models, frozen=frozen, feature_names=list(market), base_model=current)
    p = predict_market_bundle(bundle, history[valid], market[valid])
    vr = rows[valid]
    result = []
    for name in p:
        metrics = score_probabilities(zip(vr.match_id, vr.a_won), zip(vr.match_id, p[name]))
        result.append(dict(model=name, recommended=name == choice, **metrics,
            brier=float(np.mean((vr.a_won.to_numpy() - p[name].to_numpy()) ** 2))))
    table = pd.DataFrame(result).sort_values(['accuracy', 'mean_log_loss'], ascending=[False, True])
    table.to_csv(out / 'validation_comparison.csv', index=False)
    pd.concat([vr.reset_index(drop=True), p.reset_index(drop=True)], axis=1).to_csv(out / 'validation_predictions.csv', index=False)
    pd.concat([orows[['match_id', 'season', 'a_won']].reset_index(drop=True), pd.DataFrame(predictions)], axis=1).to_csv(out / 'oof_predictions.csv', index=False)
    reference = p[REFERENCE].to_numpy()
    intervals = {n: dict(accuracy=accuracy_interval(vr, reference, p[n].to_numpy()),
                        log_loss=cluster_interval(vr, reference, p[n].to_numpy())) for n in p}
    compared_to_open = {n: accuracy_interval(vr, p.opening_market.to_numpy(), p[n].to_numpy()) for n in p}
    summary = dict(recommendation=choice, candidate_count=len(specs), validation_matches=len(vr),
        matched_2023=int(metadata.loc[valid, 'status'].eq('matched').sum()),
        opening_available=int(market.loc[valid, 'market_open_available'].sum()),
        closing_available=int(market.loc[valid, 'market_close_available'].sum()),
        correct_required_for_70=int(np.ceil(.7 * len(vr))), comparisons_vs_history=intervals,
        accuracy_comparisons_vs_opening=compared_to_open, final_test_evaluated=False,
        exploratory_after_2023=True, information_regime=registry['information_regime'])
    (out / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    joblib.dump(bundle, out / 'frozen_models.joblib')
    print(table.to_string(index=False), flush=True)


if __name__ == '__main__':
    from scripts.train_market_round5 import main as canonical_main
    with threadpool_limits(limits=4):
        canonical_main()
