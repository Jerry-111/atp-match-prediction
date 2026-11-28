"""Bounded residual learning around separately timed prematch price priors."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit, logit
from threadpoolctl import threadpool_limits
from scripts.metrics import score_probabilities
from scripts.train_models import ROOT, mean_loss, sha256, mirror_features
from scripts.train_round5 import accuracy, accuracy_interval
from scripts.train_round4 import cluster_interval
from scripts.train_market_round5 import market_masks

CONDITIONS = ('opening', 'latest')
ALPHAS = (.25, .5, 1.)
REFERENCE = 'raw_market'


def mirror_input(x):
    out = mirror_features(x)
    if 'reference_probability' in out:
        out['reference_probability'] = 1-x.reference_probability
    return out


def build_inputs(history, market, reference, condition):
    if condition not in CONDITIONS or not history.index.equals(market.index):
        raise ValueError('Known quote condition and identical feature indices required')
    reference = np.asarray(reference, dtype=float)
    if reference.shape != (len(history),) or not np.isfinite(reference).all() or np.any((reference <= 0)|(reference >= 1)):
        raise ValueError('Aligned interior R4 probabilities required')
    names = ['market_open_logit_diff', 'market_open_overround', 'market_open_available']
    if condition == 'latest':
        names += ['market_close_logit_diff', 'market_close_overround', 'market_close_available',
                  'market_movement_logit_diff', 'market_both_available']
    if market.columns.duplicated().any() or not set(names).issubset(market) or not np.isfinite(market[names].to_numpy()).all():
        raise ValueError('Unique, finite, complete quote columns required')
    for n in names:
        if n.endswith('available') and not market[n].isin((0, 1)).all():
            raise ValueError('Quote availability must be binary')
    close = market.market_close_available.eq(1).to_numpy() if condition == 'latest' else np.zeros(len(market), dtype=bool)
    quote = np.where(close, market['market_close_logit_diff'] if condition == 'latest' else 0, market.market_open_logit_diff)
    available = close | market.market_open_available.eq(1).to_numpy()
    overround = np.where(close, market['market_close_overround'] if condition == 'latest' else 0, market.market_open_overround)
    x = history.copy()
    x['quote_logit_diff'] = quote
    x['quote_absolute_logit'] = np.abs(quote)
    x['quote_overround'] = overround
    x['quote_available'] = available.astype(float)
    x['reference_logit_diff'] = logit(reference)
    x['reference_probability'] = reference  # Exact fallback, never a fitted predictor.
    if condition == 'latest':
        x['quote_movement_diff'] = market.market_movement_logit_diff
        x['quote_both_available'] = market.market_both_available
        x['quote_closing_available'] = close.astype(float)
    if x.columns.duplicated().any() or not np.isfinite(x.to_numpy()).all():
        raise ValueError('Invalid merged residual inputs')
    return x


def raw_probability(x):
    if not x.quote_available.isin((0, 1)).all():
        raise ValueError('Binary quote availability required')
    return np.where(x.quote_available.eq(1), expit(x.quote_logit_diff), x.reference_probability)


def candidate_specs(old_columns, event_columns, condition):
    market = ['quote_logit_diff', 'quote_absolute_logit', 'quote_overround', 'format_best_of_five',
              'surface_is_clay', 'surface_is_grass']
    if condition == 'latest':
        market += ['quote_movement_diff', 'quote_both_available', 'quote_closing_available']
    groups = {'price': market, 'history': list(dict.fromkeys(market+old_columns+['reference_logit_diff'])),
              'event': list(dict.fromkeys(market+old_columns+event_columns+['reference_logit_diff']))}
    specs = []
    for group, columns in groups.items():
        for c in (.003, .03):
            specs.append(dict(name=f'{group}_linear_C{c:g}', kind='linear', group=group,
                columns=[n for n in columns if n.endswith('_diff')], C=c))
        specs.append(dict(name=f'{group}_cat_d3', kind='catboost', group=group, columns=columns,
                          depth=3, iterations=300, learning_rate=.03, l2_leaf_reg=50))
    return specs


def offset_objective(beta, design, y, offset, C):
    eta = offset+design@beta
    objective = np.sum(np.logaddexp(0., eta)-y*eta)+.5*np.dot(beta, beta)/C
    gradient = design.T@(expit(eta)-y)+beta/C
    return objective/len(y), gradient/len(y)


class ResidualModel:
    def __init__(self, spec):
        self.spec = spec

    def _input(self, x):
        if x.empty or x.columns.duplicated().any() or not set(self.spec['columns']).issubset(x):
            raise ValueError('Unique nonempty frozen features required')
        xx = x[self.spec['columns']].to_numpy(dtype=float)
        if not np.isfinite(xx).all():
            raise ValueError('Finite residual features required')
        return xx

    def fit(self, x, rows):
        if not x.index.equals(rows.index) or rows.empty or not rows.season.isin((2021, 2022)).all():
            raise ValueError('Aligned 2021/22-only fitting rows required')
        if not rows.a_won.isin((0, 1)).all() or pd.to_datetime(rows.tourney_date).ge('2023-01-01').any():
            raise ValueError('Binary development-only training labels required')
        if not x.quote_available.eq(1).all():
            raise ValueError('Residual fitting only uses available-price rows')
        xx, yy, z = self._input(x), rows.a_won.to_numpy(), x.quote_logit_diff.to_numpy()
        if self.spec['kind'] == 'linear':
            self.scale_ = np.sqrt(np.mean(xx*xx, axis=0))
            self.scale_[self.scale_ < 1e-10] = 1.
            design = np.clip(xx/self.scale_, -8., 8.)
            result = minimize(offset_objective, np.zeros(design.shape[1]), args=(design, yy, z, self.spec['C']),
                method='L-BFGS-B', jac=True, options=dict(maxiter=1000, ftol=1e-12, gtol=1e-7))
            if not result.success:
                raise ValueError(f'Offset optimizer failed: {result.message}')
            self.beta_ = result.x
            self.optimization_ = dict(iterations=result.nit, gradient_norm=float(np.max(np.abs(result.jac))))
        else:
            from catboost import CatBoostClassifier, Pool
            mirrored = self._input(mirror_input(x))
            pool = Pool(np.concatenate([xx, mirrored]), np.r_[yy, 1-yy], baseline=np.r_[z, -z],
                        weight=np.full(2*len(yy), .5))
            self.estimator_ = CatBoostClassifier(depth=self.spec['depth'], iterations=self.spec['iterations'],
                learning_rate=self.spec['learning_rate'], l2_leaf_reg=self.spec['l2_leaf_reg'],
                loss_function='Logloss', random_seed=158, thread_count=2, verbose=False,
                allow_writing_files=False, boost_from_average=False)
            self.estimator_.fit(pool)
        self.training_rows_ = len(rows)
        return self

    def residual(self, x):
        xx = self._input(x)
        if self.spec['kind'] == 'linear':
            return np.clip(xx/self.scale_, -8., 8.)@self.beta_
        # Predict WITHOUT a Pool baseline: add the exact double-precision
        # price offset once, outside CatBoost's float32 baseline storage.
        a = self.estimator_.predict(xx, prediction_type='RawFormulaVal')
        b = self.estimator_.predict(self._input(mirror_input(x)), prediction_type='RawFormulaVal')
        return .5*(a-b)

    def predict_proba(self, x, alpha=1.):
        if alpha not in ALPHAS:
            raise ValueError('Only frozen residual scales allowed')
        candidate = np.clip(expit(x.quote_logit_diff.to_numpy()+alpha*self.residual(x)), 1e-8, 1-1e-8)
        return np.where(x.quote_available.eq(1), candidate, x.reference_probability)


def predict_bundle(bundle, x):
    if x.columns.duplicated().any() or not set(bundle['feature_names']).issubset(x):
        raise ValueError('Frozen inference schema missing')
    xx = x[bundle['feature_names']]
    if not np.isfinite(xx.to_numpy()).all() or not xx.reference_probability.between(0, 1, inclusive='neither').all():
        raise ValueError('Finite inputs and valid reference probability required')
    p = {REFERENCE: raw_probability(xx)}
    for name, model in bundle['models'].items():
        for alpha in ALPHAS:
            p[f'{name}_alpha{alpha:g}'] = model.predict_proba(xx, alpha)
    return pd.DataFrame(p, index=x.index)


def select_candidate(y, predictions, mask):
    y, mask = np.asarray(y), np.asarray(mask, dtype=bool)
    if y.shape != mask.shape or not mask.any() or REFERENCE not in predictions:
        raise ValueError('Aligned nonempty historical selection required')
    limit = mean_loss(y[mask], np.asarray(predictions[REFERENCE])[mask])+.005
    allowed = [n for n, p in predictions.items() if mean_loss(y[mask], np.asarray(p)[mask]) <= limit]
    name = min(allowed, key=lambda n: (-accuracy(y[mask], np.asarray(predictions[n])[mask]),
                                      mean_loss(y[mask], np.asarray(predictions[n])[mask]), n))
    return dict(choice=name, selection_rows=int(mask.sum()), accuracy=accuracy(y[mask], np.asarray(predictions[name])[mask]),
        mean_log_loss=mean_loss(y[mask], np.asarray(predictions[name])[mask]), loss_limit=limit, eligible_models=len(allowed))


def choose_procedure(y, predictions, folds):
    early, later = np.asarray(folds) == '2021H2', np.asarray(folds) == '2022'
    if not early.any() or not later.any():
        raise ValueError('Both fixed forward folds required')
    chosen = select_candidate(y, predictions, early)
    reference = np.asarray(predictions[REFERENCE])[later]
    candidate = np.asarray(predictions[chosen['choice']])[later]
    y = np.asarray(y)[later]
    gain, loss_change = accuracy(y, candidate)-accuracy(y, reference), mean_loss(y, candidate)-mean_loss(y, reference)
    return dict(early_selection=chosen, accuracy_improvement=gain, loss_change=loss_change,
        procedure='adopt_residual' if gain >= .005-1e-12 and loss_change <= .005+1e-12 else 'retain_raw_market',
        reference_2022_accuracy=accuracy(y, reference), candidate_2022_accuracy=accuracy(y, candidate),
        reference_2022_loss=mean_loss(y, reference), candidate_2022_loss=mean_loss(y, candidate))


def score_table(rows, predictions):
    return pd.DataFrame([dict(model=n, **score_probabilities(zip(rows.match_id, rows.a_won), zip(rows.match_id, p)),
        brier=float(np.mean((rows.a_won.to_numpy()-np.asarray(p))**2))) for n,p in predictions.items()])


def audited_file(directory, name):
    audit = json.loads((directory/'artifact_audit.json').read_text())
    if sha256(directory/name) != audit['artifact_sha256'][name]:
        raise ValueError(f'Audited input changed: {directory/name}')
    return directory/name


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT/'results/round8/market_residual')
    parser.add_argument('--predict-features', type=Path)
    parser.add_argument('--condition', choices=CONDITIONS, default='opening')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.predict_features:
        out = args.output/args.condition
        p = predict_bundle(joblib.load(out/'frozen_models.joblib'), pd.read_csv(args.predict_features))
        p.to_csv(out/'inference_predictions.csv', index=False)
        return
    event_dir, market_dir, old_dir = ROOT/'results/round7/event', ROOT/'results/round5/market', ROOT/'results/round5'
    targets_path = ROOT/'results/round3/development_targets.csv'
    all_rows = pd.read_csv(targets_path)
    rows = all_rows[all_rows.season.isin((2021, 2022, 2023))].copy()
    history_path = audited_file(event_dir, 'development_features.csv')
    market_path = audited_file(market_dir, 'features.csv')
    link_path = audited_file(market_dir, 'linked_matches.csv')
    history = pd.read_csv(history_path).loc[rows.index]
    market = pd.read_csv(market_path).set_index('match_id').loc[rows.match_id].set_axis(rows.index)
    links = pd.read_csv(link_path, usecols=['match_id', 'market_match_utc']).set_index('match_id').loc[rows.match_id]
    utc = pd.Series(pd.to_datetime(links.market_match_utc, utc=True, errors='coerce').to_numpy(), index=rows.index)
    old_oof = pd.read_csv(audited_file(old_dir, 'oof_predictions.csv'))
    old_val = pd.read_csv(audited_file(old_dir, 'validation_predictions.csv'))
    prior = pd.concat([old_oof, old_val]).set_index('match_id').loc[rows.match_id]
    if not prior[['season', 'a_won']].reset_index(drop=True).equals(rows[['season', 'a_won']].reset_index(drop=True)):
        raise ValueError('Prior prediction identity changed')
    reference = prior.previous_probability_recommendation.to_numpy()
    old_columns = list(pd.read_csv(old_dir/'development_features.csv', nrows=0))
    event_columns = [n for n in history if n not in old_columns]
    if (len(old_columns), len(event_columns)) != (326, 27):
        raise ValueError('Frozen historical/event schema changed')
    for condition in CONDITIONS:
        out = args.output/condition
        out.mkdir(parents=True, exist_ok=True)
        x = build_inputs(history, market, reference, condition)
        specs = candidate_specs(old_columns, event_columns, condition)
        folds = {fold: market_masks(rows, fold) for fold in ('2021H2', '2022')}
        all_valid = rows.index.isin(np.concatenate([rows.index[va] for _,va in folds.values()]))
        orows = rows[all_valid]
        oof = {REFERENCE: raw_probability(x[all_valid])}
        registry = dict(round=8, branch='market_residual', condition=condition, candidates=specs,
            alpha=ALPHAS, fitted_configurations=9, selection_entries=28, feature_names=list(x),
            source_sha256={str(p.relative_to(ROOT)):sha256(p) for p in (targets_path,history_path,market_path,link_path)},
            protocol_sha256=sha256(ROOT/'docs/research/round8_market_residual.md'), runner_sha256=sha256(Path(__file__)),
            final_test_evaluated=False, exploratory_after_2023=True, decision_threshold=.5)
        (out/'search_registry.json').write_text(json.dumps(registry,indent=2)+'\n')
        x.to_csv(out/'development_features.csv',index=False)
        populations=[]
        for number,spec in enumerate(specs,1):
            values = {a:pd.Series(index=orows.index,dtype=float) for a in ALPHAS}
            for fold,(tr,va) in folds.items():
                cutoff = pd.to_datetime(rows.loc[va,'tourney_date']).min()-pd.Timedelta(days=21)
                eligible = tr & x.quote_available.eq(1) & utc.le(cutoff.tz_localize('UTC'))
                model = ResidualModel(spec).fit(x[eligible],rows[eligible])
                populations.append(dict(model=spec['name'],fold=fold,fit_rows=int(eligible.sum()),cutoff=str(cutoff),
                    last_market_utc=str(utc[eligible].max()),validation_rows=int(va.sum())))
                for a in ALPHAS:
                    values[a].loc[rows.index[va]]=model.predict_proba(x[va],a)
            for a in ALPHAS:
                oof[f'{spec["name"]}_alpha{a:g}']=values[a].to_numpy()
            print(f'{condition} [{number}/9] {spec["name"]}: CV accuracy={accuracy(orows.a_won,values[1.].to_numpy()):.5f}',flush=True)
        fold_labels=np.where(orows.season.eq(2021),'2021H2','2022')
        gate=choose_procedure(orows.a_won,oof,fold_labels)
        pooled=select_candidate(orows.a_won,oof,np.ones(len(orows),dtype=bool))
        recommendation=pooled['choice'] if gate['procedure']=='adopt_residual' else REFERENCE
        frozen=dict(condition=condition,recommendation=recommendation,procedure_gate=gate,pooled_selection=pooled,
            fit_end_season=2022,decision_threshold=.5,exploratory_after_2023=True,final_test_evaluated=False)
        (out/'frozen_selection.json').write_text(json.dumps(frozen,indent=2)+'\n')
        print('Frozen',condition,json.dumps(frozen),flush=True)
        pd.concat([orows[['match_id','season','a_won']].reset_index(drop=True),pd.DataFrame(oof)],axis=1).to_csv(out/'oof_predictions.csv',index=False)
        cv=[]
        for fold in ('2021H2','2022','pooled'):
            mask=np.ones(len(orows),dtype=bool) if fold=='pooled' else fold_labels==fold
            table=score_table(orows[mask],{n:p[mask] for n,p in oof.items()});table['fold']=fold;cv.append(table)
        pd.concat(cv).to_csv(out/'temporal_cv.csv',index=False)
        va=rows.season.eq(2023)
        cutoff=pd.to_datetime(rows.loc[va,'tourney_date']).min()-pd.Timedelta(days=21)
        tr=rows.season.le(2022)&pd.to_datetime(rows.tourney_date).le(cutoff)&x.quote_available.eq(1)&utc.le(cutoff.tz_localize('UTC'))
        models={s['name']:ResidualModel(s).fit(x[tr],rows[tr]) for s in specs}
        bundle=dict(models=models,feature_names=list(x),specs=specs,frozen=frozen)
        joblib.dump(bundle,out/'frozen_models.joblib')
        p=predict_bundle(bundle,x[va]);vr=rows[va]
        table=score_table(vr,{n:p[n].to_numpy() for n in p});table['recommended']=table.model.eq(recommendation)
        table.to_csv(out/'validation_comparison.csv',index=False)
        pd.concat([vr[['match_id','season','tourney_date','a_won']].reset_index(drop=True),p.reset_index(drop=True)],axis=1).to_csv(out/'validation_predictions.csv',index=False)
        populations.append(dict(model='all_final',fold='2023',fit_rows=int(tr.sum()),cutoff=str(cutoff),last_market_utc=str(utc[tr].max()),validation_rows=int(va.sum())))
        pd.DataFrame(populations).to_csv(out/'fit_populations.csv',index=False)
        comparisons={n:dict(accuracy=accuracy_interval(vr,p[REFERENCE].to_numpy(),p[n].to_numpy()),
            log_loss=cluster_interval(vr,p[REFERENCE].to_numpy(),p[n].to_numpy())) for n in p}
        summary=dict(condition=condition,recommendation=recommendation,pooled_choice=pooled['choice'],fitted_configurations=9,
            training_rows=int(tr.sum()),validation_matches=len(vr),available_quotes=int(x.loc[va,'quote_available'].sum()),
            fallback_rows=int(x.loc[va,'quote_available'].eq(0).sum()),comparisons_vs_raw_market=comparisons,
            correct_required_for_70=1812,final_test_evaluated=False,exploratory_after_2023=True)
        (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
        print(table.sort_values(['accuracy','mean_log_loss'],ascending=[False,True]).to_string(index=False),flush=True)

if __name__=='__main__':
    from scripts.train_market_residual_round8 import main as canonical_main
    with threadpool_limits(limits=2):
        canonical_main()
