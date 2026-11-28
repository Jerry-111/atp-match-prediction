"""Historical opening-market graphs, with strict actual-UTC T-21 prefixes.

No current quote, source outcome, closing quote or post-2023 raw file is read.
Natural positive logits and a BO3 reference scale adapt Penn et al.'s graph.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix, eye
from scipy.sparse.csgraph import connected_components
from scipy.sparse.linalg import spsolve
from scipy.special import expit, logit

from scripts.market_data import de_vig
from scripts.multilevel_features import _player
from scripts.train_distillation_round7 import LINK_COLUMNS, SOURCE_COLUMNS, _id, validate_rows
from scripts.train_models import ROOT, sha256

YEARS = (2021, 2022, 2023)
RIDGE = .001
HISTORY_COLUMNS = ['match_id', 'source_match_id', 'source_season', 'match_utc',
                   'player_a_id', 'player_b_id', 'surface', 'best_of', 'opening_logit_diff']


def _atp_id(value):
    """ATP codes are strings; integer synthetic/Sackmann IDs are also stable."""
    text = _player(value)
    if isinstance(value, (int, float, np.integer, np.floating)):
        if not np.isfinite(value) or value <= 0 or value != int(value):
            raise ValueError('Invalid numeric ATP identity')
        return str(int(value))
    return text


def graph_specs():
    return [dict(name=f'market_graph_h{h}_surface{s}',
                 half_life=None if h == 'inf' else float(h), other_surface=.25 if s else 1.)
            for h in ('90', '365', 'inf') for s in (0, 1)]


def series_probability(q, best_of):
    q = np.asarray(q, dtype=float)
    best_of = np.asarray(best_of)
    if not np.isfinite(q).all() or np.any((q < 0) | (q > 1)) or not np.isin(best_of, [3, 5]).all():
        raise ValueError('Finite unit set probabilities and BO3/BO5 formats required')
    # Complement evaluation above .5 avoids cancellation at the boundaries.
    small = np.minimum(q, 1-q)
    p3 = small**2 * (3-2*small)
    p5 = small**3 * (10-15*small+6*small**2)
    p = np.where(best_of == 3, p3, p5)
    return np.where(q > .5, 1-p, p)


def set_probability(p, best_of):
    p = np.asarray(p, dtype=float)
    if not np.isfinite(p).all() or np.any((p < 0) | (p > 1)) or not np.isin(best_of, [3, 5]).all():
        raise ValueError('Finite unit match probabilities and BO3/BO5 formats required')
    small = np.minimum(p, 1-p)
    lo, hi = np.zeros_like(p), np.full_like(p, .5)
    for _ in range(55):
        middle = (lo+hi)/2
        below = series_probability(middle, best_of) < small
        lo, hi = np.where(below, middle, lo), np.where(below, hi, middle)
    q = (lo+hi)/2
    return np.where(p > .5, 1-q, q)


def format_probability(p, from_best_of, to_best_of):
    return series_probability(set_probability(p, from_best_of), to_best_of)


def build_opening_history(rows, linked, source):
    """Revalidate previously frozen identities; source outcomes are ignored."""
    validate_rows(rows)
    if not rows.best_of.isin([3, 5]).all():
        raise ValueError('Verified preknown BO3/BO5 required')
    links = linked.loc[pd.to_numeric(linked.season, errors='coerce').isin(YEARS), LINK_COLUMNS].copy()
    src = source.loc[pd.to_numeric(source.source_season, errors='coerce').isin(YEARS),
                     SOURCE_COLUMNS+['source_season']].copy()
    # Reject actual final-year rows before looking at their identities/prices.
    times = pd.to_datetime(src.date, utc=True, errors='coerce')
    src = src[times.dt.year.le(2023)].copy()
    if links.match_id.duplicated().any() or src.match_id.duplicated().any():
        raise ValueError('Duplicate fixed target/source links')
    targets = rows[rows.season.isin(YEARS)]
    links = links[links.match_id.isin(targets.match_id)].set_index('match_id')
    if set(links.index) != set(targets.match_id):
        raise ValueError('Fixed links omit eligible targets')
    src.index = src.match_id.map(_id)
    accepted = links[links.status.eq('matched')]
    if accepted.market_match_id.map(_id).duplicated().any():
        raise ValueError('Source matches cannot be reused')
    source_to_atp, atp_to_source, history = {}, {}, []
    for _, row in targets.iterrows():
        link = links.loc[row.match_id]
        if int(link.season) != int(row.season) or str(link.target_event_id) != str(row.tourney_id):
            raise ValueError('Fixed target metadata changed')
        if link.status != 'matched':
            continue
        identifier = _id(link.market_match_id)
        if identifier not in src.index:
            raise ValueError('Fixed source match missing from permitted raw years')
        raw = src.loc[identifier]
        a, b = _id(link.market_player_a_id), _id(link.market_player_b_id)
        first, second = _id(raw.joueur1_id), _id(raw.joueur2_id)
        stamp = pd.to_datetime(raw.date, utc=True, errors='coerce')
        year = int(raw.source_season)
        compatible = pd.notna(stamp) and (stamp.year == year or (
            stamp.year == year-1 and stamp.month == 12 and stamp.day >= 25))
        if (year != int(row.season) or not compatible or stamp.year > 2023
                or stamp != pd.to_datetime(link.market_match_utc, utc=True)
                or _id(raw.tournoi_id) != _id(link.market_event_id)
                or a == b or first == second or {a, b} != {first, second}):
            raise ValueError('Fixed source identity/event/time changed')
        for market_id, atp_id in ((a, _atp_id(row.player_a_id)), (b, _atp_id(row.player_b_id))):
            if (source_to_atp.get(market_id, atp_id) != atp_id
                    or atp_to_source.get(atp_id, market_id) != market_id):
                raise ValueError('Source/player identity is not one-to-one')
            source_to_atp[market_id], atp_to_source[atp_id] = atp_id, market_id
        odds = (raw.cote1_ouverture, raw.cote2_ouverture)
        oa, ob = odds if a == first else odds[::-1]
        z, _, available = de_vig(oa, ob)
        if available:
            history.append(dict(match_id=row.match_id, source_match_id=identifier,
                source_season=year, match_utc=stamp, player_a_id=_atp_id(row.player_a_id),
                player_b_id=_atp_id(row.player_b_id), surface=str(row.surface),
                best_of=int(row.best_of), opening_logit_diff=float(z)))
    return pd.DataFrame(history, columns=HISTORY_COLUMNS).sort_values(
        ['match_utc', 'source_match_id'], kind='stable').reset_index(drop=True)


def load_opening_history(rows, *, cache=ROOT/'data/raw/market',
                         manifest_path=ROOT/'data/market-manifest.json',
                         linkage_path=ROOT/'results/round5/market/linked_matches.csv',
                         audit_path=ROOT/'results/round5/market/artifact_audit.json'):
    validate_rows(rows)
    audit = json.loads(Path(audit_path).read_text())
    if sha256(linkage_path) != audit['artifact_sha256']['linked_matches.csv']:
        raise ValueError('Audited source links changed')
    if sha256(manifest_path) != audit['source_manifest_sha256']:
        raise ValueError('Pinned source manifest changed')
    entries = {int(e['season']): e for e in json.loads(Path(manifest_path).read_text())['files']}
    linked = pd.read_csv(linkage_path, usecols=LINK_COLUMNS)
    frames = []
    for year in YEARS:
        path = Path(cache)/f'{year}.csv'
        if sha256(path) != entries[year]['sha256']:
            raise ValueError('Pinned source file changed')
        frame = pd.read_csv(path, sep=';', usecols=SOURCE_COLUMNS)
        if len(frame) != entries[year]['rows']:
            raise ValueError('Pinned source row count changed')
        frames.append(frame.assign(source_season=year))
    return build_opening_history(rows, linked, pd.concat(frames, ignore_index=True))


def fit_graph(a, b, z, weights, ridge=RIDGE):
    """Exact minimizer of sum w*((r_a-r_b)-z)^2 + ridge*sum r^2."""
    a, b, z, weights = [np.asarray(v) for v in (a, b, z, weights)]
    if not (a.ndim == b.ndim == z.ndim == weights.ndim == 1 and len(a) == len(b) == len(z) == len(weights)):
        raise ValueError('Aligned graph vectors required')
    if (not np.isfinite(z).all() or not np.isfinite(weights).all() or (weights <= 0).any()
            or (a == b).any() or not np.isfinite(ridge) or ridge <= 0):
        raise ValueError('Finite edges, positive weights/ridge and distinct players required')
    nodes = np.unique(np.r_[a, b])
    if not len(nodes):
        return nodes, np.array([]), np.array([], dtype=int), np.array([])
    ia, ib = np.searchsorted(nodes, a), np.searchsorted(nodes, b)
    n = len(nodes)
    lap = coo_matrix((np.r_[weights, weights, -weights, -weights],
        (np.r_[ia, ib, ia, ib], np.r_[ia, ib, ib, ia])), shape=(n, n)).tocsr()
    rhs = np.bincount(ia, weights* z, minlength=n)-np.bincount(ib, weights*z, minlength=n)
    strength = spsolve(lap + ridge*eye(n, format='csr'), rhs)
    _, components = connected_components(lap, directed=False)
    count = np.bincount(np.r_[ia, ib], minlength=n)
    return nodes, strength, components, count


def build_market_graph_features(targets, history, *, settings=None, locked_before=None):
    """Pure temporal graph queries; all *_diff odd and all other fields even."""
    dates = validate_rows(targets)
    if not targets.best_of.isin([3, 5]).all() or targets.index.duplicated().any():
        raise ValueError('Unique indices and BO3/BO5 required')
    settings = graph_specs() if settings is None else settings
    if len({s['name'] for s in settings}) != len(settings):
        raise ValueError('Unique graph setting names required')
    for s in settings:
        if (s['half_life'] is not None and (not np.isfinite(s['half_life']) or s['half_life'] <= 0)) or not 0 < s['other_surface'] <= 1:
            raise ValueError('Positive decay/surface weights required')
    # Season guard first, then actual-date guard, then parse allowed edges only.
    h = history[pd.to_numeric(history.source_season, errors='coerce').isin(YEARS)].copy()
    h['match_utc'] = pd.to_datetime(h.match_utc, utc=True, errors='coerce')
    h = h[h.match_utc.dt.year.le(2023)].copy()
    if locked_before is not None:
        h = h[h.match_utc.lt(pd.Timestamp(locked_before, tz='UTC'))].copy()
    # Also prevent a supplied later price from affecting an earlier-only query.
    cutoffs = pd.to_datetime(dates, utc=True)-pd.Timedelta(days=21)
    h = h[h.match_utc.le(cutoffs.max())].copy()
    if h.source_match_id.duplicated().any() or h.match_id.duplicated().any():
        raise ValueError('History source IDs cannot repeat')
    compatible = h.match_utc.dt.year.eq(h.source_season) | (h.match_utc.dt.year.eq(h.source_season-1)
        & h.match_utc.dt.month.eq(12) & h.match_utc.dt.day.ge(25))
    if not compatible.all() or not h.best_of.isin([3, 5]).all():
        raise ValueError('History date/season or format mismatch')
    for c in ('player_a_id', 'player_b_id'):
        h[c] = h[c].map(_atp_id)
    if (h.player_a_id == h.player_b_id).any():
        raise ValueError('Self edge forbidden')
    h = h.sort_values(['match_utc', 'source_match_id'], kind='stable')
    values = h.opening_logit_diff.to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError('Finite allowed quote logits required')
    h['bo3_logit'] = logit(np.clip(format_probability(expit(values), h.best_of.to_numpy(), 3), 1e-12, 1-1e-12))
    columns = [s['name']+'_diff' for s in settings]+['market_graph_available',
        'market_graph_log_count_diff', 'market_graph_log_count_sum']
    result = pd.DataFrame(0., index=targets.index, columns=columns)
    for cutoff in sorted(cutoffs.unique()):
        loc = targets.index[cutoffs.eq(cutoff)]
        past = h[h.match_utc.le(cutoff)]
        if past.empty:
            continue
        days = (cutoff-past.match_utc).dt.total_seconds().to_numpy()/86400
        a, b, z = past.player_a_id.to_numpy(), past.player_b_id.to_numpy(), past.bo3_logit.to_numpy()
        for spec in settings:
            decay = np.ones(len(past)) if spec['half_life'] is None else 2.**(-days/spec['half_life'])
            groups = [(None, loc)] if spec['other_surface'] == 1 else targets.loc[loc].groupby('surface', sort=True).groups.items()
            for surface, idx in groups:
                weight = decay if surface is None else decay*np.where(past.surface.eq(surface), 1., spec['other_surface'])
                nodes, rating, components, count = fit_graph(a, b, z, weight)
                lookup = {n: i for i, n in enumerate(nodes)}
                group = targets.loc[idx]
                ai = group.player_a_id.map(_atp_id).map(lookup).fillna(-1).to_numpy(dtype=int)
                bi = group.player_b_id.map(_atp_id).map(lookup).fillna(-1).to_numpy(dtype=int)
                known = (ai >= 0) & (bi >= 0)
                valid = known & (components[ai] == components[bi])
                selected = group.index[valid]
                ai, bi = ai[valid], bi[valid]
                p = format_probability(expit(rating[ai]-rating[bi]), 3, group.best_of.to_numpy()[valid])
                result.loc[selected, spec['name']+'_diff'] = logit(np.clip(p, 1e-12, 1-1e-12))
                result.loc[selected, 'market_graph_available'] = 1.
                result.loc[selected, 'market_graph_log_count_diff'] = np.log1p(count[ai])-np.log1p(count[bi])
                result.loc[selected, 'market_graph_log_count_sum'] = np.log1p(count[ai])+np.log1p(count[bi])
    if not np.isfinite(result.to_numpy()).all():
        raise ValueError('Nonfinite graph features')
    result.attrs.update(settings=settings, history_lag_days=21, source_years=YEARS,
        source_outcomes_read=False, cutoff='source UTC <= target date UTC midnight - 21 days',
        odd_features=[c for c in result if c.endswith('_diff')],
        even_features=[c for c in result if not c.endswith('_diff')])
    return result
