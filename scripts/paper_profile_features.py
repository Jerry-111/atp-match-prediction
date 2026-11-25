"""Pre-match age/ranking-point features motivated by Buhamra et al. (2025).

Implements the paper's age-distance transformations, not its whole fitted model.
Only known pre-match attributes are read from the target record; target scores,
point statistics and outcomes are never used as predictors. Missing ages use
the fixed reference30 and missing points use0, with explicit indicators.

Audited mixed-date events in model_event_metadata have verified weekly ranks,
but no verified ranking-point or age corrections. Their raw points and ages
are withheld for both players rather than combined with the corrected ranks.
"""
from math import isfinite, log1p

import pandas as pd

from scripts.advanced_features import _date
from scripts.baseline import DEFAULT_CACHE
from scripts.model_event_metadata import EVENT_METADATA, corrected_tour_row
from scripts.serve_features import _annual_rows, development_entries

PAPER_DOI = '10.1177/22150218251338954'
PAPER_PREPRINT_URL = 'https://arxiv.org/abs/2502.01613v2'
POINT_SNAPSHOT_POLICY = (
    'Ranking points are set missing for all EVENT_METADATA audited events: '
    'verified pre-event rank corrections do not supply verified ranking points; '
    'raw points can mix weekly snapshots within the same event. '
    'Missing points use fixed zero plus explicit missing indicators.'
)
AGE_SNAPSHOT_POLICY = (
    'Ages are set missing for all EVENT_METADATA audited events: verified '
    'pre-event rank corrections do not supply verified ages or birth dates; '
    'raw ages vary with the mixed source event dates. '
    'Missing ages use fixed reference 30 plus explicit missing indicators.'
)


def _number(value, fallback, valid):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return fallback, 1.
    return (number, 0.) if isfinite(number) and valid(number) else (fallback, 1.)


def profile_row(target, raw):
    suppress_profile = raw['tourney_id'] in EVENT_METADATA
    raw = corrected_tour_row(raw)
    a, b = str(target['player_a_id']), str(target['player_b_id'])
    keys = {str(raw['winner_id']): 'winner', str(raw['loser_id']): 'loser'}
    if a == b or set(keys) != {a,b} or _date(target['tourney_date']) != _date(raw['tourney_date']):
        raise ValueError('Profile source identity/date mismatch')
    values = []
    for player in (a,b):
        prefix = keys[player]
        age, missing_age = _number(
            None if suppress_profile else raw.get(prefix+'_age'),
            30., lambda n: 10 <= n <= 80)
        points, missing_points = _number(
            None if suppress_profile else raw.get(prefix+'_rank_points'),
            0., lambda n: n >= 0)
        values.append(dict(age=age, age30=abs(age-30), age_interval=max(28-age,0,age-32),
            log_points=log1p(points), points=points/1000,
            age_missing=missing_age, points_missing=missing_points))
    aa,bb = values
    out = {'profile_'+k+'_diff': aa[k]-bb[k] for k in aa}
    out.update(profile_age_mean=(aa['age']+bb['age'])/2,
               profile_log_points_mean=(aa['log_points']+bb['log_points'])/2,
               profile_age_missing_sum=aa['age_missing']+bb['age_missing'],
               profile_points_missing_sum=aa['points_missing']+bb['points_missing'])
    return out


def build_paper_profile_features(targets, inventory=None, *, cache_dir=DEFAULT_CACHE, raw_rows=None):
    if targets.empty or not pd.to_numeric(targets.season).between(2009,2023).all():
        raise ValueError('Paper development targets must be2009..2023; final holdout locked')
    if targets.match_id.duplicated().any():
        raise ValueError('Profile targets must have unique match IDs')
    expected = {str(r['match_id']): r for r in targets.to_dict('records')}
    entries = development_entries() if raw_rows is None else []
    iterator = _annual_rows(cache_dir,entries) if raw_rows is None else raw_rows
    found, suppressed_events, suppressed_appearances = {}, set(), 0
    for season,raw in iterator:
        if not 2009 <= int(season) <= 2023:
            continue
        key = f"{raw['tourney_id']}:{int(raw['match_num'])}"
        if key not in expected:
            continue
        if key in found or int(season)!=int(expected[key]['season']):
            raise ValueError('Duplicate/mismatched profile source')
        found[key] = profile_row(expected[key],raw)
        if raw['tourney_id'] in EVENT_METADATA:
            suppressed_events.add(raw['tourney_id'])
            suppressed_appearances += 2
    if expected.keys()-found.keys():
        raise ValueError('Profile source omitted target rows')
    out = pd.DataFrame([found[str(k)] for k in targets.match_id],index=targets.index)
    out.attrs.update(paper_doi=PAPER_DOI,paper_preprint_url=PAPER_PREPRINT_URL,
        version=3,age_reference=30,age_interval=[28,32],
        source_files=entries,information_time='Current known pre-match attributes; historical outcome lag remains21days',
        points_snapshot_policy=POINT_SNAPSHOT_POLICY,
        points_snapshot_audit_source='scripts/model_event_metadata.py:EVENT_METADATA',
        points_snapshot_masked_events=sorted(suppressed_events),
        points_snapshot_masked_appearances=suppressed_appearances,
        age_snapshot_policy=AGE_SNAPSHOT_POLICY,
        age_snapshot_masked_events=sorted(suppressed_events),
        age_snapshot_masked_appearances=suppressed_appearances,
        missing_age_appearances=int(out.profile_age_missing_sum.sum()),
        missing_points_appearances=int(out.profile_points_missing_sum.sum()))
    return out
