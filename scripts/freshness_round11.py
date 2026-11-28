"""Bounded whole-edition information updates using the unchanged Elo engine.

Only a separately reviewed, hash-bound ledger may authorize extra histories.
The baseline remains inclusive T-21. Same-date matches are unordered batches.
No current target result or target technical statistic is read by this module.
"""
from __future__ import annotations

from datetime import date, timedelta
from collections import defaultdict
import hashlib
import json

import numpy as np
import pandas as pd

from scripts.advanced_features import HistoricalState, _date, elo_probability
from scripts.multilevel_features import _season_date

COLUMNS = ['log_rank_diff', 'elo_k32_diff', 'shrunk_surface_elo_k32_diff']
ELO_COLUMNS = COLUMNS[1:]
VERSION = 'bounded-completed-edition-elo-v1'
TARGET_META = ['match_id', 'tourney_id', 'season', 'tourney_date', 'surface', 'player_a_id', 'player_b_id']


def _dates_and_ids(rows, minimum_season, maximum_season):
    seasons = pd.to_numeric(rows.season, errors='coerce')
    if (rows.empty or rows.match_id.duplicated().any() or not rows.index.is_unique
            or not seasons.isin(range(minimum_season, maximum_season + 1)).all()):
        raise ValueError('Unique development rows and permitted integer seasons required')
    days = []
    for r in rows[TARGET_META].to_dict('records'):
        day = _date(r['tourney_date'])
        _season_date(day, int(r['season']))
        if day.year > maximum_season or not str(r['match_id']).startswith(str(r['tourney_id']) + ':'):
            raise ValueError('Date/edition identity mismatch or sealed year')
        if str(r['player_a_id']) == str(r['player_b_id']):
            raise ValueError('A player cannot oppose themself')
        if str(r['surface']) not in ('Hard', 'Clay', 'Grass'):
            raise ValueError('Unexpected completed Tour surface')
        days.append(day)
    return pd.Series(days, index=rows.index)


def tour_history(inventory):
    """Filter sealed seasons before interpreting outcomes or other fields."""
    seasons = pd.to_numeric(inventory.season, errors='coerce')
    history = inventory.loc[inventory.source.eq('tour') & seasons.isin(range(2006, 2024))].copy()
    # Published inventory IDs have the source namespace; target/ledger IDs do
    # not. Strip exactly one known prefix after source/year filtering and keep
    # the original mapping. A second or foreign prefix fails identity checks.
    history['inventory_match_id'] = history.match_id.astype(str)
    history['match_id'] = history.inventory_match_id.map(
        lambda value: value[len('tour:'):] if value.startswith('tour:') else value)
    days = _dates_and_ids(history, 2006, 2023)
    if not pd.to_numeric(history.a_won, errors='coerce').isin((0, 1)).all():
        raise ValueError('History outcomes must be binary')
    return history, days


def rating_inputs(state, row, latest):
    """The original overall K32 and n/(n+20) surface-shrunk K32 inputs."""
    a, b, surface = str(row['player_a_id']), str(row['player_b_id']), str(row['surface'])
    ratings = [state.ratings[32][p].at(latest) for p in (a, b)]
    surface_ratings = [state.surface_ratings[32][p, surface].at(latest) for p in (a, b)]
    counts = [state.surface[p, surface].summary(date.min, latest)[0] for p in (a, b)]
    shrunk = [ratings[i] + counts[i] / (counts[i] + 20.0) * (surface_ratings[i] - ratings[i])
              for i in range(2)]
    return np.array([ratings[0] - ratings[1], shrunk[0] - shrunk[1]])


def validate_pairs(pairs, targets, target_days, history, history_days):
    """Require the entire published source edition and entire target edition.

    Calendar evidence is adjudicated upstream; this function verifies its
    hash-bound identity scope and refuses partially selected source matches.
    """
    if not isinstance(pairs, list):
        raise ValueError('Qualified pairs must be a list')
    by_target, seen = defaultdict(set), set()
    hgroups = {str(k): set(g.match_id.astype(str)) for k, g in history.groupby('tourney_id')}
    tgroups = {str(k): set(g.match_id.astype(str)) for k, g in targets.groupby('tourney_id')}
    for pair in pairs:
        source, target = str(pair['source_tourney_id']), str(pair['target_tourney_id'])
        key = source, target
        if (pair.get('admitted_for_prediction') is not True or key in seen or source == target
                or source not in hgroups or target not in tgroups):
            raise ValueError('Unapproved, duplicate, same-edition, absent or sealed pair')
        seen.add(key)
        for field, expected in [('source_match_ids', hgroups[source]), ('target_match_ids', tgroups[target])]:
            values = pair.get(field)
            if (not isinstance(values, list) or len(values) != len(set(values))
                    or set(values) != expected):
                raise ValueError('Ledger must identify a whole published edition: ' + field)
        tm = targets.tourney_id.astype(str).eq(target)
        hm = history.tourney_id.astype(str).eq(source)
        tdays, hdays = target_days[tm].unique(), history_days[hm].unique()
        if len(tdays) != 1 or len(hdays) != 1:
            raise ValueError('Edition must have one canonical nominal date')
        if not tdays[0] - timedelta(days=21) < hdays[0] <= tdays[0]:
            raise ValueError('Extra edition must lie after baseline cutoff and at/before nominal target')
        if any(k not in pair or pair[k] is None for k in ('source_upper_utc', 'target_lower_utc')):
            raise ValueError('Explicit source and target UTC bounds are required')
        upper = pd.to_datetime(pair['source_upper_utc'], utc=True, errors='coerce')
        lower = pd.to_datetime(pair['target_lower_utc'], utc=True, errors='coerce')
        if (pd.isna(upper) or pd.isna(lower) or not upper < lower
                or pd.Timestamp(pair['source_upper_utc']).tzinfo is None
                or pd.Timestamp(pair['target_lower_utc']).tzinfo is None):
            raise ValueError('Ledger temporal bounds must strictly separate and specify timezone')
        by_target[target].update(hgroups[source])
    return dict(by_target)


def build_freshness_inputs(targets, inventory, stale_features, pairs):
    """Return fresh 3-column input, per-target coverage, and audit metadata.

    All rows retain their original index/order. Stale input values are copied
    exactly on no-extra rows and on rating coordinates unchanged by additions.
    The old engine is replayed on prefix UNION whole certified editions for
    each distinct target edition, never with a globally shared mutable state.
    """
    target_days = _dates_and_ids(targets, 2009, 2023)
    if (not stale_features.index.equals(targets.index) or stale_features.columns.duplicated().any()
            or not set(COLUMNS).issubset(stale_features.columns)):
        raise ValueError('Stale inputs must align exactly with target index and columns')
    stale = stale_features[COLUMNS].astype(float)
    if not np.isfinite(stale.to_numpy()).all():
        raise ValueError('Stale inputs must be finite')
    if 'log_rank_diff' in targets and not np.array_equal(stale.log_rank_diff.to_numpy(), targets.log_rank_diff.to_numpy()):
        raise ValueError('Static rank input changed from the published target')
    history, history_days = tour_history(inventory)
    extras = validate_pairs(pairs, targets, target_days, history, history_days)
    base_state = HistoricalState(history)
    computed = np.array([rating_inputs(base_state, r, target_days.loc[i] - timedelta(days=21))
                         for i, r in targets[TARGET_META].to_dict('index').items()])
    baseline_error = float(np.max(np.abs(computed - stale[ELO_COLUMNS].to_numpy())))
    if baseline_error > 1e-9:
        raise ValueError(f'Original Elo inputs do not reproduce: {baseline_error}')
    fresh = stale.copy()
    coverage = pd.DataFrame({'match_id': targets.match_id.to_numpy(), 'season': targets.season.to_numpy(),
        'tourney_id': targets.tourney_id.to_numpy(), 'extra_source_matches': 0,
        'extra_source_editions': 0, 'direct_player_extra_matches': 0, 'has_extra_history': False,
        'rating_input_changed': False}, index=targets.index)
    positions = pd.Series(np.arange(len(targets)), index=targets.index)
    replay_records = []
    for target, identifiers in sorted(extras.items()):
        mask = targets.tourney_id.astype(str).eq(target)
        day = target_days[mask].iloc[0]
        cutoff = day - timedelta(days=21)
        additional = history.match_id.astype(str).isin(identifiers)
        selected = history_days.le(cutoff) | additional
        replay = history.loc[selected]
        state = HistoricalState(replay)
        latest = history_days[selected].max()
        source_count = int(history.loc[additional, 'tourney_id'].nunique())
        extra_history = history.loc[additional]
        for i, row in targets.loc[mask, TARGET_META].to_dict('index').items():
            value = rating_inputs(state, row, latest)
            original = computed[positions.loc[i]]
            # Preserve CSV bit identity when this coordinate did not move.
            for j, column in enumerate(ELO_COLUMNS):
                if value[j] != original[j]:
                    fresh.loc[i, column] = value[j]
            players = {str(row['player_a_id']), str(row['player_b_id'])}
            direct = extra_history.player_a_id.astype(str).isin(players) | extra_history.player_b_id.astype(str).isin(players)
            coverage.loc[i, ['extra_source_matches', 'extra_source_editions', 'direct_player_extra_matches',
                             'has_extra_history']] = [len(identifiers), source_count, int(direct.sum()), True]
        replay_records.append({'target_tourney_id': target, 'baseline_cutoff_day': str(cutoff),
            'last_included_nominal_day': str(latest), 'extra_source_matches': len(identifiers),
            'extra_source_editions': source_count, 'history_rows': len(replay),
            'history_match_ids_sha256': hashlib.sha256('\n'.join(sorted(replay.match_id.astype(str))).encode()).hexdigest()})
    coverage['rating_input_changed'] = np.any(stale.to_numpy() != fresh.to_numpy(), axis=1)
    no_extra = ~coverage.has_extra_history
    if not np.array_equal(stale.loc[no_extra].to_numpy(), fresh.loc[no_extra].to_numpy()):
        raise AssertionError('No-extra feature fallback must be exact')
    metadata = {'version': VERSION, 'columns': COLUMNS, 'old_elo_rebuild_max_abs_error': baseline_error,
        'qualified_pairs': len(pairs), 'target_editions_with_extra_history': len(extras),
        'extra_unique_match_ids': len(set().union(*extras.values())) if extras else 0,
        'replays': replay_records, 'source_population': 'unchanged completed ranked Tour inventory',
        'nominal_date_batching': True, 'full_target_count': len(targets),
        'information_regime': 'Tier B completed-edition historical outcomes under explicit publication assumptions',
        'inventory_id_mapping': 'Strip exactly one tour: prefix after Tour/development-year filtering; source population unchanged.',
        'inventory_id_mapping_sha256': hashlib.sha256(json.dumps(sorted(zip(
            history.inventory_match_id, history.match_id)), separators=(',', ':')).encode()).hexdigest(),
        'source_namespace_prefix_rows': int(history.inventory_match_id.str.startswith('tour:').sum()),
        'pair_identity_sha256': hashlib.sha256(json.dumps(pairs, sort_keys=True).encode()).hexdigest()}
    return fresh, coverage, metadata


def raw_elo_probability(features):
    return np.array([elo_probability(v) for v in features.elo_k32_diff.to_numpy()])
