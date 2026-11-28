"""Actual-prematch own earlier-round state, a separate regime from T-21.

Only the target player's own strictly earlier knockout rounds in the same
edition can contribute. Their completion follows from that player's later
participation; no ordering between different players' matches is assumed.
Current/future rounds and every other event are excluded, even with dates
identical to the target. No actual completion timestamp is invented.
"""
from collections import Counter, defaultdict
from math import isfinite, log, log1p

import numpy as np
import pandas as pd

from scripts.advanced_features import _date
from scripts.baseline import DEFAULT_CACHE
from scripts.model_event_metadata import corrected_tour_row
from scripts.multilevel_features import _season_date
from scripts.paper_margin_features import SET_TOKEN, score_weights
from scripts.reconcile_matches import DUPLICATE_RECORDS
from scripts.serve_features import _annual_rows, _counts, development_entries

ROUND_ORDER = {'R128': 0, 'R64': 1, 'R32': 2, 'R16': 3, 'QF': 4, 'SF': 5, 'F': 6}
# Official draw proves this is a misplaced copy of the final, not an R32 match.
# This row is already absent from the published target cohort. Reuse its
# exclusion when expanding raw prior-round history beyond inventory filters.
# Evidence and source hash: docs/research/round7_event_source_audit.md.
INVALID_PRIOR_IDS = frozenset({'2015-316:1'})
DUPLICATE_PRIOR_IDS = {duplicate.removeprefix('tour:'): retained.removeprefix('tour:')
                      for duplicate, retained in DUPLICATE_RECORDS.items() if duplicate.startswith('tour:')}
# Both source copies were outside the original positive-rank inventory. The
# broader event-history regime admits unranked opponents, so reconcile them too.
# Official draw plus exact source comparison: round7_event_source_audit.md.
DUPLICATE_PRIOR_IDS['2022-2807:5'] = '2022-2807:1'
EXCLUDED_PRIOR_IDS = INVALID_PRIOR_IDS | frozenset(DUPLICATE_PRIOR_IDS)


def verify_prior_duplicates(by_id):
    """Verify named same-knockout-round copies, never dedupe by player pair."""
    for duplicate, retained in DUPLICATE_PRIOR_IDS.items():
        if duplicate not in by_id:
            continue
        a, b = by_id[duplicate], by_id.get(retained)
        if (b is None or a.get('round') not in ROUND_ORDER
                or {k: v for k, v in a.items() if k != 'match_num'} !=
                   {k: v for k, v in b.items() if k != 'match_num'}):
            raise ValueError(f'Known prior-source duplicates no longer match: {duplicate} / {retained}')


def _prior_summary(records, player):
    totals = defaultdict(float)
    rounds = set()
    last_sets = last_games = 0.
    for _, key, raw in sorted(records, key=lambda x: (x[0], x[1])):
        stage = raw['round']
        if stage in rounds:
            raise ValueError(f'Duplicate player/round in knockout path: {player} {key}')
        rounds.add(stage)
        if str(raw['winner_id']) != player:
            raise ValueError(f'Earlier knockout loss followed by later participation: {player} {key}')
        totals['prior_matches'] += 1
        score = str(raw.get('score', '')).strip().upper()
        try:
            score_weights(score, raw.get('best_of'))
        except (ValueError, TypeError, KeyError):
            totals['noncomplete_matches'] += 1
            if 'W/O' in score or 'WALKOVER' in score:
                totals['walkovers'] += 1
            continue
        totals['completed_matches'] += 1
        sets = [tuple(map(int, SET_TOKEN.fullmatch(token).groups())) for token in score.split()]
        games = sum(a+b for a, b in sets)
        totals['sets'] += len(sets)
        totals['games'] += games
        totals['games_won'] += sum(a for a, _ in sets)
        totals['sets_lost'] += sum(a < b for a, b in sets)
        last_sets, last_games = len(sets), games
        try:
            minutes = float(raw.get('minutes'))
        except (ValueError, TypeError):
            minutes = float('nan')
        if isfinite(minutes) and minutes > 0:
            totals['minutes'] += minutes
        else:
            totals['missing_minutes'] += 1
        try:
            rank = float(raw.get('loser_rank'))
        except (ValueError, TypeError):
            rank = float('nan')
        if isfinite(rank) and rank >= 1:
            totals['opponent_log_rank'] += log(rank)
            totals['opponent_rank_count'] += 1
        counts = _counts(raw)
        if counts is None:
            totals['missing_points'] += 1
        else:
            (sw, sn), (ow, on) = counts
            totals['serve_won'] += sw
            totals['serve_n'] += sn
            totals['return_won'] += on-ow
            totals['return_n'] += on
    result = {key: totals[key] for key in ('prior_matches', 'completed_matches', 'noncomplete_matches',
        'walkovers', 'sets', 'sets_lost', 'last_sets', 'last_games', 'missing_minutes', 'missing_points')}
    result.update(last_sets=last_sets, last_games=last_games,
        log_games=log1p(totals['games']), log_minutes=log1p(totals['minutes']),
        game_share=(totals['games_won']+10)/(totals['games']+20),
        point_share=(totals['serve_won']+totals['return_won']+50)/(totals['serve_n']+totals['return_n']+100),
        serve_rate=(totals['serve_won']+65)/(totals['serve_n']+100),
        return_rate=(totals['return_won']+35)/(totals['return_n']+100),
        opponent_log_rank=(totals['opponent_log_rank']+2*log(100))/(totals['opponent_rank_count']+2),
        log_points=log1p(totals['serve_n']+totals['return_n']))
    return result


def build_event_state_features(targets, *, cache_dir=DEFAULT_CACHE, raw_rows=None):
    if targets.empty or targets.match_id.duplicated().any() or not pd.to_numeric(targets.season).between(2009, 2023).all():
        raise ValueError('Unique development targets 2009..2023 required; final test locked')
    target_records = targets.to_dict('records')
    editions = set()
    for row in target_records:
        _season_date(_date(row['tourney_date']), int(row['season']))
        if row['round'] not in ROUND_ORDER or str(row['player_a_id']) == str(row['player_b_id']):
            raise ValueError('Distinct players and known knockout target round required')
        editions.add((int(row['season']), str(row['tourney_id'])))
    entries = [e for e in development_entries() if int(e['season']) >= 2009] if raw_rows is None else []
    source = _annual_rows(cache_dir, entries) if raw_rows is None else raw_rows
    by_id, source_events, by_player = {}, {}, defaultdict(list)
    for season, raw in source:
        if not 2009 <= int(season) <= 2023:
            continue
        event = (int(season), str(raw['tourney_id']))
        if event not in editions:
            continue
        raw = corrected_tour_row(raw)
        _season_date(_date(raw['tourney_date']), int(season))
        key = f"{raw['tourney_id']}:{int(raw['match_num'])}"
        if key in by_id:
            raise ValueError('Duplicate source match identity')
        by_id[key] = raw
        source_events[key] = event
    verify_prior_duplicates(by_id)
    for key, raw in by_id.items():
        if key in EXCLUDED_PRIOR_IDS or raw['round'] not in ROUND_ORDER:
            continue  # RR, qualifying and unknown formats cannot establish order.
        for player in (str(raw['winner_id']), str(raw['loser_id'])):
            by_player[source_events[key], player].append((ROUND_ORDER[raw['round']], key, raw))
    output, references = [], []
    for row in target_records:
        key = str(row['match_id'])
        raw = by_id.get(key)
        a, b = str(row['player_a_id']), str(row['player_b_id'])
        if raw is None or ({a, b} != {str(raw['winner_id']), str(raw['loser_id'])}
                or row['round'] != raw['round'] or str(row['surface']) != str(raw['surface'])
                or _date(row['tourney_date']) != _date(raw['tourney_date'])):
            raise ValueError(f'Target/source metadata identity mismatch: {key}')
        event = (int(row['season']), str(row['tourney_id']))
        order = ROUND_ORDER[row['round']]
        histories = ([[], []] if key in EXCLUDED_PRIOR_IDS else
            [[item for item in by_player[event, player] if item[0] < order] for player in (a, b)])
        aa, bb = [_prior_summary(history, player) for history, player in zip(histories, (a, b))]
        result = {f'event_{name}_diff': aa[name]-bb[name] for name in aa}
        result.update(event_round_order=float(order), event_prior_matches_sum=aa['prior_matches']+bb['prior_matches'],
            event_completed_matches_sum=aa['completed_matches']+bb['completed_matches'],
            event_log_points_sum=aa['log_points']+bb['log_points'],
            event_missing_points_sum=aa['missing_points']+bb['missing_points'],
            event_missing_minutes_sum=aa['missing_minutes']+bb['missing_minutes'],
            event_any_prior=float(bool(histories[0] or histories[1])))
        if key in EXCLUDED_PRIOR_IDS:
            result = {name: 0. for name in result}
        output.append(result)
        for side, history in zip(('a', 'b'), histories):
            references.extend(dict(target_id=key, player_side=side, prior_id=prior_key,
                target_round=row['round'], prior_round=prior_raw['round']) for _, prior_key, prior_raw in history)
    frame = pd.DataFrame(output, index=targets.index)
    if not np.isfinite(frame.to_numpy()).all():
        raise ValueError('Nonfinite event features')
    counts = Counter(int(r['season']) for r, v in zip(target_records, frame.event_any_prior) if v)
    frame.attrs.update(version=2, information_regime='Actual prematch own earlier knockout rounds; separate from T-21',
        temporal_rule='Same event and player; strictly smaller round; no lost-match path continuation',
        current_or_future_rounds_used=False, other_players_matches_used=False,
        source_files=entries, targets_with_prior_by_season=dict(sorted(counts.items())),
        fallback='Prediction returns original R4 probability when neither player has a prior round',
        priors=dict(points=100, games=20, opponent_rank_pseudomatches=2), final_test_evaluated=False)
    frame.attrs.update(excluded_prior_ids=sorted(EXCLUDED_PRIOR_IDS),
        observed_excluded_prior_ids=sorted(set(by_id) & EXCLUDED_PRIOR_IDS),
        verified_duplicate_prior_map=dict(sorted(DUPLICATE_PRIOR_IDS.items())),
        neutralized_target_ids=sorted(set(targets.match_id) & EXCLUDED_PRIOR_IDS),
        last_match_load_definition='Most recent complete conventional-score match')
    return frame, pd.DataFrame(references)
