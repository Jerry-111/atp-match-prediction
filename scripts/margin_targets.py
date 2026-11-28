"""Training-only performance targets: never read 2023+ target scores/stats."""
from __future__ import annotations

from math import isfinite

import pandas as pd

from scripts.advanced_features import _date
from scripts.baseline import DEFAULT_CACHE
from scripts.model_event_metadata import corrected_tour_row
from scripts.multilevel_features import _player, _season_date
from scripts.paper_margin_features import score_weights
from scripts.serve_features import _annual_rows, _counts, development_entries


def point_share(raw, player_a):
    """A's total points won / all points, including A's return points."""
    counts = _counts(raw)
    if counts is None:
        return None
    (ww, wn), (lw, ln) = counts
    winner_fraction = (ww+ln-lw)/(wn+ln)
    return winner_fraction if str(raw['winner_id']) == str(player_a) else 1.-winner_fraction


def build_performance_targets(rows, *, cache_dir=DEFAULT_CACHE, raw_rows=None):
    """Read outcomes of training rows only; return aligned targets and masks.

    These postmatch quantities are supervised labels, not inference features.
    Reject 2023 even though its binary outcome is the exploratory comparator.
    """
    if rows.empty or not pd.to_numeric(rows.season, errors='coerce').isin(range(2009, 2023)).all():
        raise ValueError('Performance targets are training-only 2009--2022; 2023+ target stats are locked')
    expected = {}
    for row in rows.to_dict('records'):
        day, season = _date(row['tourney_date']), int(row['season'])
        _season_date(day, season)
        if day.year > 2022:
            raise ValueError('Performance target dates must precede 2023')
        key = str(row['match_id']).removeprefix('tour:')
        if key in expected:
            raise ValueError('Duplicate performance target match ID')
        if row['a_won'] not in (0, 1) or _player(row['player_a_id']) == _player(row['player_b_id']):
            raise ValueError('Performance targets require binary outcomes and distinct player IDs')
        expected[key] = row
    entries = [e for e in development_entries() if 2009 <= int(e['season']) <= 2022] if raw_rows is None else []
    source = _annual_rows(cache_dir, entries) if raw_rows is None else raw_rows
    matched = {}
    for season, raw in source:
        if not 2009 <= int(season) <= 2022:
            continue  # Do not inspect even the IDs of injected 2023+ rows.
        key = f"{raw['tourney_id']}:{int(raw['match_num'])}"
        if key not in expected:
            continue
        if key in matched:
            raise ValueError('Duplicate raw performance target ID')
        row, raw = expected[key], corrected_tour_row(raw)
        a, b = str(row['player_a_id']), str(row['player_b_id'])
        winner, loser = str(raw['winner_id']), str(raw['loser_id'])
        if ({a, b} != {winner, loser} or float(winner == a) != float(row['a_won'])
                or int(row['season']) != int(season) or _date(row['tourney_date']) != _date(raw['tourney_date'])
                or str(row['surface']) != str(raw['surface'])):
            raise ValueError(f'Performance source identity/date/outcome mismatch: {key}')
        points = point_share(raw, a)
        try:
            game_winner, _ = score_weights(raw['score'], raw['best_of'])
            games = game_winner if winner == a else 1.-game_winner
        except (KeyError, TypeError, ValueError):
            games = None
        matched[key] = dict(point_share=float('nan') if points is None else points,
                            game_share=float('nan') if games is None else games,
                            point_available=int(points is not None), game_available=int(games is not None))
    if expected.keys()-matched.keys():
        raise ValueError('Raw source omitted training performance targets')
    frame = pd.DataFrame([matched[str(k).removeprefix('tour:')] for k in rows.match_id], index=rows.index)
    for name in ('point', 'game'):
        observed = frame.loc[frame[f'{name}_available'].eq(1), f'{name}_share']
        if not all(isfinite(v) and 0 <= v <= 1 for v in observed):
            raise ValueError('Performance shares must lie in [0, 1]')
    frame.attrs.update(training_rows=len(rows), point_valid=int(frame.point_available.sum()),
                       game_valid=int(frame.game_available.sum()), source_files=entries,
                       target_stats_last_season=2022, read_2023_target_stats=False,
                       final_test_evaluated=False, injected_raw_rows=raw_rows is not None)
    return frame
