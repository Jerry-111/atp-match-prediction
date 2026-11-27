"""Past observed ranking trajectories; no retrospective career-high lookup.

Rank snapshots sharing a player/date must agree or that date is omitted.
Only the published 2006--2023 history is used, and every target query ends
at T-21. Peaks are left-truncated observed peaks, not official career highs.
"""
from bisect import bisect_left, bisect_right
from collections import defaultdict
from datetime import timedelta
from math import log, log1p

import numpy as np
import pandas as pd

from scripts.advanced_features import _date
from scripts.multilevel_features import _season_date


def build_rank_trajectory_features(targets, inventory):
    if targets.empty or not pd.to_numeric(targets.season).between(2009, 2023).all():
        raise ValueError('Rank trajectory targets must be2009..2023; final test locked')
    if targets.match_id.duplicated().any():
        raise ValueError('Duplicate target match IDs')
    for row in targets.to_dict('records'):
        _season_date(_date(row['tourney_date']), int(row['season']))
    snapshots = defaultdict(set)
    cutoff = max(_date(t) for t in targets.tourney_date) - timedelta(days=21)
    history = inventory.loc[pd.to_numeric(inventory.season).between(2006, 2023)]
    for row in history.to_dict('records'):
        day = _date(row['tourney_date'])
        _season_date(day, int(row['season']))
        if day > cutoff:
            continue
        for side in ('a', 'b'):
            try:
                rank = float(row[f'rank_{side}'])
            except (TypeError, ValueError):
                continue
            if np.isfinite(rank) and rank >= 1 and rank.is_integer():
                snapshots[str(row[f'player_{side}_id']), day].add(rank)
    banks = defaultdict(list)
    conflicts = 0
    for (player, day), values in snapshots.items():
        if len(values) != 1:
            conflicts += 1
            continue
        banks[player].append((day, log(next(iter(values)))))
    packed = {}
    for player, records in banks.items():
        records.sort()
        days, ranks = zip(*records)
        ranks = np.asarray(ranks)
        packed[player] = (days, ranks, np.minimum.accumulate(ranks))

    def summary(player, latest, current_rank):
        # Fixed unseen rank100 is explicit, rather than learned on all seasons.
        days, ranks, peaks = packed.get(player, ((), np.array([]), np.array([])))
        end = bisect_right(days, latest)
        unseen = float(end == 0)
        last = float(ranks[end - 1]) if end else log(100.)
        peak = float(peaks[end - 1]) if end else log(100.)
        low = bisect_left(days, latest - timedelta(days=365))
        recent = ranks[low:end]
        old = {}
        for window in (90, 365):
            ix = bisect_right(days, latest - timedelta(days=window))
            old[window] = (float(ranks[ix - 1]) if ix else last, float(ix == 0))
        return dict(last_strength=-last, peak_strength=-peak,
            recent_peak_strength=-float(recent.min()) if len(recent) else -last,
            current_gap_from_peak=log(current_rank)-peak,
            trend90=old[90][0]-last, trend365=old[365][0]-last,
            volatility365=float(np.std(recent)) if len(recent) > 1 else 0.,
            log_snapshot_count=log1p(end), log_recent_snapshot_count=log1p(len(recent)),
            log_days_since_snapshot=log1p((latest-days[end-1]).days) if end else log1p(3650),
            unseen=unseen, missing90=old[90][1], missing365=old[365][1])

    records = []
    for row in targets.to_dict('records'):
        a, b = str(row['player_a_id']), str(row['player_b_id'])
        ra, rb = float(row['rank_a']), float(row['rank_b'])
        if a == b or not np.isfinite([ra, rb]).all() or min(ra, rb) < 1:
            raise ValueError('Distinct players and valid known target ranks required')
        latest = _date(row['tourney_date']) - timedelta(days=21)
        aa, bb = summary(a, latest, ra), summary(b, latest, rb)
        result = {f'trajectory_{key}_diff': aa[key]-bb[key] for key in aa}
        result.update(trajectory_unseen_sum=aa['unseen']+bb['unseen'],
            trajectory_missing90_sum=aa['missing90']+bb['missing90'],
            trajectory_missing365_sum=aa['missing365']+bb['missing365'],
            trajectory_log_snapshot_count_sum=aa['log_snapshot_count']+bb['log_snapshot_count'])
        records.append(result)
    output = pd.DataFrame(records, index=targets.index)
    if not np.isfinite(output.to_numpy()).all():
        raise ValueError('Nonfinite rank trajectory')
    output.attrs.update(version=1, history_lag_days=21, history_start_season=2006,
        latest_history_cutoff=str(cutoff), conflicting_player_date_snapshots=conflicts,
        usable_player_date_snapshots=sum(len(v[0]) for v in packed.values()),
        rank_only=True, source='Published inventory across Tour, Challenger and qualifying',
        peak_definition='Best observed rank since2006 and before each T-21 cutoff; not official career high',
        missing_prior_rank=100, points_omitted='No new ranking-point trajectories due changing point rules',
        final_test_evaluated=False)
    return output
