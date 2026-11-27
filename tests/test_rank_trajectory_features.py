import numpy as np
import pandas as pd
import pytest

from scripts.rank_trajectory_features import build_rank_trajectory_features
from scripts.train_models import mirror_features


def targets():
    return pd.DataFrame([dict(match_id='x', season=2023, tourney_date='2023-06-01',
        player_a_id='a', player_b_id='b', rank_a=15, rank_b=30, a_won=1)])


def history():
    return pd.DataFrame([
        dict(season=2022, tourney_date='2022-01-01', player_a_id='a', player_b_id='b', rank_a=50, rank_b=20),
        dict(season=2023, tourney_date='2023-05-11', player_a_id='a', player_b_id='b', rank_a=10, rank_b=30),
    ])


def test_known_trajectory_and_exact_cutoff():
    out = build_rank_trajectory_features(targets(), history())
    assert out.trajectory_peak_strength_diff.iloc[0] == pytest.approx(np.log(2))
    assert out.trajectory_trend90_diff.iloc[0] == pytest.approx(np.log(7.5))
    assert out.trajectory_current_gap_from_peak_diff.iloc[0] == pytest.approx(0.)


def test_future_target_outcomes_and_test_fields_cannot_change_features():
    t, h = targets(), history()
    expected = build_rank_trajectory_features(t, h)
    future = h.iloc[-1].copy()
    future.tourney_date = '2023-05-12'
    future.rank_a, future.rank_b = 1, 2000
    final = dict(season=2024, tourney_date='not a date', rank_a='bad', rank_b='bad')
    h = pd.concat([h, pd.DataFrame([future.to_dict(), final])], ignore_index=True)
    t['a_won'], t['w_svpt'] = 0, 999999
    pd.testing.assert_frame_equal(expected, build_rank_trajectory_features(t, h))


def test_conflicting_snapshot_omitted_and_input_order_irrelevant():
    h = history()
    conflict = h.iloc[-1].copy()
    conflict.rank_a = 9
    h = pd.concat([h, pd.DataFrame([conflict])], ignore_index=True)
    out = build_rank_trajectory_features(targets(), h)
    assert out.attrs['conflicting_player_date_snapshots'] == 1
    assert out.trajectory_peak_strength_diff.iloc[0] == pytest.approx(np.log(20/50))
    pd.testing.assert_frame_equal(out, build_rank_trajectory_features(targets(), h.iloc[::-1]))


def test_exchange_symmetry_and_unseen():
    t, h = targets(), history()
    expected = mirror_features(build_rank_trajectory_features(t, h))
    t[['player_a_id', 'player_b_id']] = t[['player_b_id', 'player_a_id']].to_numpy()
    t[['rank_a', 'rank_b']] = t[['rank_b', 'rank_a']].to_numpy()
    pd.testing.assert_frame_equal(expected, build_rank_trajectory_features(t, h))
    out = build_rank_trajectory_features(t, h.iloc[:0])
    assert out.trajectory_unseen_sum.iloc[0] == 2
    assert np.isfinite(out.to_numpy()).all()


def test_final_targets_rejected():
    t = targets()
    t.season = 2024
    with pytest.raises(ValueError, match='locked'):
        build_rank_trajectory_features(t, history())
