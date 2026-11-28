import numpy as np
import pandas as pd
import pytest

from scripts.event_state_features import build_event_state_features
from scripts.train_models import mirror_features


def fixture():
    def raw(n, rd, w, l, score='6-4 6-4'):
        return dict(tourney_id='2022-1', match_num=n, round=rd, tourney_date='20220103',
            surface='Hard', winner_id=w, loser_id=l, score=score, best_of=3, minutes=90,
            loser_rank=100, w_1stWon=30, w_2ndWon=10, w_svpt=60, l_1stWon=25, l_2ndWon=10, l_svpt=60)
    rows = [(2022, raw(1, 'R32', 'a', 'x')), (2022, raw(2, 'R32', 'b', 'y', '6-4 4-6 6-4')),
        (2022, raw(3, 'R16', 'a', 'b')), (2022, raw(4, 'QF', 'a', 'z'))]
    target = pd.DataFrame([dict(match_id='2022-1:3', season=2022, tourney_id='2022-1',
        tourney_date='2022-01-03', round='R16', surface='Hard', player_a_id='a', player_b_id='b', a_won=1)])
    return target, rows


def test_only_own_earlier_rounds_hand_calculation():
    target, rows = fixture()
    x, refs = build_event_state_features(target, raw_rows=rows)
    assert x.event_sets_diff.iloc[0] == -1
    assert x.event_last_sets_diff.iloc[0] == -1
    assert x.event_prior_matches_sum.iloc[0] == 2
    assert set(refs.prior_id) == {'2022-1:1', '2022-1:2'}


def test_current_future_and_unrelated_stats_outcomes_do_not_enter():
    target, rows = fixture()
    expected, _ = build_event_state_features(target, raw_rows=rows)
    target.a_won = 0
    for _, raw in rows[2:]:
        raw['winner_id'], raw['loser_id'] = raw['loser_id'], raw['winner_id']
        raw.update(score='6-0 6-0', minutes=9999, w_svpt=999999)
    unrelated = dict(rows[0][1], match_num=5, winner_id='c', loser_id='d', minutes=99999)
    rows += [(2022, unrelated), (2024, {'invalid': 'final fields never read'})]
    actual, _ = build_event_state_features(target, raw_rows=rows)
    pd.testing.assert_frame_equal(expected, actual)


def test_exchange_and_no_history():
    target, rows = fixture()
    x, _ = build_event_state_features(target, raw_rows=rows)
    target[['player_a_id', 'player_b_id']] = target[['player_b_id', 'player_a_id']].to_numpy()
    flipped, _ = build_event_state_features(target, raw_rows=rows)
    pd.testing.assert_frame_equal(mirror_features(x), flipped)
    absent, refs = build_event_state_features(target, raw_rows=rows[2:])
    assert absent.event_any_prior.iloc[0] == 0
    assert len(refs) == 0
    assert np.isfinite(absent.to_numpy()).all()


def test_loss_path_duplicate_and_final_guards():
    target, rows = fixture()
    rows[0][1]['winner_id'], rows[0][1]['loser_id'] = 'x', 'a'
    with pytest.raises(ValueError, match='Earlier knockout loss'):
        build_event_state_features(target, raw_rows=rows)
    target, rows = fixture()
    duplicate = dict(rows[0][1], match_num=9)
    with pytest.raises(ValueError, match='Duplicate player/round'):
        build_event_state_features(target, raw_rows=rows+[(2022, duplicate)])
    target.season = 2024
    with pytest.raises(ValueError, match='locked'):
        build_event_state_features(target, raw_rows=[])


def test_bad_statistic_is_missing_without_removing_target():
    target, rows = fixture()
    rows[0][1]['w_svpt'] = -1
    out, _ = build_event_state_features(target, raw_rows=rows)
    assert len(out) == 1
    assert out.event_missing_points_sum.iloc[0] == 1


def test_official_draw_exception_is_excluded_and_target_neutralized():
    target, rows = fixture()
    for _, raw in rows:
        raw.update(tourney_id='2015-316', tourney_date='20150720')
    rows = [(2015, raw) for _, raw in rows]
    # Misplaced final falsely says a later participant lost in R32.
    rows[0][1].update(winner_id='x', loser_id='a')
    target = target.assign(match_id='2015-316:3', season=2015,
        tourney_id='2015-316', tourney_date='2015-07-20')
    actual, refs = build_event_state_features(target, raw_rows=rows)
    expected, _ = build_event_state_features(target, raw_rows=rows[1:])
    pd.testing.assert_frame_equal(actual, expected)
    assert '2015-316:1' not in set(refs.prior_id)
    bogus = target.assign(match_id='2015-316:1', round='R32', player_a_id='x', player_b_id='a')
    neutral, refs = build_event_state_features(bogus, raw_rows=rows)
    assert not neutral.to_numpy().any()
    assert refs.empty


@pytest.mark.parametrize('event,season,kept,duplicate,current', [
    ('2021-416', 2021, 14, 25, 29), ('2021-321', 2021, 5, 13, 20),
    ('2022-2807', 2022, 1, 5, 21)])
def test_verified_source_duplicates_count_once_and_mismatch_raises(event, season, kept, duplicate, current):
    target, rows = fixture()
    prior = dict(rows[0][1], tourney_id=event, tourney_date=f'{season}0601', match_num=kept)
    twin = dict(prior, match_num=duplicate)
    actual = dict(rows[2][1], tourney_id=event, tourney_date=f'{season}0601', match_num=current)
    target = target.assign(match_id=f'{event}:{current}', season=season, tourney_id=event,
                           tourney_date=f'{season}-06-01')
    x, lineage = build_event_state_features(target, raw_rows=[(season, prior), (season, twin), (season, actual)])
    expected, _ = build_event_state_features(target, raw_rows=[(season, prior), (season, actual)])
    pd.testing.assert_frame_equal(x, expected)
    assert x.event_prior_matches_sum.iloc[0] == 1
    assert set(lineage.prior_id) == {f'{event}:{kept}'}
    assert x.attrs['neutralized_target_ids'] == []
    with pytest.raises(ValueError, match='duplicates no longer match'):
        build_event_state_features(target, raw_rows=[(season, prior), (season, dict(twin, minutes=999)), (season, actual)])
    with pytest.raises(ValueError, match='duplicates no longer match'):
        build_event_state_features(target, raw_rows=[(season, twin), (season, actual)])
