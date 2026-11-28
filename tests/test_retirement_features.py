from datetime import date
from math import exp,log1p

import numpy as np
import pandas as pd
import pytest

from scripts.retirement_features import build_retirement_features, player_state, GROUPS, status_from_score


def target(day='2022-06-30'):
    return pd.DataFrame([dict(match_id='target',season=2022,tourney_date=day,
                             player_a_id='A',player_b_id='B',a_won=1)],index=[17])


def inventory(days=('2022-01-01','2022-03-01','2022-06-09')):
    return pd.DataFrame([dict(match_id=f'tour:{i}',season=2022,source='tour',
                             tourney_date=d,player_a_id='A',player_b_id='C',a_won=1)
                         for i,d in enumerate(days)])


def raw(day='20220609',score='0-0 RET',loser='A',num=1,level='250',round='RR'):
    return (2022,dict(tourney_id='2022-999',match_num=num,tourney_date=day,tourney_level=level,
                     winner_id='C',loser_id=loser,score=score,round=round))


def test_boundary_and_hand_calculated_states():
    x=build_retirement_features(target(),inventory(),raw_rows=[raw()])
    assert x.index.tolist()==[17] and x.shape==(1,12)
    assert x.loc[17,'ordinary_gap_diff']==-1  # A played at C, B unseen.
    assert x.loc[17,'ret_recency_diff']==1 and x.loc[17,'ret_presence_diff']==1
    assert x.loc[17,'ret_return_state_diff']==1  # same-date completion is not after RET.
    assert x.loc[17,'ordinary_return_state_diff']==1
    assert len(GROUPS['I'])==6 and len(GROUPS['R'])==len(GROUPS['RC'])==10
    assert set(GROUPS['I'])<=set(GROUPS['R']) & set(GROUPS['RC'])


def test_cutoff_future_and_target_statistics_ignored():
    base=build_retirement_features(target(),inventory(),raw_rows=[raw()])
    future=raw('20220610',loser='future',num=2)
    current=raw('20220630',score='RET DEF',num=3)
    poisoned=dict(current[1],w_svpt=object(),l_svpt=object(),winner_id=None,loser_id=None)
    x=build_retirement_features(target().assign(a_won='poison'),inventory().assign(a_won='poison'),
        raw_rows=[raw(),future,(2022,poisoned),(2024,{'unreadable':object()})])
    np.testing.assert_array_equal(x,base)


def test_same_date_permutation_and_swap():
    rows=[raw(),raw(loser='B',num=2),raw('20220301',num=3)]
    inv=pd.concat([inventory(),inventory().assign(match_id=lambda x:x.match_id+'copy')])
    a=build_retirement_features(target(),inv,raw_rows=rows)
    b=build_retirement_features(target(),inv.sample(frac=1,random_state=3),raw_rows=rows[::-1])
    np.testing.assert_array_equal(a,b)
    swapped=target().assign(player_a_id='B',player_b_id='A')
    c=build_retirement_features(swapped,inv,raw_rows=rows)
    signs=np.array([-1 if name.endswith('_diff') else 1 for name in a])
    np.testing.assert_allclose(c,a*signs,atol=0,rtol=0)


def test_non_ret_markers_and_non_tour_levels_do_not_enter_state():
    none=build_retirement_features(target(),inventory(),raw_rows=[])
    rows=[raw(score='W/O'),raw(score='DEF',num=2),raw(score='RET DEF',num=3),
          raw(level='D',num=4),raw(level='C',num=5)]
    x=build_retirement_features(target(),inventory(),raw_rows=rows)
    np.testing.assert_array_equal(x,none)
    assert status_from_score('0-0 RET')=='retirement'


def test_window_and_completed_date_decay():
    cutoff=date(2022,6,9);first=date(2021,7,1)
    completed=[date(2022,1,1),date(2022,5,1),date(2022,6,1)]
    state=player_state(completed,[date(2022,5,1),date(2022,6,1)],
                       [date(2022,5,1)],first,cutoff)
    assert state[0]==log1p(8)/log1p(365)
    assert state[3]==exp(-39/90)
    assert state[5]==exp(-39/90)*exp(-1/5)
    assert player_state([],[],[date(2021,6,30)],first,cutoff)[3:]==(0.,0.,0.)
    assert player_state([],[],[first],first,cutoff)[4]==1


def test_date_identity_and_final_year_guards():
    with pytest.raises(ValueError,match='Final'):
        build_retirement_features(target().assign(season=2024),inventory(),raw_rows=[])
    with pytest.raises(ValueError,match='season'):
        build_retirement_features(target(day='2024-06-30'),inventory(),raw_rows=[])
    with pytest.raises(ValueError,match='Duplicate RET'):
        build_retirement_features(target(),inventory(),raw_rows=[raw(),raw()])
    with pytest.raises(ValueError,match='identity'):
        build_retirement_features(target(),inventory(),raw_rows=[raw(loser='C')])
    bad_final=pd.DataFrame([dict(source='tour',season=2025,tourney_date='poison',a_won='poison')])
    clean=build_retirement_features(target(),inventory(),raw_rows=[])
    actual=build_retirement_features(target(),pd.concat([inventory(),bad_final]),raw_rows=[])
    np.testing.assert_array_equal(clean,actual)
