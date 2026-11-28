import unittest
from datetime import timedelta

import numpy as np
import pandas as pd
from numpy.testing import assert_allclose
from scipy.optimize import check_grad
from scipy.special import expit

from scripts.joint_point_model import (OFFSET, attach_metadata, candidate_specs, design_matrix,
    event_identity, fit_snapshot, generate_raw_predictions, load_point_history,
    objective_and_gradient)


def synthetic_history(matches=500, players=12, seed=158):
    rng = np.random.default_rng(seed)
    serves, returns = rng.normal(0, .2, players), rng.normal(0, .15, players)
    rows = []
    for match in range(matches):
        a, b = rng.choice(players, 2, replace=False)
        for i, j in ((a,b), (b,a)):
            probability = expit(OFFSET + serves[i] - returns[j])
            rows.append(dict(date=pd.Timestamp('2019-12-01')-pd.Timedelta(days=match % 1400),
                match_id=f'm{match}', surface=('Hard','Clay','Grass')[match % 3],
                event=event_identity(f'2019-{match % 6}', f'Event {match % 6}', ('Hard','Clay','Grass')[match % 3]),
                server=f'p{i:03}', returner=f'p{j:03}', won=rng.binomial(80, probability), total=80))
    return pd.DataFrame(rows), serves, returns


def queries():
    return pd.DataFrame(dict(match_id=['2019-99:1','2019-99:2'], season=[2019,2019],
        tourney_date=['2019-12-22']*2, player_a_id=['p000','new_a'],
        player_b_id=['p001','new_b'], surface=['Hard','Grass'], event_key=['',''], best_of=[3,5]))


class JointPointModelTests(unittest.TestCase):
    def test_sparse_gradient_matches_finite_difference(self):
        history, _, _ = synthetic_history(matches=20, players=4)
        spec = candidate_specs()[-1]
        x, keys, precision = design_matrix(history, spec)
        won, total = history.won.to_numpy(), history.total.to_numpy()
        weights = np.linspace(.2, 1, len(history))
        coef = np.linspace(-.04,.05,len(keys))
        objective = lambda b: objective_and_gradient(b,x,won,total,weights,precision)
        error = check_grad(lambda b: objective(b)[0],lambda b: objective(b)[1],coef)
        self.assertLess(error, 2e-5)

    def test_joint_opponent_recovery(self):
        history, serves, returns = synthetic_history(matches=900)
        state = fit_snapshot(history, '2019-12-01', candidate_specs()[1])
        values = dict(zip(state.keys, state.coefficients))
        self.assertGreater(np.corrcoef(serves,[values[f'serve:p{i:03}'] for i in range(len(serves))])[0,1], .97)
        self.assertGreater(np.corrcoef(returns,[values[f'return:p{i:03}'] for i in range(len(returns))])[0,1], .94)
        self.assertTrue(state.diagnostics['converged'])

    def test_order_independent_and_exchange_complement(self):
        history, _, _ = synthetic_history()
        spec = candidate_specs()[-1]
        a = fit_snapshot(history, '2019-12-01', spec)
        b = fit_snapshot(history.sample(frac=1,random_state=9), '2019-12-01', spec)
        assert_allclose(a.coefficients,b.coefficients,atol=1e-12,rtol=0)
        target = queries()
        swap = target.copy()
        swap[['player_a_id','player_b_id']] = target[['player_b_id','player_a_id']].to_numpy()
        assert_allclose(a.predict_proba(target),1-a.predict_proba(swap),atol=2e-14,rtol=0)
        self.assertAlmostEqual(a.predict_proba(target)[1], .5, places=12)

    def test_current_future_and_target_labels_cannot_change_prediction(self):
        history, _, _ = synthetic_history()
        target = queries()
        raw, _, _ = generate_raw_predictions(target, history, candidate_specs()[2:3])
        future = history.iloc[:4].copy()
        future['date'], future['won'], future['total'] = pd.Timestamp('2019-12-02'), np.nan, -1
        poisoned = pd.concat([history,future],ignore_index=True)
        target['a_won'], target['w_svpt'], target['score'] = [0,1], [999,999], ['6-0 6-0']*2
        altered, _, _ = generate_raw_predictions(target, poisoned, candidate_specs()[2:3])
        assert_allclose(raw,altered,atol=0,rtol=0)

    def test_lookback_and_cutoff_are_inclusive(self):
        history, _, _ = synthetic_history(matches=2)
        cutoff = pd.Timestamp('2019-12-01')
        history.loc[:1,'date'] = cutoff-pd.Timedelta(days=1460)
        history.loc[2:,'date'] = cutoff
        state = fit_snapshot(history, cutoff, candidate_specs()[1])
        self.assertEqual(state.diagnostics['observations'],4)
        history.loc[:1,'date'] -= pd.Timedelta(days=1)
        history.loc[2:,'date'] += pd.Timedelta(days=1)
        state = fit_snapshot(history, cutoff, candidate_specs()[1])
        self.assertEqual(state.diagnostics['observations'],0)
        assert_allclose(state.predict_proba(queries()),.5,atol=1e-14)

    def test_source_event_identity_prevents_code_reuse(self):
        self.assertNotEqual(event_identity('2020-424','New York','Hard'),event_identity('2023-424','Dallas','Hard'))
        self.assertNotEqual(event_identity('2020-8998','Adelaide','Hard'),event_identity('2021-8998','Great Ocean Road','Hard'))
        self.assertEqual(event_identity('2020-999','  SOME   City ','Hard'),event_identity('2021-999','some city','Hard'))
        self.assertNotEqual(event_identity('2020-999','some city','Hard'),event_identity('2021-999','some city','Clay'))

    def test_raw_join_and_holdout_guard(self):
        inventory = pd.DataFrame([dict(source='tour',season=2018,match_id='tour:2018-99999:1',
            player_a_id='a',player_b_id='b',tourney_date='2018-01-01',surface='Hard')])
        raw = dict(tourney_id='2018-99999',match_num='1',winner_id='a',loser_id='b',
            tourney_date='20180101',surface='Hard',tourney_name='Fixture',best_of='3',
            w_1stWon='40',w_2ndWon='10',w_svpt='80',l_1stWon='30',l_2ndWon='12',l_svpt='70')
        frame, metadata, audit = load_point_history(inventory,raw_rows=[(2018,raw),(2024,{'poison':True})])
        self.assertEqual(len(frame),2)
        self.assertEqual(audit['valid_history_matches'],1)
        target = queries().iloc[:1].copy()
        target['season']=2024
        with self.assertRaises(ValueError): attach_metadata(target,metadata)
        target = queries().iloc[:1].copy()
        target['tourney_date']='2024-12-22'
        state = fit_snapshot(frame, '2019-12-01', candidate_specs()[1])
        with self.assertRaises(ValueError): state.predict_proba(target)
        with self.assertRaises(ValueError): fit_snapshot(frame,'2024-01-01',candidate_specs()[1])

    def test_five_nested_structures_and_same_pool(self):
        history, _, _ = synthetic_history()
        specs = candidate_specs()
        self.assertEqual(len(specs),5)
        states = [fit_snapshot(history,'2019-12-01',s) for s in specs]
        self.assertEqual(len({s.diagnostics['observations'] for s in states}),1)
        self.assertFalse(any(k.startswith('return:') for k in states[0].keys))
        self.assertTrue(any(k.startswith('return:') for k in states[2].keys))
        self.assertFalse(any(k.startswith('preference:') for k in states[2].keys))
        self.assertTrue(any(k.startswith('preference:') for k in states[3].keys))
        self.assertTrue(any(k.startswith('event:') for k in states[4].keys))


if __name__ == '__main__':
    unittest.main()
