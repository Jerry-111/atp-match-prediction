from copy import deepcopy
from datetime import timedelta
from pathlib import Path
import json
import tempfile
import unittest

import numpy as np
import pandas as pd

from scripts.advanced_features import HistoricalState, _date
from scripts.freshness_round11 import COLUMNS, rating_inputs, build_freshness_inputs
from scripts.train_freshness_round11 import (PROBABILITY_COLUMNS, choose_procedure,
    predict_freshness, check_approval)
from scripts.train_models import mirror_features, sha256, temporal_masks


def history_row(match, day, a, b, won, surface='Hard', season=2022):
    return dict(match_id=match, tourney_id=match.split(':')[0], tourney_date=day,
        player_a_id=a, player_b_id=b, a_won=won, surface=surface, season=season, source='tour')


def fixture():
    history = pd.DataFrame([
        history_row('2022-base:1','2022-01-10','A','B',1),
        history_row('2022-base:2','2022-01-10','B','C',1),
        history_row('2022-source:1','2022-01-24','A','C',0),
        history_row('2022-source:2','2022-01-24','B','C',1),
        history_row('2022-future:1','2022-03-01','A','B',0),
    ])
    targets = pd.DataFrame([
        dict(match_id='2022-target:1',tourney_id='2022-target',season=2022,tourney_date='2022-01-31',
             player_a_id='A',player_b_id='B',surface='Hard',log_rank_diff=.3,a_won='ignored'),
        dict(match_id='2022-target:2',tourney_id='2022-target',season=2022,tourney_date='2022-01-31',
             player_a_id='D',player_b_id='E',surface='Clay',log_rank_diff=-.4,a_won='ignored'),
        dict(match_id='2022-other:1',tourney_id='2022-other',season=2022,tourney_date='2022-02-01',
             player_a_id='A',player_b_id='C',surface='Hard',log_rank_diff=.8,a_won='ignored'),
    ], index=[10,30,50])
    state=HistoricalState(history)
    stale=pd.DataFrame([dict(zip(COLUMNS,[r.log_rank_diff,*rating_inputs(state,r._asdict(),_date(r.tourney_date)-timedelta(days=21))]))
                        for r in targets.itertuples()],index=targets.index)
    pair=dict(source_tourney_id='2022-source',target_tourney_id='2022-target',
        source_match_ids=['2022-source:1','2022-source:2'],target_match_ids=['2022-target:1','2022-target:2'],
        source_upper_utc='2022-01-27T12:00:00Z',target_lower_utc='2022-01-30T10:00:00Z',
        admitted_for_prediction=True)
    return targets,history,stale,pair


class FakeSharedModel:
    def predict_proba(self,x):
        z=x.to_numpy() @ np.array([.2,.003,.002])
        return 1/(1+np.exp(-z))


class FreshnessFeaturesTests(unittest.TestCase):
    def test_empty_ledger_exact_and_cutoff_inclusive(self):
        t,h,s,p=fixture(); f,c,m=build_freshness_inputs(t,h,s,[])
        np.testing.assert_array_equal(f,s)
        self.assertFalse(c.has_extra_history.any())
        # January 10 is exactly T-21: both original outcomes contribute.
        self.assertEqual(s.loc[10,'elo_k32_diff'],16.)
        self.assertEqual(m['old_elo_rebuild_max_abs_error'],0.)

    def test_whole_date_replay_hand_computation_and_fallback(self):
        t,h,s,p=fixture(); f,c,m=build_freshness_inputs(t,h,s,[p])
        # The two source matches share the same frozen start-of-day C rating.
        probability=lambda d:1/(1+10**(-d/400))
        a=1516+32*(0-probability(32))
        b=1500+32*(1-probability(16))
        self.assertAlmostEqual(f.loc[10,'elo_k32_diff'],a-b,places=12)
        self.assertEqual(c.loc[10,'direct_player_extra_matches'],2)
        self.assertTrue(c.loc[30,'has_extra_history'])
        np.testing.assert_array_equal(f.loc[30],s.loc[30])
        np.testing.assert_array_equal(f.loc[50],s.loc[50])
        self.assertEqual(m['extra_unique_match_ids'],2)

    def test_future_and_target_poisoning_permutation_exchange(self):
        t,h,s,p=fixture(); expected,_,_=build_freshness_inputs(t,h,s,[p])
        poisoned=h.copy();poisoned.loc[poisoned.tourney_id.eq('2022-future'),'a_won']=1
        t['a_won']='not a label';t['w_ace']='not an input'
        result,_,_=build_freshness_inputs(t,poisoned.sample(frac=1,random_state=8),s,[p])
        np.testing.assert_array_equal(result,expected)
        swapped=t.copy();swapped[['player_a_id','player_b_id']]=t[['player_b_id','player_a_id']].to_numpy()
        swapped['log_rank_diff']=-t.log_rank_diff
        result,_,_=build_freshness_inputs(swapped,h,mirror_features(s),[p])
        np.testing.assert_allclose(result,-expected,rtol=0,atol=1e-12)

    def test_sealed_history_rejected_before_outcome_conversion(self):
        t,h,s,p=fixture();sealed=history_row('2024-bad:1','garbage','bad','bad','poison',season=2024)
        result,_,_=build_freshness_inputs(t,pd.concat([h,pd.DataFrame([sealed])],ignore_index=True),s,[p])
        expected,_,_=build_freshness_inputs(t,h,s,[p]);np.testing.assert_array_equal(result,expected)
        t.loc[10,'season']=2024
        with self.assertRaises(ValueError):build_freshness_inputs(t,h,s,[p])

    def test_fail_closed_ledger_identity_boundaries(self):
        t,h,s,p=fixture()
        mutations=[dict(source_match_ids=['2022-source:1']),dict(target_match_ids=['2022-target:1']),
                   dict(admitted_for_prediction=False),dict(source_upper_utc=None),
                   dict(target_lower_utc='2022-01-27T12:00:00Z'),
                   dict(source_upper_utc='2022-01-27T12:00:00'),
                   dict(source_tourney_id='2022-target')]
        for mutation in mutations:
            with self.subTest(mutation=mutation),self.assertRaises(ValueError):
                build_freshness_inputs(t,h,s,[dict(p,**mutation)])
        with self.assertRaises(ValueError):build_freshness_inputs(t,h,s,[p,p])
        h.loc[0,'tourney_date']='2024-01-10'
        with self.assertRaises(ValueError):build_freshness_inputs(t,h,s,[])

    def test_original_matrix_must_match(self):
        t,h,s,p=fixture();s.loc[10,'elo_k32_diff']+=.01
        with self.assertRaises(ValueError):build_freshness_inputs(t,h,s,[p])

    def test_published_tour_id_namespace_is_explicit(self):
        t,h,s,p=fixture();expected,_,_=build_freshness_inputs(t,h,s,[p])
        h['match_id']='tour:'+h.match_id
        actual,_,meta=build_freshness_inputs(t,h,s,[p])
        np.testing.assert_array_equal(actual,expected)
        self.assertEqual(meta['source_namespace_prefix_rows'],len(h))
        h.loc[0,'match_id']='tour:'+h.loc[0,'match_id']
        with self.assertRaises(ValueError):build_freshness_inputs(t,h,s,[p])

    def test_shared_probability_fallback_and_complement(self):
        t,h,s,p=fixture();f,c,_=build_freshness_inputs(t,h,s,[p]);model=FakeSharedModel()
        values=predict_freshness(model,s,f)
        swapped=predict_freshness(model,mirror_features(s),mirror_features(f))
        np.testing.assert_allclose(values+swapped,1,rtol=0,atol=2e-16)
        for alg in ('overall_elo','rank_rating_logistic'):
            self.assertEqual(values.loc[30,alg+'_stale'],values.loc[30,alg+'_fresh'])
            self.assertEqual(values.loc[50,alg+'_stale'],values.loc[50,alg+'_fresh'])
        self.assertListEqual(list(values),PROBABILITY_COLUMNS)


class SelectionTests(unittest.TestCase):
    def example(self):
        r=pd.DataFrame({'season':[2021]*4+[2022]*4,'a_won':[1]*8})
        p={name:np.full(8,.49) for name in PROBABILITY_COLUMNS}
        return r,p

    def test_gain_selection_and_gate_vs_own_control(self):
        r,p=self.example()
        p['overall_elo_fresh'][0]=.51
        p['rank_rating_logistic_fresh'][:2]=.51
        p['rank_rating_logistic_fresh'][4]=.51
        result=choose_procedure(r,p)
        self.assertEqual(result['early_selection']['choice'],'rank_rating_logistic')
        self.assertTrue(result['procedure_gate']['gate_passed'])
        self.assertEqual(result['recommendation'],'rank_rating_logistic_fresh')
        # Other algorithm wins more in 2022: it still cannot replace the 2021 choice.
        p['overall_elo_fresh'][4:]=.6
        self.assertEqual(choose_procedure(r,p)['early_selection']['choice'],'rank_rating_logistic')

    def test_no_gain_no_selection_and_no_2023_selection(self):
        r,p=self.example();result=choose_procedure(r,p)
        self.assertIsNone(result['early_selection']['choice'])
        self.assertFalse(result['procedure_gate']['gate_passed'])
        with self.assertRaises(ValueError):choose_procedure(r,pd.DataFrame(p).iloc[::-1])
        r.loc[7,'season']=2023
        with self.assertRaises(ValueError):choose_procedure(r,p)

    def test_loss_guard_and_tie_break(self):
        r,p=self.example()
        for alg in ('overall_elo','rank_rating_logistic'):
            p[alg+'_fresh'][0]=.51
        result=choose_procedure(r,p)
        self.assertEqual(result['early_selection']['choice'],'overall_elo')
        p['overall_elo_fresh'][1]=.01
        self.assertEqual(choose_procedure(r,p)['early_selection']['choice'],'rank_rating_logistic')
        p['rank_rating_logistic_fresh'][4]=.51
        p['rank_rating_logistic_fresh'][5]=.01
        self.assertFalse(choose_procedure(r,p)['procedure_gate']['gate_passed'])

    def test_equal_net_correct_uses_loss_not_subtraction_roundoff(self):
        r=pd.DataFrame({'season':[2021]*11+[2022]*4,'a_won':[1]*15})
        p={name:np.full(15,.49) for name in PROBABILITY_COLUMNS}
        for mode in ('stale','fresh'):
            p['overall_elo_'+mode][:5]=.51
            p['rank_rating_logistic_'+mode][0]=.51
        p['overall_elo_fresh'][5]=.7
        p['rank_rating_logistic_fresh'][1]=.51
        selected=choose_procedure(r,p)['early_selection']
        self.assertEqual(selected['choice'],'overall_elo')
        self.assertEqual([c['net_correct'] for c in selected['candidates']],[1,1])

    def test_original_purge_uses_first_target_day(self):
        rows=pd.DataFrame({'season':[2020,2020,2020,2021],
                           'tourney_date':['2020-12-13','2020-12-14','2020-12-15','2021-01-04']})
        train,valid=temporal_masks(rows,2021)
        self.assertListEqual(train.tolist(),[True,True,False,False])
        self.assertListEqual(valid.tolist(),[False,False,False,True])

    def test_explicit_approval_hashes_required(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);protocol=root/'p.md';protocol.write_text('fixed')
            target=root/'results/round3/development_targets.csv';target.parent.mkdir(parents=True);target.write_text('fixed target')
            content={'pairs':[{'admitted_for_prediction':True}], 'protocol_sha256':sha256(protocol),
                'root_limited_experiment_authorized':True,'final_test_outcomes_opened':False,
                'strict_asof_reconstruction':False,'models_fitted_at_freeze':0,
                'input_sha256':{'p.md':sha256(protocol),'results/round3/development_targets.csv':sha256(target)}}
            manifest=root/'m.json';manifest.write_text(json.dumps(content))
            with self.assertRaises(ValueError):check_approval(protocol,manifest,None,None,root=root)
            check_approval(protocol,manifest,sha256(protocol),sha256(manifest),root=root)
            with self.assertRaises(ValueError):check_approval(protocol,manifest,'changed',sha256(manifest),root=root)
            for field,value in [('root_limited_experiment_authorized',False),('final_test_outcomes_opened',True),
                    ('models_fitted_at_freeze',1),('strict_asof_reconstruction',True),('protocol_sha256','wrong')]:
                manifest.write_text(json.dumps(dict(content,**{field:value})))
                with self.subTest(field=field),self.assertRaises(ValueError):
                    check_approval(protocol,manifest,sha256(protocol),sha256(manifest),root=root)
            manifest.write_text(json.dumps(content));target.write_text('changed')
            with self.assertRaises(ValueError):check_approval(protocol,manifest,sha256(protocol),sha256(manifest),root=root)


if __name__=='__main__':unittest.main()
