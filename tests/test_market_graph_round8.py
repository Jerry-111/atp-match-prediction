"""Graph mathematics, temporal prefixes, source identity and exact fallbacks."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from scipy.special import expit, logit

from scripts.market_graph_features import (RIDGE, SOURCE_COLUMNS, build_market_graph_features,
    build_opening_history, fit_graph, format_probability, graph_specs,
    load_opening_history, series_probability, set_probability, sha256)
from scripts.train_market_graph_round8 import candidate_specs, graph_predictions
from scripts.train_distillation_round7 import REFERENCE, choose_procedure, select_candidate


def target(a=1,b=3, date='2021-03-01', key='target'):
    return dict(match_id=key,season=int(date[:4]),tourney_date=date,tourney_id='2021-1',
                player_a_id=a,player_b_id=b,surface='Hard',best_of=3)


def edge(a,b,z,stamp='2021-02-08T00:00:00Z',key=1,surface='Hard'):
    return dict(match_id=f'm{key}',source_match_id=key,source_season=int(stamp[:4]),
                match_utc=stamp,player_a_id=a,player_b_id=b,surface=surface,best_of=3,
                opening_logit_diff=z)


class GraphTests(unittest.TestCase):
    def test_hand_chain_cycle_and_dense_normal_equations(self):
        nodes,r,c,n = fit_graph([1,2],[2,3],[.7,.7],[1.,1.])
        np.testing.assert_allclose(r,[.7/(1+RIDGE),0,-.7/(1+RIDGE)],atol=1e-13)
        np.testing.assert_array_equal(n,[1,2,1])
        self.assertEqual(len(set(c)),1)
        a,b,z,w = np.array([1,2,1]),np.array([2,3,3]),np.array([.7,-.2,.9]),np.array([2.,3.,.4])
        _,r,_,_ = fit_graph(a,b,z,w)
        B = np.array([[1,-1,0],[0,1,-1],[1,0,-1]])
        expected = np.linalg.solve(B.T@np.diag(w)@B+RIDGE*np.eye(3),B.T@(w*z))
        np.testing.assert_allclose(r,expected,atol=1e-12)

    def test_format_inversion_and_symmetry(self):
        p = np.linspace(.0001,.9999,101)
        for best in (3,5):
            np.testing.assert_allclose(series_probability(set_probability(p,best),best),p,atol=1e-14)
            q = format_probability(p,3,best)
            np.testing.assert_allclose(format_probability(1-p,3,best),1-q,atol=1e-14)
        self.assertGreater(format_probability(.7,3,5),.7)
        np.testing.assert_allclose(format_probability(format_probability(p,3,5),5,3),p,atol=1e-12)

    def test_inclusive_timestamp_future_poison_and_no_target_stats(self):
        targets = pd.DataFrame([target()],index=[8])
        history = pd.DataFrame([edge(1,2,.7),edge(2,3,.7,key=2),
            edge(1,3,-99,'2021-02-08T00:00:01Z',3)])
        out = build_market_graph_features(targets,history)
        self.assertEqual(out.iloc[0].market_graph_available,1)
        self.assertAlmostEqual(out.iloc[0].market_graph_hinf_surface0_diff,1.4/(1+RIDGE),places=12)
        history.loc[2,'opening_logit_diff'] = np.nan
        history = pd.concat([history,pd.DataFrame([dict(source_season=2024,match_utc='not parsed',opening_logit_diff=object())])],ignore_index=True)
        poisoned = targets.assign(a_won=object(),w_svpt=object(),cote1_ouverture=object())
        pd.testing.assert_frame_equal(out,build_market_graph_features(poisoned,history))

    def test_prefix_invariance_with_future_queries_and_same_time_permutation(self):
        t = pd.DataFrame([target(),target(date='2021-06-01',key='later')])
        h = pd.DataFrame([edge(1,2,.7,key=1),edge(2,3,.3,key=2),edge(1,3,-2,'2021-04-01T12:00:00Z',3)])
        result = build_market_graph_features(t,h)
        pd.testing.assert_frame_equal(result.iloc[[0]],build_market_graph_features(t.iloc[[0]],h))
        pd.testing.assert_frame_equal(result,build_market_graph_features(t,h.sample(frac=1,random_state=19)))
        changed = h.copy(); changed.loc[2,'opening_logit_diff']=2
        np.testing.assert_array_equal(result.iloc[0],build_market_graph_features(t,changed).iloc[0])

    def test_swap_disconnected_unknown_and_locked_history(self):
        t = pd.DataFrame([target(),target(1,5,key='split'),target(1,9,key='unknown')])
        h = pd.DataFrame([edge(1,2,.7),edge(2,3,.2,key=2),edge(4,5,.8,key=3)])
        result = build_market_graph_features(t,h)
        self.assertEqual(result.market_graph_available.tolist(),[1,0,0])
        swapped = t.copy(); swapped[['player_a_id','player_b_id']] = t[['player_b_id','player_a_id']]
        parity = np.array([-1 if c.endswith('_diff') else 1 for c in result])
        np.testing.assert_allclose(build_market_graph_features(swapped,h),result.to_numpy()*parity,atol=1e-12)
        reference = np.array([.6,.314159265,.123456789])
        probs = graph_predictions(reference,result)
        for c in probs:
            np.testing.assert_array_equal(probs[c].to_numpy()[1:],reference[1:])
        reversed_p = graph_predictions(1-reference,build_market_graph_features(swapped,h))
        np.testing.assert_allclose(probs+reversed_p,1,atol=1e-13)
        h23 = pd.DataFrame([edge(1,3,.8,'2023-01-01T00:00:00Z')])
        t23 = pd.DataFrame([target(date='2023-03-01')])
        self.assertEqual(build_market_graph_features(t23,h23).iloc[0].market_graph_available,1)
        self.assertEqual(build_market_graph_features(t23,h23,locked_before='2023-01-01').iloc[0].market_graph_available,0)

    def test_weighted_decay_surface_and_year_guards(self):
        t=pd.DataFrame([target(date='2021-07-01')])
        h=pd.DataFrame([edge(1,3,1,'2021-01-01T00:00:00Z',1,'Clay'),edge(1,3,-1,'2021-06-01T00:00:00Z',2,'Hard')])
        out=build_market_graph_features(t,h)
        self.assertAlmostEqual(out.iloc[0].market_graph_hinf_surface0_diff,0)
        self.assertLess(out.iloc[0].market_graph_h90_surface0_diff,0)
        self.assertLess(out.iloc[0].market_graph_hinf_surface1_diff,0)
        with self.assertRaises(ValueError):
            build_market_graph_features(pd.DataFrame([target(date='2024-01-01')]),h)
        with self.assertRaises(ValueError):
            build_market_graph_features(pd.DataFrame([target(date='2023-01-01')]).assign(tourney_date='2024-01-01'),h)
        with self.assertRaises(ValueError):
            build_market_graph_features(t,pd.concat([h,h.iloc[[0]]]))

    def source_fixture(self):
        rows = pd.DataFrame([target('D643','N409',date='2021-01-04',key='m')])
        link = dict(match_id='m',season=2021,status='matched',target_event_id='2021-1',
            market_match_id=88,market_event_id=77,market_match_utc='2021-01-07T10:00:00Z',
            market_player_a_id=12,market_player_b_id=11)
        source = dict(match_id=88,date=link['market_match_utc'],tournoi_id=77,joueur1_id=11,
            joueur2_id=12,cote1_ouverture=3.,cote2_ouverture=1.5,source_season=2021)
        return rows,pd.DataFrame([link]),pd.DataFrame([source])

    def test_source_orientation_outcome_ignored_and_identity_reject(self):
        rows,links,src=self.source_fixture()
        expected=build_opening_history(rows,links,src)
        self.assertEqual(expected.iloc[0].player_a_id,'D643')
        self.assertEqual(expected.iloc[0].player_b_id,'N409')
        self.assertAlmostEqual(expected.iloc[0].opening_logit_diff,np.log(2))
        src['vainqueur_id']=object();src['score']=object();src['cote1_cloture']=object()
        pd.testing.assert_frame_equal(expected,build_opening_history(rows,links,src))
        changed=src.copy();changed.loc[0,'joueur1_id']=99
        with self.assertRaises(ValueError):build_opening_history(rows,links,changed)
        changed=src.copy();changed.loc[0,'date']='2024-01-07T10:00:00Z'
        with self.assertRaises(ValueError):build_opening_history(rows,links,changed)
        changed=links.copy();changed.loc[0,'market_event_id']=42
        with self.assertRaises(ValueError):build_opening_history(rows,changed,src)

    def test_loader_reads_only_permitted_years_and_opening_identity_columns(self):
        rows,links,src=self.source_fixture()
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);files=[]
            for year in (2021,2022,2023):
                data=src[SOURCE_COLUMNS] if year==2021 else src[SOURCE_COLUMNS].iloc[:0]
                p=root/f'{year}.csv';data.to_csv(p,sep=';',index=False)
                files.append(dict(season=year,sha256=sha256(p),rows=len(data)))
            manifest=root/'manifest.json';manifest.write_text(json.dumps(dict(files=files)))
            lp=root/'links.csv';links.to_csv(lp,index=False)
            audit=root/'audit.json';audit.write_text(json.dumps(dict(artifact_sha256={'linked_matches.csv':sha256(lp)},source_manifest_sha256=sha256(manifest))))
            original=pd.read_csv;seen=[]
            def spy(path,*args,**kwargs):
                seen.append((Path(path).name,kwargs.get('usecols')))
                return original(path,*args,**kwargs)
            with patch('scripts.market_graph_features.pd.read_csv',side_effect=spy):
                loaded=load_opening_history(rows,cache=root,manifest_path=manifest,linkage_path=lp,audit_path=audit)
            self.assertEqual(len(loaded),1)
            self.assertEqual([name for name,_ in seen],['links.csv','2021.csv','2022.csv','2023.csv'])
            for name,cols in seen[1:]:self.assertEqual(cols,SOURCE_COLUMNS)

    def test_fixed_menu_selection_masks_ignore_report_only_candidates(self):
        specs=candidate_specs()
        self.assertEqual(len(graph_specs()),6);self.assertEqual(len(specs),12)
        y=np.array([1,0,1,0])
        predictions={s['name']:np.array([.6,.4,.4,.6]) for s in specs}
        predictions[REFERENCE]=np.array([.6,.4,.6,.4])
        predictions['not_eligible']=np.array([1.,0.,1.,0.])
        chosen=select_candidate(y,predictions,specs,np.array([1,1,0,0],dtype=bool))
        self.assertNotEqual(chosen['choice'],'not_eligible')
        gate=choose_procedure(y,predictions,specs,np.array(['2021H2','2021H2','2022','2022']))
        self.assertEqual(gate['procedure'],'retain_round4')


if __name__=='__main__':unittest.main()
