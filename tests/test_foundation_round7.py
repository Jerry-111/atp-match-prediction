"""Closed temporal context, matched inputs, symmetry and foundation selection."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import joblib
import numpy as np
import pandas as pd

from scripts.train_foundation_round7 import (
    EXTRA_COLUMNS, REFERENCE, R5_COMPARATOR, R5_LOSS, SymmetricContextClassifier,
    add_blends, blend_registry, candidate_specs, choose_foundation_procedure,
    context_indices, feature_columns, mirror, select_candidate,
)


class RecordingBackend:
    def fit(self, x, y, **kwargs):
        self.x, self.y, self.kwargs = np.array(x), np.array(y), kwargs
        return self

    def predict_proba(self, x):
        p = 1 / (1 + np.exp(-(.2 + np.asarray(x)[:, 0] + .3 * np.asarray(x)[:, 1])))
        return np.column_stack([1-p, p])


class FoundationTests(unittest.TestCase):
    def setUp(self):
        self.columns = ['strength_diff', 'context_sum'] + [f'feature{i}_diff' for i in range(62)]
        rng = np.random.default_rng(158)
        self.x = pd.DataFrame(rng.normal(size=(40,64)),columns=self.columns)
        self.rows = pd.DataFrame(dict(match_id=[f'm{i}' for i in range(40)],season=2019,
                                     tourney_date='2019-06-01',a_won=np.arange(40)%2))

    def test_frozen_schema_and_small_registry(self):
        compact = [f'ml{i}_diff' for i in range(48)]
        self.assertEqual(len(feature_columns(compact+list(EXTRA_COLUMNS),compact)),64)
        with self.assertRaises(ValueError):
            feature_columns(compact, compact)
        specs = candidate_specs(self.columns)
        self.assertEqual(len(specs),6)
        self.assertEqual({s['context_rows'] for s in specs},{2048,4096})
        self.assertEqual({s['n_estimators'] for s in specs if s['kind']=='tabicl'},{4})
        self.assertEqual(len(blend_registry(specs)),6)
        p={REFERENCE:np.array([.2,.7]),**{s['name']:np.array([.6,.1]) for s in specs}}
        self.assertEqual(len(add_blends(p,blend_registry(specs))),13)

    def test_context_recent_purge_independent_of_outcome(self):
        rows=pd.DataFrame(dict(match_id=[f'm{i:04}' for i in range(4100)],season=2019,
            tourney_date=pd.date_range('2009-01-01',periods=4100,freq='D'),a_won=np.arange(4100)%2))
        rows['season']=rows.tourney_date.dt.year
        rows=pd.concat([rows,pd.DataFrame(dict(match_id=['purged','valid'],season=[2019,2020],
            tourney_date=pd.to_datetime(['2019-12-31','2020-01-06']),a_won=[1,0]))],ignore_index=True)
        actual=context_indices(rows,2020,2048)
        shuffled=rows.sample(frac=1,random_state=7).assign(a_won=lambda d:1-d.a_won)
        np.testing.assert_array_equal(actual,context_indices(shuffled,2020,2048))
        self.assertEqual(len(actual),2048)
        self.assertLessEqual(rows.loc[actual,'tourney_date'].max(),pd.Timestamp('2019-12-16'))
        self.assertNotIn('purged',rows.loc[actual,'match_id'].to_list())
        with self.assertRaises(ValueError):
            context_indices(rows,2024,2048)

    def test_mirror_training_and_equal_control_weights(self):
        for spec in candidate_specs(self.columns):
            with self.subTest(kind=spec['kind']),patch('scripts.train_foundation_round7.make_backend',return_value=RecordingBackend()):
                model=SymmetricContextClassifier(spec,'unused').fit(self.x,self.rows)
            backend=model.backend_
            np.testing.assert_array_equal(backend.y,np.concatenate([self.rows.a_won,1-self.rows.a_won]))
            np.testing.assert_allclose(backend.x[40:],mirror(self.x).to_numpy(),rtol=1e-6)
            if spec['kind']=='tabicl':
                self.assertEqual(backend.kwargs,{})
            else:
                weights=next(iter(backend.kwargs.values()))
                self.assertAlmostEqual(weights.sum(),len(self.rows))
            self.assertEqual(model.context_ids_,self.rows.match_id.to_list())

    def test_prediction_symmetry_chunks_permutation_and_serialization(self):
        spec=candidate_specs(self.columns)[0]
        with patch('scripts.train_foundation_round7.make_backend',return_value=RecordingBackend()):
            model=SymmetricContextClassifier(spec,'unused').fit(self.x,self.rows)
        p=model.predict_proba(self.x)
        np.testing.assert_allclose(p+model.predict_proba(mirror(self.x)),1,atol=1e-15)
        np.testing.assert_array_equal(p,model.predict_proba(self.x,chunk_size=7))
        permutation=np.random.default_rng(3).permutation(len(self.x))
        np.testing.assert_array_equal(p,model.predict_proba(self.x.iloc[permutation])[np.argsort(permutation)])
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'model.joblib';joblib.dump(model,path)
            np.testing.assert_array_equal(p,joblib.load(path).predict_proba(self.x.assign(a_won='POISON')))

    def test_context_holdout_date_and_label_guards(self):
        spec=candidate_specs(self.columns)[0]
        for changed in [self.rows.assign(season=2023),self.rows.assign(tourney_date='2024-01-01'),
                        self.rows.assign(a_won=.5),self.rows.assign(match_id='duplicate')]:
            with self.assertRaises(ValueError):
                SymmetricContextClassifier(spec,'unused').fit(self.x,changed)
        with self.assertRaises(ValueError):
            SymmetricContextClassifier(spec,'unused').fit(self.x.iloc[::-1],self.rows)

    def test_comparator_exclusion_early_gate_and_loss_guard(self):
        y=np.ones(6);seasons=np.array([2020,2020,2021,2021,2022,2022])
        p={REFERENCE:np.full(6,.499),'early':np.full(6,.501),
           'late':np.array([.498]*4+[.999]*2),R5_COMPARATOR:np.full(6,.99),R5_LOSS:np.full(6,.99)}
        gate=choose_foundation_procedure(y,p,seasons)
        self.assertEqual(gate['early_selection']['choice'],'early')
        self.assertEqual(gate['procedure'],'adopt_foundation')
        p['early'][-2:]=.498
        gate=choose_foundation_procedure(y,p,seasons)
        self.assertEqual(gate['early_selection']['choice'],'early')
        self.assertEqual(gate['procedure'],'retain_round4')
        selection=select_candidate(np.array([1,1,1,0]),
            {REFERENCE:np.array([.55,.55,.49,.1]),'badloss':np.array([.51,.51,.51,.49])},np.ones(4,dtype=bool))
        self.assertEqual(selection['choice'],REFERENCE)


if __name__=='__main__':
    unittest.main()
