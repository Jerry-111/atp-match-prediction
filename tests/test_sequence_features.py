import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.sequence_features import (COLUMNS, TOKENS, build_sequence_features,
    load_sequences, save_sequences)


def match(day,number=1,a='A',b='B',won=1,valid=True,surface='Hard'):
    year=int(day[:4]);tid=f'{year}-1'
    inv=dict(match_id=f'tour:{tid}:{number}',season=year,source='tour',tourney_id=tid,
        tourney_date=day,surface=surface,player_a_id=a,player_b_id=b,a_won=won,rank_a=10,rank_b=20)
    winner,loser=(a,b) if won else (b,a)
    raw=dict(tourney_id=tid,match_num=number,tourney_date=day,surface=surface,winner_id=winner,loser_id=loser,
        w_ace=10,w_df=3,w_svpt=100,w_1stIn=60,w_1stWon=45,w_2ndWon=20,
        l_ace=5,l_df=4,l_svpt=100,l_1stIn=60,l_1stWon=35,l_2ndWon=15)
    if not valid:raw['w_ace']=999
    return inv,(year,raw)


def fixture(records,target_day='2023-06-01',a='A',b='B'):
    target,_=match(target_day,999,a,b)
    t=pd.DataFrame([{**target,'match_id':target['match_id'][5:]}],index=[19])
    inventory=pd.DataFrame([r[0] for r in records]+[target])
    return t,inventory,[r[1] for r in records]


class SequenceFeatureTests(unittest.TestCase):
    def test_same_date_aggregation_manual_rates_and_order(self):
        records=[match('2023-05-01',1),match('2023-05-01',2,won=0)]
        t,inv,raw=fixture(records);s=build_sequence_features(t,inv,raw_rows=raw)
        self.assertEqual(s.values.shape,(1,2,16,22));self.assertEqual(s.mask.sum(),2)
        token=dict(zip(COLUMNS,s.values[0,0,-1]))
        self.assertAlmostEqual(token['win_rate'],.5)
        self.assertAlmostEqual(token['serve_rate'],(65+50+65)/300,places=6)
        self.assertAlmostEqual(token['return_rate'],(50+35+35)/300,places=6)
        self.assertAlmostEqual(token['log_matches'],np.log(3),places=6)
        reverse=build_sequence_features(t,inv.iloc[::-1],raw_rows=raw[::-1])
        np.testing.assert_array_equal(s.values,reverse.values)

    def test_cutoff_inclusive_future_target_poison_and_final_guard(self):
        records=[match('2023-05-11',1),match('2023-05-12',2)]
        t,inv,raw=fixture(records);s=build_sequence_features(t,inv,raw_rows=raw)
        self.assertEqual(s.mask.sum(),2)
        raw[1][1].update(w_svpt='POISON',winner_id=object())
        # The target frame's outcome is never consulted.
        t['a_won']=object();t['w_svpt']=object();t['score']=object()
        inv['a_won']=inv.a_won.astype(object)
        inv.loc[inv.tourney_date.eq('2023-05-12'),'a_won']='POISON'
        poison=build_sequence_features(t,inv,raw_rows=raw+[(2024,{}),(2025,{})])
        np.testing.assert_array_equal(s.values,poison.values)
        t.loc[19,'season']=2024
        with self.assertRaises(ValueError):build_sequence_features(t,inv,raw_rows=[])

    def test_window_and_last16_no_arbitrary_tie_order(self):
        cutoff=pd.Timestamp('2023-05-11');first=cutoff-pd.Timedelta(days=1460)
        records=[match(str(first.date()),1),match(str((first-pd.Timedelta(days=1)).date()),2)]
        t,inv,raw=fixture(records);s=build_sequence_features(t,inv,raw_rows=raw)
        self.assertEqual(s.mask.sum(),2)
        self.assertEqual(s.dates[0,0,-1],first.date().toordinal())
        many=[match(str((cutoff-pd.Timedelta(days=7*i)).date()),i+1) for i in range(20)]
        t,inv,raw=fixture(many);s=build_sequence_features(t,inv,raw_rows=raw)
        self.assertEqual(s.mask.sum(),32)
        self.assertEqual(s.dates[0,0,0],(cutoff-pd.Timedelta(days=7*15)).date().toordinal())
        self.assertEqual(s.values[0,0,0,-2],0)

    def test_missing_statistics_retains_outcome_ranks_and_masks(self):
        t,inv,raw=fixture([match('2023-05-01',valid=False)])
        s=build_sequence_features(t,inv,raw_rows=raw);z=dict(zip(COLUMNS,s.values[0,0,-1]))
        self.assertAlmostEqual(z['win_rate'],2/3,places=6)
        self.assertEqual(z['missing_stats_fraction'],1)
        self.assertEqual(z['log_serve_points'],0)
        self.assertAlmostEqual(z['serve_rate'],.65,places=6)
        self.assertAlmostEqual(z['log_own_rank'],np.log(10),places=6)
        self.assertEqual(s.metadata['missing_statistic_matches'],1)

    def test_zero_history_swap_and_roundtrip(self):
        t,inv,raw=fixture([match('2023-05-01')],a='A',b='UNKNOWN')
        s=build_sequence_features(t,inv,raw_rows=raw)
        self.assertFalse(s.mask[0,1].any());self.assertTrue((s.values[0,1]==0).all())
        swapped=t.copy();swapped[['player_a_id','player_b_id']]=t[['player_b_id','player_a_id']]
        r=build_sequence_features(swapped,inv,raw_rows=raw)
        np.testing.assert_array_equal(r.values,s.swapped().values)
        np.testing.assert_array_equal(r.dates,s.swapped().dates)
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'sequence.npz';save_sequences(p,s);loaded=load_sequences(p)
            np.testing.assert_array_equal(loaded.values,s.values)
            np.testing.assert_array_equal(loaded.match_ids,s.match_ids)
            self.assertEqual(loaded.columns,COLUMNS)

    def test_earlier_query_not_changed_by_later_target_source_statistics(self):
        h1=match('2022-01-01',1);h2=match('2023-01-01',2)
        t1,i1,_=fixture([h1,h2],target_day='2022-06-01')
        t2,i2,_=fixture([h1,h2],target_day='2023-06-01')
        targets=pd.concat([t1,t2],ignore_index=True)
        inventory=pd.concat([i1,i2]).drop_duplicates('match_id')
        earlier_target_raw=match('2022-06-01',999)[1]
        raw=[h1[1],h2[1],earlier_target_raw]
        original=build_sequence_features(targets,inventory,raw_rows=raw)
        h2[1][1].update(w_ace=0,w_1stWon=1,w_2ndWon=1)
        changed=build_sequence_features(targets,inventory,raw_rows=raw)
        np.testing.assert_array_equal(original.values[0],changed.values[0])
        self.assertGreater(np.max(np.abs(original.values[1]-changed.values[1])),0)

    def test_source_identity_rank_outcome_mismatch_and_duplicate_fail(self):
        t,inv,raw=fixture([match('2023-05-01')])
        changed=inv.copy();changed.loc[0,'rank_a']=np.nan
        with self.assertRaises(ValueError):build_sequence_features(t,changed,raw_rows=raw)
        changed=inv.copy();changed.loc[0,'a_won']=0
        with self.assertRaises(ValueError):build_sequence_features(t,changed,raw_rows=raw)
        with self.assertRaises(ValueError):build_sequence_features(t,inv,raw_rows=raw+raw)
        with self.assertRaises(ValueError):build_sequence_features(t,inv,raw_rows=[])


if __name__=='__main__':unittest.main()
