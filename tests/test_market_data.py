"""Outcome-blind market linkage, price symmetry and locked-source tests."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from scipy.special import expit

from scripts.market_data import build_market_features,de_vig,load_market_rows,normalize_name


class MarketDataTests(unittest.TestCase):
    def fixture(self):
        targets=[];source=[]
        for i,(a,b) in enumerate([('Alpha One','Beta Two'),('Gamma Three','Delta Four'),
                                   ('Epsilon Five','Zeta Six'),('Eta Seven','Theta Eight')]):
            targets.append(dict(match_id=f'2021-X:{i+1}',season=2021,tourney_id='2021-X',
                tourney_date='2021-02-01',surface='Hard',round='R32',
                player_a_name=a,player_b_name=b,player_a_id=f'A{i}',player_b_id=f'B{i}'))
            source.append(dict(match_id=100+i,source_season=2021,date=f'2021-02-0{i+1} 12:00:00',
                tournoi='Example Open',tournoi_id=50,categorie='Main tour',genre='atp',surface='dur',tour=4,
                joueur1=a,joueur1_id=2*i+1,joueur2=b,joueur2_id=2*i+2,
                cote1_ouverture=1.5,cote2_ouverture=2.8,cote1_cloture=1.6,cote2_cloture=2.5))
        return pd.DataFrame(targets,index=[17,2,11,45]),pd.DataFrame(source)

    def test_normalized_implied_probabilities_and_invalid_pairs(self):
        z,margin,valid=de_vig(1.5,2.8)
        self.assertAlmostEqual(expit(z),(1/1.5)/(1/1.5+1/2.8))
        self.assertAlmostEqual(margin,1/1.5+1/2.8-1)
        self.assertEqual(valid,1)
        reverse=de_vig(2.8,1.5)
        self.assertAlmostEqual(reverse[0],-z);self.assertEqual(reverse[1:],(margin,valid))
        for a,b in [(None,2),('bad',2),(1,2),(.9,2),(2,np.nan),(np.inf,2),(2,-3)]:
            self.assertEqual(de_vig(a,b),(0.,0.,0.))
        self.assertEqual(normalize_name('Félix Auger-Aliassime'),normalize_name('Felix Auger Aliassime'))

    def test_exact_identity_index_swap_and_permutation(self):
        target,source=self.fixture()
        x,m=build_market_features(target,market_rows=source)
        self.assertEqual(x.shape,(4,8));self.assertTrue(x.index.equals(target.index))
        self.assertTrue(m.status.eq('matched').all())
        self.assertEqual(list(m.market_match_id),['100','101','102','103'])
        swapped=target.copy()
        for suffix in ('id','name'):
            swapped[[f'player_a_{suffix}',f'player_b_{suffix}']]=target[[f'player_b_{suffix}',f'player_a_{suffix}']].to_numpy()
        reversed_x,reversed_m=build_market_features(swapped,market_rows=source.iloc[::-1])
        for name in x:
            np.testing.assert_allclose(reversed_x[name],-x[name] if name.endswith('_diff') else x[name])
        self.assertEqual(list(m.market_match_id),list(reversed_m.market_match_id))
        self.assertEqual(list(reversed_m.source_name_a),list(m.source_name_b))

    def test_target_and_source_outcomes_are_ignored_before_matching(self):
        target,source=self.fixture()
        expected,metadata=build_market_features(target,market_rows=source)
        poison=source.assign(vainqueur_id='POISON',score='POISON',duree_min='POISON')
        poison=pd.concat([poison,pd.DataFrame([dict(source_season=2024,genre='POISON',score='POISON')])])
        actual,other=build_market_features(target.assign(a_won=999,score='POISON'),market_rows=poison)
        pd.testing.assert_frame_equal(expected,actual)
        pd.testing.assert_frame_equal(metadata,other)

    def test_event_anchors_resolve_nearby_rematch_without_outcomes(self):
        target,source=self.fixture()
        extra=dict(source.iloc[0],match_id=999,tournoi_id=99,date='2021-02-09 12:00:00')
        x,m=build_market_features(target,market_rows=pd.concat([source,pd.DataFrame([extra])]))
        self.assertEqual(m.iloc[0].candidate_count,2)
        self.assertEqual(m.iloc[0].market_match_id,'100')
        self.assertEqual(m.iloc[0].event_anchor_count,3)
        self.assertTrue(x.market_open_available.eq(1).all())
        extra['tournoi_id']=50
        _,ambiguous=build_market_features(target,market_rows=pd.concat([source,pd.DataFrame([extra])]))
        self.assertEqual(ambiguous.iloc[0].status,'within_event_ambiguous')

    def test_event_conflict_or_reused_source_is_missing_not_guessed(self):
        target,source=self.fixture()
        conflicting=source.copy();conflicting.loc[0,'tournoi_id']=99
        x,m=build_market_features(target,market_rows=conflicting)
        self.assertTrue(x.market_open_available.eq(0).all())
        self.assertTrue(m.status.eq('event_anchor_ambiguous_or_insufficient').all())
        duplicate=dict(target.iloc[0],match_id='2021-X:99')
        repeated=pd.concat([target,pd.DataFrame([duplicate],index=[99])])
        x,m=build_market_features(repeated,market_rows=source)
        self.assertEqual(m.loc[17,'status'],'source_match_reused_ambiguous')
        self.assertEqual(m.loc[99,'status'],'source_match_reused_ambiguous')
        self.assertEqual(x.loc[17,'market_open_available'],0.)
        self.assertEqual(len(x),5)

    def test_missing_opening_closing_and_pre2021_keep_full_cohort(self):
        target,source=self.fixture()
        source.loc[0,['cote1_cloture','cote2_cloture']]=np.nan
        source.loc[1,'cote1_ouverture']=1.
        old=dict(target.iloc[0],match_id='2020-X:1',season=2020,tourney_id='2020-X',tourney_date='2020-02-01')
        target=pd.concat([target,pd.DataFrame([old],index=[99])])
        x,m=build_market_features(target,market_rows=source)
        self.assertEqual(len(x),5);self.assertEqual(x.loc[17,'market_close_available'],0.)
        self.assertEqual(x.loc[2,'market_open_available'],0.)
        self.assertTrue((x.loc[99]==0).all())
        self.assertEqual(m.loc[99,'status'],'outside_source_years')
        self.assertEqual(x.attrs['coverage_by_season']['2021']['opening_available'],3)

    def test_surface_date_round_and_name_collision_reject_candidates(self):
        target,source=self.fixture()
        for column,value in [('surface','gazon'),('date','2021-02-25 12:00:00'),('tour',9)]:
            changed=source.copy();changed.loc[0,column]=value
            x,m=build_market_features(target,market_rows=changed)
            self.assertEqual(m.iloc[0].status,'no_pair_date_surface_round_candidate')
            self.assertEqual(x.iloc[0].market_open_available,0.)
        collision=dict(source.iloc[1],match_id=999,joueur1='Alpha One',joueur1_id=99)
        _,m=build_market_features(target,market_rows=pd.concat([source,pd.DataFrame([collision])]))
        self.assertEqual(m.iloc[0].status,'name_unmatched_or_ambiguous')

    def test_holdout_and_corrupt_cache_block_before_fetch(self):
        target,source=self.fixture()
        with patch('scripts.market_data.load_market_rows') as loader:
            with self.assertRaisesRegex(ValueError,'holdout locked'):
                build_market_features(target.assign(season=2024))
            loader.assert_not_called()
        with patch('scripts.market_data.subprocess.run') as fetch:
            with self.assertRaisesRegex(ValueError,'holdout is locked'):
                load_market_rows(seasons=[2025])
            fetch.assert_not_called()
            with tempfile.TemporaryDirectory() as directory:
                Path(directory,'2021.csv').write_text('corrupt')
                with self.assertRaisesRegex(ValueError,'hash mismatch'):
                    load_market_rows(directory,seasons=[2021])
            fetch.assert_not_called()

    def test_duplicate_target_cannot_manufacture_third_event_anchor(self):
        target,source=self.fixture()
        target=target.iloc[:2].copy();source=source.iloc[:2].copy()
        repeated=pd.concat([target,pd.DataFrame([dict(target.iloc[0],match_id='2021-X:99')],index=[99])])
        x,m=build_market_features(repeated,market_rows=source)
        self.assertTrue(m.event_anchor_count.eq(2).all())
        self.assertTrue(x.market_open_available.eq(0).all())

    def test_player_crosswalk_must_be_bijective(self):
        target,source=self.fixture()
        # Source aliases with one ID cannot silently become two ATP players.
        source.loc[1,'joueur1_id']=source.loc[0,'joueur1_id']
        x,m=build_market_features(target,market_rows=source)
        self.assertEqual(x.iloc[0].market_open_available,0.)
        self.assertEqual(x.iloc[1].market_open_available,0.)
        self.assertIn('player_mapping_not_bijective',m.iloc[0].name_rule)

    def test_calendar_year_cannot_hide_final_holdout_under_old_season(self):
        target,source=self.fixture()
        with patch('scripts.market_data.load_market_rows') as loader:
            with self.assertRaisesRegex(ValueError,'holdout date'):
                build_market_features(target.assign(season=2023,tourney_date='2024-01-01'))
            loader.assert_not_called()
        source=source.assign(source_season=2023,date='2024-01-02 12:00:00')
        x,m=build_market_features(target.assign(season=2023,tourney_date='2023-12-31'),market_rows=source)
        self.assertTrue(x.market_open_available.eq(0).all())


if __name__=='__main__':unittest.main()
