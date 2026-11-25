import copy
import unittest

import pandas as pd

from scripts.paper_profile_features import build_paper_profile_features, profile_row


class PaperProfileTests(unittest.TestCase):
    def setUp(self):
        self.target=dict(match_id='2010-x:1',season=2010,tourney_date='2010-06-01',player_a_id='A',player_b_id='B')
        self.raw=dict(tourney_id='2010-x',match_num='1',tourney_date='20100601',winner_id='B',loser_id='A',
                      winner_age='35',loser_age='25',winner_rank_points='1000',loser_rank_points='2000')

    def test_paper_age_transformations_and_orientation(self):
        x=profile_row(self.target,self.raw)
        self.assertEqual(x['profile_age_diff'],-10)
        self.assertEqual(x['profile_age30_diff'],0)
        self.assertEqual(x['profile_age_interval_diff'],0)
        self.assertEqual(x['profile_points_diff'],1)
        self.assertGreater(x['profile_log_points_diff'],0)
        other=dict(self.target,player_a_id='B',player_b_id='A')
        z=profile_row(other,self.raw)
        for k in x:
            self.assertAlmostEqual(z[k],-x[k] if k.endswith('_diff') else x[k])

    def test_score_outcome_stats_are_irrelevant(self):
        changed=dict(self.raw,score='POISON',w_svpt='POISON',l_svpt='POISON')
        self.assertEqual(profile_row(self.target,self.raw),profile_row(dict(self.target,a_won=999),changed))

    def test_missingness_preserves_row_without_learned_imputation(self):
        raw=dict(self.raw,winner_age='',loser_rank_points='nan')
        x=profile_row(self.target,raw)
        self.assertEqual(x['profile_age_missing_sum'],1)
        self.assertEqual(x['profile_points_missing_sum'],1)
        frame=build_paper_profile_features(pd.DataFrame([self.target]),raw_rows=[(2010,raw),(2024,{'bad':True})])
        self.assertEqual(frame.shape,(1,11))

    def test_identity_and_test_guards(self):
        with self.assertRaises(ValueError):
            profile_row(dict(self.target,player_a_id='C'),self.raw)
        with self.assertRaises(ValueError):
            build_paper_profile_features(pd.DataFrame([dict(self.target,season=2024)]),raw_rows=[])
        with self.assertRaises(ValueError):
            build_paper_profile_features(pd.DataFrame([self.target]),raw_rows=[])

    def test_audited_event_profile_is_masked_across_mixed_date_snapshots(self):
        target = dict(match_id='2019-418:1', season=2019, tourney_date='2019-07-29',
                      player_a_id='KE17', player_b_id='GB93')
        early = dict(tourney_id='2019-418', match_num='1', tourney_date='20190727',
                     round='R64', winner_id='KE17', loser_id='GB93',
                     winner_age='24.249', loser_age='28.953', winner_rank_points='1000',
                     loser_rank_points='418')
        source_before = copy.deepcopy(early)
        late = dict(early, tourney_date='20190729', round='QF',
                    winner_rank_points='975', loser_rank_points='393',
                    winner_age='24.255', loser_age='28.958')
        expected = profile_row(target, early)
        self.assertEqual(profile_row(target, late), expected)
        self.assertEqual(early, source_before)
        self.assertEqual(expected['profile_points_diff'], 0.)
        self.assertEqual(expected['profile_log_points_diff'], 0.)
        self.assertEqual(expected['profile_log_points_mean'], 0.)
        self.assertEqual(expected['profile_points_missing_sum'], 2.)
        self.assertEqual(expected['profile_points_missing_diff'], 0.)
        self.assertEqual(expected['profile_age_diff'], 0.)
        self.assertEqual(expected['profile_age30_diff'], 0.)
        self.assertEqual(expected['profile_age_interval_diff'], 0.)
        self.assertEqual(expected['profile_age_mean'], 30.)
        self.assertEqual(expected['profile_age_missing_sum'], 2.)
        swapped = profile_row(dict(target, player_a_id='GB93', player_b_id='KE17'), late)
        for name, value in expected.items():
            self.assertEqual(swapped[name], -value if name.endswith('_diff') else value)

        frame = build_paper_profile_features(pd.DataFrame([target]), raw_rows=[(2019, late)])
        self.assertEqual(frame.attrs['points_snapshot_masked_events'], ['2019-418'])
        self.assertEqual(frame.attrs['points_snapshot_masked_appearances'], 2)
        self.assertEqual(frame.attrs['missing_points_appearances'], 2)
        self.assertEqual(frame.attrs['age_snapshot_masked_events'], ['2019-418'])
        self.assertEqual(frame.attrs['age_snapshot_masked_appearances'], 2)
        self.assertEqual(frame.attrs['missing_age_appearances'], 2)
        self.assertEqual(frame.attrs['paper_preprint_url'], 'https://arxiv.org/abs/2502.01613v2')
        self.assertIn('verified pre-event rank corrections', frame.attrs['points_snapshot_policy'])

    def test_normal_event_points_and_provenance_are_preserved(self):
        frame = build_paper_profile_features(pd.DataFrame([self.target]), raw_rows=[(2010, self.raw)])
        self.assertEqual(frame.iloc[0]['profile_points_diff'], 1.)
        self.assertEqual(frame.iloc[0]['profile_age_diff'], -10.)
        self.assertEqual(frame.attrs['points_snapshot_masked_events'], [])
        self.assertEqual(frame.attrs['points_snapshot_masked_appearances'], 0)
        self.assertEqual(frame.attrs['age_snapshot_masked_appearances'], 0)
