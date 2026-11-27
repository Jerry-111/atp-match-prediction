"""Winner data may test a frozen match assignment, never choose one."""
import unittest

import pandas as pd

from scripts.audit_round6_labels import audit_year, load_pinned_year


def fixture():
    targets = pd.DataFrame([
        dict(match_id='2021-event:1', season=2021, tourney_date='2021-02-01',
             tourney_id='2021-event', round='R32', player_a_id='A', player_b_id='B',
             player_a_name='A Person', player_b_name='B Person', a_won=1),
        dict(match_id='2021-event:2', season=2021, tourney_date='2021-02-01',
             tourney_id='2021-event', round='R32', player_a_id='C', player_b_id='D',
             player_a_name='C Person', player_b_name='D Person', a_won=0),
    ])
    linked = pd.DataFrame([
        dict(match_id='2021-event:1', season=2021, status='matched',
             target_event_id='2021-event', market_match_id=101., market_event_id=44.,
             market_player_a_id=2., market_player_b_id=1., market_match_utc='2021-02-03T12:00:00+00:00'),
        dict(match_id='2021-event:2', season=2021, status='matched',
             target_event_id='2021-event', market_match_id=102., market_event_id=44.,
             market_player_a_id=3., market_player_b_id=4., market_match_utc='2021-02-03T14:00:00+00:00'),
    ])
    source = pd.DataFrame([
        dict(match_id='101', date='2021-02-03 12:00:00', tournoi_id='44',
             joueur1_id='1', joueur2_id='2', vainqueur_id='2'),
        dict(match_id='102', date='2021-02-03 14:00:00', tournoi_id='44',
             joueur1_id='4', joueur2_id='3', vainqueur_id='4'),
    ])
    return targets, linked, source


class SourceAuditTests(unittest.TestCase):
    def test_both_source_orientations_and_input_immutability(self):
        t, l, s = fixture()
        before = [x.copy(deep=True) for x in (t, l, s)]
        rows, summary = audit_year(t, l, s, 2021)
        self.assertEqual(summary['agreements'], 2)
        self.assertEqual(rows.source_a_won.tolist(), [1, 0])
        self.assertEqual(summary['winner_source_side_counts'], {'joueur2': 1, 'joueur1': 1})
        for a, b in zip((t, l, s), before):
            pd.testing.assert_frame_equal(a, b)

    def test_winner_change_flags_fixed_match_without_rematching(self):
        t, l, s = fixture()
        s.loc[0, 'vainqueur_id'] = '1'
        rows, summary = audit_year(t, l, s, 2021)
        self.assertEqual(summary['winner_disagreements'], 1)
        self.assertEqual(rows.market_match_id.tolist(), [101., 102.])
        self.assertEqual(rows.a_won.tolist(), [1, 0])

    def test_unmatched_rows_stay_present_and_unread(self):
        t, l, s = fixture()
        l.loc[0, 'status'] = 'name_unmatched_or_ambiguous'
        s.loc[0, 'vainqueur_id'] = '99999'
        rows, summary = audit_year(t, l, s, 2021)
        self.assertEqual(rows.match_id.tolist(), t.match_id.tolist())
        self.assertEqual(summary['unmatched'], 1)
        self.assertTrue(pd.isna(rows.loc[0, 'source_winner_id']))
        self.assertEqual(rows.loc[0, 'audit_status'], 'unmatched_not_audited')

    def test_missing_unrelated_and_fractional_winners(self):
        for winner, expected in [(None, 'winner_missing_or_invalid'),
                                 ('99', 'winner_not_in_source_pair'),
                                 ('2.5', 'winner_missing_or_invalid')]:
            with self.subTest(winner=winner):
                t, l, s = fixture()
                s.loc[0, 'vainqueur_id'] = winner
                rows, _ = audit_year(t, l, s, 2021)
                self.assertEqual(rows.loc[0, 'audit_status'], expected)
                self.assertTrue(pd.isna(rows.loc[0, 'source_a_won']))

    def test_source_pair_event_and_time_conflicts(self):
        for column, value, issue in [('joueur2_id', '99', 'source_player_pair_changed'),
                                     ('tournoi_id', '99', 'source_event_changed'),
                                     ('date', '2021-02-04 12:00:00', 'source_datetime_changed_or_invalid')]:
            with self.subTest(column=column):
                t, l, s = fixture()
                s.loc[0, column] = value
                rows, _ = audit_year(t, l, s, 2021)
                self.assertEqual(rows.loc[0, 'audit_status'], 'structural_conflict')
                self.assertIn(issue, rows.loc[0, 'issues'])

    def test_duplicate_source_rows_and_reused_assignments(self):
        t, l, s = fixture()
        rows, _ = audit_year(t, l, pd.concat([s, s.iloc[[0]]]), 2021)
        self.assertIn('source_match_id_duplicated', rows.loc[0, 'issues'])
        l.loc[1, 'market_match_id'] = 101.
        rows, _ = audit_year(t, l, s, 2021)
        self.assertTrue(rows.issues.str.contains('source_match_reused').all())

    def test_order_and_final_year_guards(self):
        t, l, s = fixture()
        with self.assertRaisesRegex(ValueError, 'order'):
            audit_year(t, l.iloc[::-1], s, 2021)
        with self.assertRaisesRegex(ValueError, '2021--2023'):
            audit_year(t, l, s, 2024)
        with self.assertRaisesRegex(ValueError, '2021--2023'):
            load_pinned_year('/nonexistent', 2024, {'season': 2024})
        t.loc[0, 'tourney_date'] = '2024-01-01'
        with self.assertRaisesRegex(ValueError, 'date'):
            audit_year(t, l, s, 2021)
        t, l, s = fixture()
        l.loc[0, 'market_match_utc'] = s.loc[0, 'date'] = '2024-01-01T12:00:00+00:00'
        rows, _ = audit_year(t, l, s, 2021)
        self.assertIn('source_date_outside_development_season', rows.loc[0, 'issues'])


if __name__ == '__main__':
    unittest.main()
