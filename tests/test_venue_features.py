import copy
from datetime import date
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from scipy.special import expit, logit

from scripts.venue_features import (STYLES, STRENGTH, build_venue_features,
    context_values, count_pair, expected_rates, venue_key)


def match(year, number=1, name='Venue', suffix='1', day=None, ace=10):
    day = day or f'{year}-06-01'
    tid = f'{year}-{suffix}'
    inv = dict(match_id=f'tour:{tid}:{number}', source='tour', season=year,
        tourney_id=tid, tourney_name=name, tourney_date=day, surface='Hard',
        player_a_id='A', player_b_id='B', a_won='DO NOT READ')
    raw = dict(tourney_id=tid, match_num=number, tourney_date=day, surface='Hard',
        winner_id='A', loser_id='B', w_1stWon=40, w_2ndWon=25, w_svpt=100,
        l_1stWon=40, l_2ndWon=25, l_svpt=100, w_ace=ace, l_ace=ace)
    return inv, (year, raw)


class VenueFeaturesTests(unittest.TestCase):
    def fixture(self, year=2023, day=None):
        target, _ = match(year, day=day)
        targets = pd.DataFrame([{**target, 'match_id': target['match_id'][5:]}], index=[9])
        x = pd.DataFrame([{c: (i+1)/10 for i, c in enumerate(STYLES+STRENGTH)}], index=[9])
        return targets, x, target

    def build(self, matches, year=2023, day=None):
        targets, x, target = self.fixture(year, day)
        inventory = pd.DataFrame([target]+[m[0] for m in matches])
        return build_venue_features(targets, inventory, x, raw_rows=[m[1] for m in matches])

    def test_neutral_priors_and_manual_expected_adjustment(self):
        np.testing.assert_allclose(expected_rates([0]*4, [0]*4, [0]*4), [.65, .06], atol=1e-15)
        actual = expected_rates([80, 100, 10, 100], [60, 100, 2, 100], [130, 200, 12, 200])
        expected = expit(logit([145/200, 16/200])+logit([125/200, 8/200])-logit([195/300, 18/300]))
        np.testing.assert_allclose(actual, expected, atol=1e-15)
        # Observed == expected removes a rate shift, even with extreme exposure.
        v = [160, 160, 200, 1, 30, 30, 200, 1]
        self.assertTrue(np.array_equal(context_values(v, v, v)['adjusted'], [0., 0.]))

    def test_three_prior_seasons_weighting_and_current_season_not_used(self):
        matches = [match(2019), match(2020), match(2021), match(2022)]
        output = self.build(matches)
        self.assertAlmostEqual(output.iloc[0].venue_exposure_serve, np.log1p(1200))
        self.assertAlmostEqual(output.iloc[0].venue_exposure_ace, np.log1p(1200))
        # Fixture poison is skipped before accessing any fields.
        targets, x, target = self.fixture()
        inventory = pd.DataFrame([target]+[m[0] for m in matches]+[{'source':'tour','season':2024}])
        poisoned = build_venue_features(targets, inventory, x,
            raw_rows=[m[1] for m in matches]+[(2023, {}), (2024, {}), (2025, {})])
        pd.testing.assert_frame_equal(output, poisoned)

    def test_lag_inclusive_and_historical_expectations_not_retroactive(self):
        targets, x, target = self.fixture(2023, '2022-12-26')
        eligible = match(2022, day='2022-12-05', ace=20)
        newer = match(2022, number=2, day='2022-12-06', ace=30)
        inventory = pd.DataFrame([target, eligible[0], newer[0]])
        out = build_venue_features(targets, inventory, x, raw_rows=[eligible[1], newer[1]])
        self.assertAlmostEqual(out.iloc[0].venue_exposure_serve, np.log1p(600))
        newer[1][1].update(w_1stWon=90, w_2ndWon=0, w_ace=80)
        altered = build_venue_features(targets, inventory, x, raw_rows=[eligible[1], newer[1]])
        pd.testing.assert_frame_equal(out, altered)
        # An old match's expected-rate adjustment cannot see a later match.
        prior = match(2021, day='2021-06-01', ace=2)
        current = match(2022, day='2022-06-01', ace=30)
        old_targets, old_x, old = self.fixture(2022)
        inv = pd.DataFrame([prior[0], current[0]])
        two_t = pd.concat([old_targets.reset_index(drop=True), targets.reset_index(drop=True)], ignore_index=True)
        two_x = pd.concat([old_x.reset_index(drop=True), x.reset_index(drop=True)], ignore_index=True)
        inv = pd.concat([inv, pd.DataFrame([target])], ignore_index=True)
        original = build_venue_features(two_t, inv, two_x, raw_rows=[prior[1], current[1]])
        current[1][1].update(w_ace=0, l_ace=0)
        changed = build_venue_features(two_t, inv, two_x, raw_rows=[prior[1], current[1]])
        np.testing.assert_array_equal(original.iloc[0], changed.iloc[0])

    def test_conservative_identity_and_missing_fields(self):
        self.assertEqual(venue_key('2022-1', "'S-Hértogenbosch", 'Grass'),
                         venue_key('2023-1', 's Hertogenbosch', 'Grass'))
        self.assertNotEqual(venue_key('2022-424', 'Memphis', 'Hard'),
                            venue_key('2023-424', 'Dallas', 'Hard'))
        output = self.build([match(2022, name='Other venue')]).iloc[0]
        self.assertEqual(output.venue_missing_serve, 1)
        self.assertEqual(output.venue_exposure_serve, 0)
        for column in output.index:
            if column.startswith(('venue_raw_', 'venue_adjusted_', 'venue_strength_')):
                self.assertEqual(output[column], 0)

    def test_ace_invalidity_does_not_destroy_valid_service_history(self):
        inv, raw = match(2022, ace=1000)
        self.assertEqual(count_pair(raw[1])[0].tolist(), [65., 100., 0., 0.])
        out = self.build([(inv, raw)]).iloc[0]
        self.assertEqual(out.venue_missing_serve, 0)
        self.assertEqual(out.venue_missing_ace, 1)
        raw[1]['w_svpt'] = 40
        self.assertIsNone(count_pair(raw[1]))
        self.assertEqual(self.build([(inv, raw)]).iloc[0].venue_missing_serve, 1)

    def test_swap_label_poison_and_source_order(self):
        matches = [match(2021, ace=10), match(2022, ace=2), match(2022, suffix='2', ace=15)]
        targets, x, target = self.fixture()
        inv = pd.DataFrame([target]+[m[0] for m in matches])
        original = build_venue_features(targets, inv, x, raw_rows=[m[1] for m in matches])
        targets[['player_a_id', 'player_b_id']] = [['B', 'A']]
        targets['a_won'] = object()
        inverse = build_venue_features(targets, inv, -x, raw_rows=[m[1] for m in matches[::-1]])
        parity = np.array([-1 if c.endswith('_diff') else 1 for c in original])
        np.testing.assert_array_equal(inverse, original.to_numpy()*parity)
        self.assertEqual(original.shape, (1, 50))
        self.assertEqual({k:len(v) for k,v in original.attrs['groups'].items()},
                         {'context':10,'surface':12,'raw':12,'adjusted':12,'strength':4})

    def test_loader_filters_years_before_open_and_identity_rejects(self):
        targets, x, target = self.fixture()
        inv, raw = match(2022)
        inventory = pd.DataFrame([target, inv])
        seen = []
        def loader(cache, entries):
            seen.extend(e['season'] for e in entries)
            return [raw]
        with patch('scripts.venue_features._annual_rows', side_effect=loader):
            build_venue_features(targets, inventory, x)
        self.assertEqual(seen, list(range(2006, 2023)))
        for changed in (targets.assign(season=2024), targets.assign(tourney_date='2024-01-01')):
            with self.assertRaises(ValueError):
                build_venue_features(changed, inventory, x, raw_rows=[raw])
        raw[1]['winner_id'] = 'WRONG'
        with self.assertRaisesRegex(ValueError, 'identity'):
            build_venue_features(targets, inventory, x, raw_rows=[raw])


if __name__ == '__main__':
    unittest.main()
