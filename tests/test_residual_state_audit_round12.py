"""Independent oracle boundary tests; no real-data or experimental model fits."""
import unittest

import numpy as np
import pandas as pd

from scripts.audit_residual_state_round12 import (
    REFERENCE, STRUCTURES, independent_metrics, independent_nodes,
    independent_outer_indices, independent_prefix_indices,
    independent_selection, independent_state_prior, quarter_start,
    independent_design, audit_transform,
    independent_snapshot, independent_prediction,
    _transform_hash,
)


def metadata(years, days=None):
    return pd.DataFrame(dict(match_id=[f'm{i}' for i in range(len(years))], season=years,
        tourney_date=days or [f'{y}-06-01' for y in years],
        player_a_id=['A'] * len(years), player_b_id=['B'] * len(years)))


class IndependentAuditTest(unittest.TestCase):
    def test_scoring_identity_threshold_and_endpoints(self):
        result = independent_metrics([0, 1, 1], [0, 1, .5])
        self.assertEqual(result['correct'], 3)
        self.assertAlmostEqual(result['mean_log_loss'], np.log(2) / 3)
        for p in ([np.nan], [1.01], [-.01]):
            with self.assertRaises(ValueError):
                independent_metrics([0], p)

    def test_selector_uses_integer_gate_13_not_12(self):
        rows = metadata([2020] * 2 + [2021] * 2 + [2022] * 2528)
        rows['a_won'] = 1
        p = pd.DataFrame(.499, index=rows.index, columns=(*STRUCTURES, REFERENCE))
        p.loc[:3, STRUCTURES[0]] = .501
        p.loc[4:15, STRUCTURES[0]] = .501
        before = independent_selection(rows, p)
        self.assertEqual(before['required_additional_correct'], 13)
        self.assertEqual(before['additional_correct'], 12)
        self.assertFalse(before['gate_passed'])
        p.loc[16, STRUCTURES[0]] = .501
        after = independent_selection(rows, p)
        self.assertTrue(after['gate_passed'])
        self.assertEqual(after['recommended'], STRUCTURES[0])

    def test_loss_guard_can_reject_more_correct_predictions(self):
        rows = metadata([2020, 2020, 2021, 2021, 2022, 2022])
        rows['a_won'] = [1, 1, 0, 0, 1, 0]
        p = pd.DataFrame(.5, index=rows.index, columns=(*STRUCTURES, REFERENCE))
        p[STRUCTURES[0]] = [.9, .9, .1, .999999, .9, .1]
        result = independent_selection(rows, p)
        self.assertNotEqual(result['early_selected'], STRUCTURES[0])

    def test_selector_rejects_future_season_and_index_misalignment(self):
        rows = metadata([2020, 2021, 2022]); rows['a_won'] = [0, 1, 0]
        p = pd.DataFrame(.5, index=rows.index, columns=(*STRUCTURES, REFERENCE))
        with self.assertRaises(ValueError): independent_selection(rows, p.iloc[::-1])
        rows.loc[2, 'season'] = 2023
        with self.assertRaises(ValueError): independent_selection(rows, p)

    def test_exact_tie_uses_lexicographic_name(self):
        rows = metadata([2020, 2021, 2022]); rows['a_won'] = [1, 0, 1]
        p = pd.DataFrame(.5, index=rows.index, columns=(*STRUCTURES, REFERENCE))
        self.assertEqual(independent_selection(rows, p)['early_selected'], min(p.columns))

    def test_prefix_does_not_read_future_labels(self):
        class Poison:
            def __float__(self): raise AssertionError('Forbidden outcome read')
            def __int__(self): raise AssertionError('Forbidden outcome read')
        rows = metadata([2021, 2022, 2022], ['2021-12-01', '2022-01-01', '2022-01-22'])
        rows['a_won'] = [1, 0, Poison()]
        self.assertEqual(list(independent_prefix_indices(rows, '2022-01-01')), [0, 1])
        fit, cutoff = independent_outer_indices(rows, 2022)
        self.assertEqual(list(fit), [0]); self.assertEqual(cutoff, '2021-12-11')

    def test_opening_week_uses_calendar_quarter_not_season(self):
        rows = metadata([2013], ['2012-12-30'])
        keys, days = independent_nodes(rows, 'dynamic')
        self.assertEqual(days, ['2012-10-01', '2012-10-01'])
        self.assertEqual(quarter_start('2012-12-30'), '2012-10-01')
        rows.loc[0, 'tourney_date'] = '2012-06-01'
        with self.assertRaises(ValueError): independent_nodes(rows, 'dynamic')

    def test_no_unobserved_quarters_and_elapsed_time_prior(self):
        rows = metadata([2021, 2022], ['2021-01-02', '2022-07-02'])
        keys, days = independent_nodes(rows, 'dynamic')
        self.assertEqual(len(keys), 4)
        u = np.array([.1, .2, -.2, -.1])
        value, gradient, links = independent_state_prior(keys, days, u, 'dynamic')
        self.assertEqual(len(links), 2)
        self.assertAlmostEqual(links[0][2], 546 / 365.25)
        for index in range(len(u)):
            step = np.eye(len(u))[index] * 1e-6
            plus = independent_state_prior(keys, days, u + step, 'dynamic')[0]
            minus = independent_state_prior(keys, days, u - step, 'dynamic')[0]
            self.assertAlmostEqual((plus - minus) / 2e-6, gradient[index], places=8)
        self.assertGreater(value, 0)

    def test_static_has_one_prior_per_player_and_no_links(self):
        rows = metadata([2020, 2021, 2022])
        keys, days = independent_nodes(rows, 'static')
        value, gradient, links = independent_state_prior(keys, days, [.35, -.35], 'static')
        self.assertEqual(len(keys), 2); self.assertEqual(links, [])
        self.assertAlmostEqual(value, 1.)
        np.testing.assert_allclose(gradient, [1 / .35, -1 / .35])

    def test_sealed_history_metadata_rejected_before_labels(self):
        rows = metadata([2024]); rows['a_won'] = 'never parse this'
        with self.assertRaises(ValueError): independent_prefix_indices(rows, '2023-12-01')

    def test_manual_bspline_design_matches_saved_old_transform(self):
        from scripts.round2_learning import candidate_specs, make_model
        rng = np.random.default_rng(158)
        x = pd.DataFrame({'log_rank_diff': rng.normal(size=80),
                          'log_rank_mean': rng.normal(size=80),
                          'surface_rate_diff': rng.normal(size=80)})
        spec = next(s for s in candidate_specs(x) if s['params']['C'] == .01
                    and s['params']['half_life'] == 3.)
        model = make_model(spec).fit(x, np.arange(len(x)) % 2, np.repeat(2021, len(x)))
        query = x.copy()
        query.loc[:3, 'log_rank_diff'] = [-100, -3, 3, 100]
        expected = model._design(query)
        np.testing.assert_allclose(independent_design(model, query), expected, rtol=0, atol=1e-13)
        self.assertLess(audit_transform(model, x), 1e-13)
        swapped = query.copy()
        swapped[['log_rank_diff', 'surface_rate_diff']] *= -1
        np.testing.assert_allclose(independent_design(model, swapped), -expected, rtol=0, atol=1e-13)

    def test_saved_state_math_has_independent_likelihood_and_node_checks(self):
        from scripts.residual_state_round12 import fit_shared_control, ResidualStateModel
        from scripts.round2_learning import candidate_specs
        rng = np.random.default_rng(158)
        rows = metadata([2020] * 20 + [2021] * 20 + [2022] * 20)
        rows['a_won'] = np.arange(60) % 2
        x = pd.DataFrame({'log_rank_diff': rng.normal(size=60),
                          'log_rank_mean': rng.normal(size=60)})
        spec = next(s for s in candidate_specs(x) if s['params']['C'] == .01
                    and s['params']['half_life'] == 3.)
        control = fit_shared_control(spec, x, rows)
        self.assertEqual(_transform_hash(control), control.residual_transform_sha256_)
        query = metadata([2023], ['2023-01-23'])
        qx = x.iloc[[0]].copy(); qx.index = query.index
        for kind in ('static', 'dynamic'):
            model = ResidualStateModel(spec, kind).fit(x, rows, shared_model=control)
            detail = independent_snapshot(model.training_state_, rows,
                independent_design(control, x), model.beta_, joint=True)
            self.assertLess(detail['objective_error'], 1e-10)
            actual = model.predict_asof(qx, query, x, rows)
            manual, _ = independent_prediction(model, qx, query)
            np.testing.assert_allclose(actual, manual, rtol=0, atol=1e-14)
            snapshot = model.state_snapshots_['2023-01-02']
            ordered = rows.sort_values(['tourney_date', 'match_id'])
            independent_snapshot(snapshot, ordered, independent_design(control, x.loc[ordered.index]),
                                 model.beta_, joint=False)
            snapshot.node_dates = tuple(['2009-01-01'] * len(snapshot.node_dates))
            with self.assertRaises(AssertionError):
                independent_snapshot(snapshot, ordered, independent_design(control, x.loc[ordered.index]),
                                     model.beta_, joint=False)


if __name__ == '__main__':
    unittest.main()
