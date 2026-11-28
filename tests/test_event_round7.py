import numpy as np
import pandas as pd
import pytest

from scripts.train_event_round7 import candidate_specs, event_fallback


def test_fallback_preserves_full_cohort_and_exact_old_probabilities():
    x = pd.DataFrame({'event_any_prior': [0, 1, 0, 1]})
    old = np.array([.12, .23, .34, .45])
    new = np.array([.9, .8, .7, .6])
    p = event_fallback(x, new, old)
    np.testing.assert_array_equal(p, [.12, .8, .34, .6])
    np.testing.assert_allclose(event_fallback(x, 1-new, 1-old), 1-p)
    with pytest.raises(ValueError):
        event_fallback(x, new[:-1], old)
    with pytest.raises(ValueError):
        event_fallback(x.assign(event_any_prior=2), new, old)


def test_twelve_fixed_configurations_have_matched_groups():
    groups = dict(base=['log_rank_diff', 'elo_k32_diff'], all_previous=['log_rank_diff', 'elo_k32_diff'],
        context=['event_prior_matches_diff'], load=['event_sets_diff'], performance=['event_point_share_diff'])
    specs = candidate_specs(groups)
    assert len(specs) == 12
    for method in ('linear', 'spline'):
        chosen = {s['name']: s for s in specs if s['method'] == method}
        assert len(chosen) == 5
        context = chosen[f'event_context_{method}']['columns']
        for group, extra in (('load', 'event_sets_diff'), ('performance', 'event_point_share_diff')):
            assert set(chosen[f'event_{group}_{method}']['columns']) == set(context) | {extra}
