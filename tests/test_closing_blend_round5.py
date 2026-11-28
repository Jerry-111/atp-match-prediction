"""Fixed closing-price compositions: no refit, exact fallback, correct symmetry."""
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from scipy.special import expit, logit

from scripts.closing_blend_round5 import predict_closing_controls, specs


class ClosingBlendRoundFiveTests(unittest.TestCase):
    def fixture(self):
        bases = {'round4': np.array([.2, .7, .4, .9]),
                 'round5_exploratory': np.array([.3, .6, .8, .75])}
        market = pd.DataFrame({'market_close_logit_diff': [-.8, .3, 1., -2.],
                               'market_close_available': [1., 0., 1., 0.],
                               'market_close_overround': [.03, 0., .04, 0.],
                               'market_open_logit_diff': ['NOT USED']*4}, index=[8, 4, 19, 2])
        return bases, market

    def test_registered_formulas_match_independent_compositions_without_fit(self):
        candidates = specs()
        self.assertEqual(len(candidates), 16)
        self.assertEqual(len({c['name'] for c in candidates}), 16)
        self.assertEqual({c['weight'] for c in candidates if c['kind']=='blend'}, {.25, .5, .75})
        bases, market = self.fixture()
        with patch('scripts.train_market_round5.MarketModel.fit', side_effect=AssertionError('No fitting allowed')):
            actual = predict_closing_controls(bases, market)
        good = market.market_close_available.eq(1).to_numpy()
        quote = expit(market.market_close_logit_diff.to_numpy())
        for spec in candidates:
            h = bases[spec['base']]
            if spec['kind']=='history': candidate = h
            elif spec['kind']=='opening': candidate = quote
            elif spec['mode']=='probability': candidate = spec['weight']*quote+(1-spec['weight'])*h
            else: candidate = expit(spec['weight']*logit(quote)+(1-spec['weight'])*logit(h))
            expected = h.copy(); expected[good] = candidate[good]
            np.testing.assert_allclose(actual[spec['name']], expected, atol=1e-15, rtol=0)
            np.testing.assert_array_equal(actual[spec['name']].to_numpy()[~good], h[~good])
        self.assertTrue(actual.index.equals(market.index))

    def test_exchange_complements_all_sixteen_and_ignores_opening_or_labels(self):
        bases, market = self.fixture()
        actual = predict_closing_controls(bases, market)
        reversed_market = market.copy()
        reversed_market['market_close_logit_diff'] *= -1
        reversed_bases = {key: 1-value for key, value in bases.items()}
        np.testing.assert_allclose(actual+predict_closing_controls(reversed_bases, reversed_market), 1., atol=1e-15, rtol=0)
        poisoned = market.assign(a_won='POISON', score='POISON', market_open_available='POISON')
        pd.testing.assert_frame_equal(actual, predict_closing_controls(bases, poisoned))

    def test_invalid_schema_prices_and_prior_alignment_rejected(self):
        bases, market = self.fixture()
        bad_inputs = [market.drop(columns='market_close_available'),
                      pd.concat([market, market[['market_close_available']]], axis=1),
                      market.assign(market_close_available=.5),
                      market.assign(market_close_logit_diff=np.nan)]
        for frame in bad_inputs:
            with self.assertRaises(ValueError):
                predict_closing_controls(bases, frame)
        with self.assertRaises(ValueError):
            predict_closing_controls({**bases, 'round4': np.array([.5])}, market)
        with self.assertRaises(ValueError):
            predict_closing_controls({**bases, 'round4': np.ones(4)}, market)


if __name__ == '__main__':
    unittest.main()
