"""Past-season tournament environment and player-style interactions.

The key is an ID/name/surface proxy, not a measured physical court speed.
Only preceding seasons contribute, and all observations obey T-21.
"""
from __future__ import annotations

from bisect import bisect_left, bisect_right
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date, timedelta
import re
import unicodedata

import numpy as np
import pandas as pd
from scipy.special import expit, logit

from scripts.advanced_features import _date
from scripts.baseline import DEFAULT_CACHE
from scripts.model_event_metadata import corrected_tour_row
from scripts.multilevel_features import _player, _season_date
from scripts.serve_features import _annual_rows, _counts, development_entries

VERSION = 'historical-venue-1.0'
LAG = 21
PLAYER_PRIOR = 100.
ENVIRONMENT_PRIOR = 2000.
PRIORS = np.array([.65, .06])
STYLES = ('style_ace_diff', 'style_first_won_diff', 'style_second_won_diff',
          'style_return_first_won_diff', 'style_return_second_won_diff', 'style_height_diff')
STRENGTH = ('log_rank_diff', 'blend_elo_k32_diff')


def venue_key(identifier, name, surface):
    identifier, name, surface = str(identifier), str(name), str(surface)
    parts = identifier.split('-', 1)
    if len(parts) != 2 or not parts[0].isdigit() or not parts[1] or surface not in ('Hard', 'Clay', 'Grass', 'Carpet'):
        raise ValueError('Invalid tournament identity or surface')
    normalized = ''.join(c for c in unicodedata.normalize('NFKD', name.casefold()) if not unicodedata.combining(c))
    normalized = re.sub(r'[^a-z0-9]', '', normalized)
    if not normalized or normalized == 'nan':
        raise ValueError('Missing tournament name')
    return parts[1], normalized, surface


@dataclass
class History:
    width: int
    days: list = field(default_factory=list)
    cumulative: list = field(default_factory=list)

    def append(self, day, vector):
        if self.days and day < self.days[-1]:
            raise ValueError('History must be chronological')
        if not self.cumulative:
            self.cumulative.append(np.zeros(self.width))
        self.days.append(day)
        self.cumulative.append(self.cumulative[-1] + np.asarray(vector, dtype=float))

    def between(self, first, last):
        if not self.days:
            return np.zeros(self.width)
        low, high = bisect_left(self.days, first), bisect_right(self.days, last)
        return self.cumulative[high] - self.cumulative[low]


def expected_rates(server, returner, tour):
    """Server/returner logit adjustment; rows contain two (wins, N) pairs."""
    def rate(v):
        v = np.asarray(v, dtype=float).reshape(2, 2)
        return (v[:, 0] + PLAYER_PRIOR*PRIORS)/(v[:, 1] + PLAYER_PRIOR)
    return expit(logit(rate(server)) + logit(rate(returner)) - logit(rate(tour)))


def count_pair(raw):
    serve = _counts(raw)
    if serve is None:
        return None
    aces = []
    for prefix, (wins, _) in zip(('w', 'l'), serve):
        try:
            value = float(raw[f'{prefix}_ace'])
        except (KeyError, TypeError, ValueError):
            value = np.nan
        aces.append(value)
    ace_valid = all(np.isfinite(a) and a >= 0 and a.is_integer() and a <= s[0]
                    for a, s in zip(aces, serve))
    return [np.array([wins, total, aces[i] if ace_valid else 0., total if ace_valid else 0.])
            for i, (wins, total) in enumerate(serve)]


def _weighted(history, season, cutoff):
    return sum((weight*history[year].between(date.min, cutoff)
                for weight, year in zip((3., 2., 1.), range(season-1, season-4, -1))),
               np.zeros(8))


def context_values(venue, surface, tour):
    """Each two-metric vector holds observed, expected, opportunities, matches."""
    v, s, t = [np.asarray(values).reshape(2, 4) for values in (venue, surface, tour)]
    def raw(m):
        return logit((m[:, 0] + ENVIRONMENT_PRIOR*PRIORS)/(m[:, 2] + ENVIRONMENT_PRIOR))
    def adjusted(m):
        expectation = logit((m[:, 1] + ENVIRONMENT_PRIOR*PRIORS)/(m[:, 2] + ENVIRONMENT_PRIOR))
        return raw(m) - expectation
    available = v[:, 2] > 0
    return {'surface': raw(s)-raw(t),
            'raw': np.where(available, raw(v)-raw(s), 0.),
            'adjusted': np.where(available, adjusted(v)-adjusted(s), 0.),
            'exposure': np.log1p(v[:, 2]), 'missing': (~available).astype(float)}


def build_venue_features(targets, inventory, features, *, cache_dir=DEFAULT_CACHE, raw_rows=None):
    """Return 50 finite fields, same target index; attrs contain groups/audit."""
    if (targets.empty or targets.match_id.duplicated().any() or not targets.index.equals(features.index)
            or features.columns.duplicated().any() or not set(STYLES+STRENGTH).issubset(features)):
        raise ValueError('Unique aligned targets and historical style/strength inputs required')
    seasons = pd.to_numeric(targets.season, errors='coerce')
    if not seasons.isin(range(2009, 2024)).all():
        raise ValueError('Final holdout locked: venue targets must be 2009..2023')
    if not np.isfinite(features[list(STYLES+STRENGTH)].to_numpy(dtype=float)).all():
        raise ValueError('Finite historical styles required')
    allowed = inventory[inventory.source.eq('tour') & pd.to_numeric(inventory.season, errors='coerce').isin(range(2006, 2024))]
    if allowed.match_id.duplicated().any():
        raise ValueError('Duplicate inventory identity')
    lookup = allowed.set_index('match_id').to_dict('index')
    queries = []
    for row in targets.to_dict('records'):
        day, season = _date(row['tourney_date']), int(row['season'])
        _season_date(day, season)
        a, b = _player(row['player_a_id']), _player(row['player_b_id'])
        key = str(row['match_id'])
        key = key if key.startswith('tour:') else 'tour:'+key
        if key not in lookup or a == b:
            raise ValueError('Target absent from eligible inventory or repeated player')
        metadata = lookup[key]
        if (int(metadata['season']) != season or _date(metadata['tourney_date']) != day
                or metadata['surface'] != row['surface'] or metadata['tourney_id'] != row['tourney_id']
                or {a, b} != {str(metadata['player_a_id']), str(metadata['player_b_id'])}):
            raise ValueError('Target identity differs from eligible inventory')
        queries.append((day, season, venue_key(row['tourney_id'], metadata['tourney_name'], row['surface'])))
    max_year = int(seasons.max())-1
    expected = {key: row for key, row in lookup.items() if 2006 <= int(row['season']) <= max_year}
    entries = [e for e in development_entries() if int(e['season']) <= max_year] if raw_rows is None else []
    source = _annual_rows(cache_dir, entries) if raw_rows is None else raw_rows
    records, seen = [], set()
    valid_serve = valid_ace = 0
    for season, raw in source:
        if not 2006 <= int(season) <= max_year:
            continue  # Filter before any future fixture/source field is touched.
        key = f"tour:{raw['tourney_id']}:{int(raw['match_num'])}"
        if key not in expected:
            continue
        if key in seen:
            raise ValueError('Duplicate raw history identity')
        seen.add(key)
        raw, row = corrected_tour_row(raw), expected[key]
        day = _date(row['tourney_date'])
        _season_date(day, int(season))
        winner, loser = _player(raw['winner_id']), _player(raw['loser_id'])
        if (winner == loser or {winner, loser} != {str(row['player_a_id']), str(row['player_b_id'])}
                or _date(raw['tourney_date']) != day or raw['surface'] != row['surface']
                or int(row['season']) != int(season)):
            raise ValueError('Raw statistics disagree with eligible inventory identity')
        counts = count_pair(raw)
        if counts is None:
            continue
        valid_serve += 1
        valid_ace += int(counts[0][3] > 0)
        venue = venue_key(row['tourney_id'], row['tourney_name'], row['surface'])
        records.append((day, key, int(season), venue, winner, loser, counts))
    if set(expected)-seen:
        raise ValueError('Pinned raw source omitted eligible inventory matches')
    records.sort(key=lambda r: (r[0], r[1]))
    server, allowed_rates = defaultdict(lambda: History(4)), defaultdict(lambda: History(4))
    global_rates = History(4)
    for day, _, _, _, winner, loser, pair in records:
        for player, opponent, own, other in ((winner, loser, pair[0], pair[1]), (loser, winner, pair[1], pair[0])):
            server[player].append(day, own)
            allowed_rates[player].append(day, other)
            global_rates.append(day, own)
    venue_hist = defaultdict(lambda: defaultdict(lambda: History(8)))
    surface_hist = defaultdict(lambda: defaultdict(lambda: History(8)))
    tour_hist = defaultdict(lambda: History(8))
    for day, _, season, venue, winner, loser, pair in records:
        first, last = day-timedelta(days=364), day-timedelta(days=LAG)
        tour = global_rates.between(first, last)
        vector = np.zeros((2, 4))
        for player, opponent, own in ((winner, loser, pair[0]), (loser, winner, pair[1])):
            rate = expected_rates(server[player].between(first, last), allowed_rates[opponent].between(first, last), tour)
            counts = own.reshape(2, 2)
            vector[:, 0] += counts[:, 0]
            vector[:, 1] += rate*counts[:, 1]
            vector[:, 2] += counts[:, 1]
        vector[:, 3] = vector[:, 2] > 0
        for history in (venue_hist[venue][season], surface_hist[venue[2]][season], tour_hist[season]):
            history.append(day, vector.ravel())
    results, groups = [], dict(context=[], surface=[], raw=[], adjusted=[], strength=[])
    style_values = features[list(STYLES)].to_numpy(dtype=float)
    strength_values = features[list(STRENGTH)].to_numpy(dtype=float)
    for i, (day, season, venue) in enumerate(queries):
        cutoff = day-timedelta(days=LAG)
        context = context_values(_weighted(venue_hist[venue], season, cutoff),
            _weighted(surface_hist[venue[2]], season, cutoff), _weighted(tour_hist, season, cutoff))
        output = {}
        for group in ('surface', 'raw', 'adjusted', 'exposure', 'missing'):
            for metric, value in zip(('serve', 'ace'), context[group]):
                output[f'venue_{group}_{metric}'] = float(value)
        for group in ('surface', 'raw', 'adjusted'):
            for metric, value in zip(('serve', 'ace'), context[group]):
                for j, style in enumerate(STYLES):
                    output[f'venue_{group}_{metric}_{style}'] = float(value*style_values[i, j])
        for metric, value in zip(('serve', 'ace'), context['adjusted']):
            for j, strength in enumerate(STRENGTH):
                output[f'venue_strength_{metric}_{strength}'] = float(value*strength_values[i, j])
        results.append(output)
    frame = pd.DataFrame(results, index=targets.index, dtype=float)
    for name in frame:
        group = name.split('_')[1] if name.endswith('_diff') else 'context'
        groups[group].append(name)
    if frame.shape[1] != 50 or not np.isfinite(frame.to_numpy()).all():
        raise ValueError('Expected 50 finite tournament-environment fields')
    frame.attrs.update(feature_version=VERSION, groups=groups, prior_season_weights=[3, 2, 1],
        lag_days=LAG, player_prior_points=PLAYER_PRIOR, environment_prior_points=ENVIRONMENT_PRIOR,
        historical_matches=len(expected), valid_serve_matches=valid_serve, valid_ace_matches=valid_ace,
        source_seasons=sorted({int(e['season']) for e in entries}), source_files=entries,
        injected_raw_rows=raw_rows is not None,
        references=['https://www.tennisabstract.com/blog/2013/11/19/the-speed-of-every-2013-surface/',
                    'https://tennisabstract.com/reports/surfaceSpeedATP.html'])
    return frame
