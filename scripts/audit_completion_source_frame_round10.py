"""Offline independent metadata checks and an unadjudicated source worklist.

This does not import the original completion-audit implementation, retrieve
webpages, inspect outcomes, change information cutoffs, or fit a model.
"""
from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from hashlib import sha256
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/round10/completion'
TARGET_SHA256 = '9bcb3ad4b0fbe745b23b776a750256f92ab9acc9fd1cb57aed0a16cef86435d2'
FIELDS = ['match_id', 'season', 'tourney_id', 'tourney_date',
          'player_a_id', 'player_b_id']
# Independently reread/adjudicated dates; Halle 2022 retains the original
# reviewer's indexed-publication evidence, which this reviewer could not reopen.
CASES = [
    ('2021-416', '2021-520', '2021-05-16', '2021-05-30'),
    ('2021-500', '2021-540', '2021-06-20', '2021-06-28'),
    ('2021-540', '2021-316', '2021-07-11', '2021-07-12'),
    ('2022-416', '2022-520', '2022-05-15', '2022-05-22'),
    ('2022-500', '2022-540', '2022-06-19', '2022-06-27'),
    ('2022-540', '2022-316', '2022-07-10', '2022-07-11'),
]


def digest(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def check_cases(rows, ledger):
    recorded = {(v['source'], v['target']): v for v in ledger['cases']}
    assert set(recorded) == {(a, b) for a, b, _, _ in CASES}
    checked, all_targets, all_new = [], set(), set()
    for source_id, target_id, completion_day, target_start in CASES:
        old = recorded[source_id, target_id]
        assert (old['completion_day'], old['target_main_start']) == (completion_day, target_start)
        source = rows.loc[rows.tourney_id.eq(source_id)]
        target = rows.loc[rows.tourney_id.eq(target_id)]
        assert len(source) and len(target)
        assert source.tourney_date.nunique() == target.tourney_date.nunique() == 1
        source_day = date.fromisoformat(source.tourney_date.iloc[0])
        target_day = date.fromisoformat(target.tourney_date.iloc[0])
        completion = date.fromisoformat(completion_day)
        opening = date.fromisoformat(target_start)
        assert source_day <= completion and abs((target_day - opening).days) <= 1
        # Independent scalar arithmetic: UTC date-start separation minus 50 h.
        gap = 24 * (opening - completion).days - 50
        upper = datetime.combine(completion, datetime.min.time(), timezone.utc) + timedelta(hours=36)
        lower = datetime.combine(opening, datetime.min.time(), timezone.utc) - timedelta(hours=14)
        assert (lower - upper).total_seconds() / 3600 == gap
        baseline_eligible = (target_day - source_day).days >= 21
        conditional_new = gap > 0 and not baseline_eligible
        participants = set(source.player_a_id) | set(source.player_b_id)
        a, b = target.player_a_id.isin(participants), target.player_b_id.isin(participants)
        source_ids, target_ids = source.match_id.tolist(), target.match_id.tolist()
        either_ids, both_ids = target.loc[a | b, 'match_id'].tolist(), target.loc[a & b, 'match_id'].tolist()
        new_ids = either_ids if conditional_new else []
        values = dict(source_pool_matches=len(source), target_matches=len(target),
            source_nominal_date=source_day.isoformat(), target_nominal_date=target_day.isoformat(),
            completion_utc_upper=upper.isoformat(), target_start_utc_lower=lower.isoformat(),
            conservative_gap_hours=gap, date_bound_qualified=gap > 0,
            already_available_under_T_minus_21=baseline_eligible,
            newly_available_under_evidence_assumptions=conditional_new,
            direct_intersection_either_player=len(either_ids),
            direct_intersection_both_players=len(both_ids),
            newly_covered_either_player=len(new_ids),
            newly_covered_both_players=len(both_ids) if conditional_new else 0,
            source_match_ids=source_ids, target_match_ids=target_ids,
            newly_covered_target_ids=new_ids)
        assert all(old[k] == v for k, v in values.items()), (source_id, target_id)
        assert not all_targets.intersection(target_ids)
        all_targets.update(target_ids)
        all_new.update(new_ids)
        checked.append(dict(source=source_id, target=target_id,
            completion_day=completion_day, target_main_start=target_start,
            independent_arithmetic_and_ids_match=True,
            completion_day_reopened_by_this_reviewer=source_id != '2022-500',
            direct_either_target_ids=either_ids, direct_both_target_ids=both_ids,
            admitted_for_prediction=False, **values))
    assert (len(all_targets), len(all_new)) == (544, 204)
    return checked


def build_frame(rows, years, checked):
    events = []
    for (season, event), group in rows.groupby(['season', 'tourney_id'], sort=True):
        assert group.tourney_date.nunique() == 1
        events.append(dict(season=int(season), tourney_id=event,
            nominal_date=date.fromisoformat(group.tourney_date.iloc[0]),
            match_ids=group.match_id.tolist()))
    prior = {(v['source'], v['target']): v for v in checked}
    pairs, targets = [], []
    for target in events:
        if target['season'] not in years:
            continue
        cutoff = target['nominal_date'] - timedelta(days=21)
        sources = [s for s in events if cutoff < s['nominal_date'] <= target['nominal_date']
                   and s['tourney_id'] != target['tourney_id']]
        targets.append(dict(season=target['season'], target_tourney_id=target['tourney_id'],
            target_nominal_date=target['nominal_date'].isoformat(),
            original_T_minus_21_date=cutoff.isoformat(), target_matches=len(target['match_ids']),
            candidate_source_count=len(sources),
            target_match_ids=json.dumps(target['match_ids'], separators=(',', ':')),
            availability_status='worklist_only_not_prediction_authorized'))
        for source in sources:
            case = prior.get((source['tourney_id'], target['tourney_id']))
            previous = ('date_bounds_pass_under_publication_assumptions' if case['date_bound_qualified']
                        else 'date_bounds_overlap_rejected') if case else 'not_reviewed'
            pairs.append(dict(season=target['season'], source_season=source['season'],
                source_tourney_id=source['tourney_id'], target_tourney_id=target['tourney_id'],
                source_nominal_date=source['nominal_date'].isoformat(),
                target_nominal_date=target['nominal_date'].isoformat(),
                original_T_minus_21_date=cutoff.isoformat(),
                nominal_gap_days=(target['nominal_date'] - source['nominal_date']).days,
                same_nominal_day=source['nominal_date'] == target['nominal_date'],
                source_matches=len(source['match_ids']), target_matches=len(target['match_ids']),
                source_match_ids=json.dumps(source['match_ids'], separators=(',', ':')),
                target_match_ids=json.dumps(target['match_ids'], separators=(',', ':')),
                prior_six_case_review=case is not None, prior_case_date_bound_status=previous,
                availability_status='candidate_only_no_new_adjudication', admitted_for_prediction=False))
    frame = pd.DataFrame(pairs).sort_values(['season', 'target_nominal_date', 'target_tourney_id', 'source_tourney_id'])
    targets = pd.DataFrame(targets).sort_values(['season', 'target_nominal_date', 'target_tourney_id'])
    assert not frame.duplicated(['source_tourney_id', 'target_tourney_id']).any()
    assert frame.nominal_gap_days.between(0, 20).all()
    assert not frame.admitted_for_prediction.any()
    return frame, targets


def main():
    target_path = ROOT / 'results/round3/development_targets.csv'
    ledger_path = OUT / 'source_feasibility.json'
    assert digest(target_path) == TARGET_SHA256
    ledger = json.loads(ledger_path.read_text())
    assert ledger['hashes']['targets'] == TARGET_SHA256
    assert ledger['hashes']['code'] == digest(ROOT / 'scripts/audit_completion_round10.py')
    rows = pd.read_csv(target_path, usecols=FIELDS)
    assert len(rows) == 36421 and rows.match_id.is_unique
    assert rows.season.between(2009, 2023).all() and not rows[FIELDS].isna().any().any()
    assert rows.tourney_id.str[:4].astype(int).eq(rows.season).all()
    checked = check_cases(rows, ledger)
    original, targets = build_frame(rows, {2021, 2022}, checked)
    extension, extension_targets = build_frame(rows, {2023}, checked)
    assert (len(targets), len(original), len(extension_targets), len(extension)) == (127, 504, 64, 259)
    assert targets.loc[targets.candidate_source_count.eq(0), 'target_tourney_id'].tolist() == ['2021-499']
    assert original.prior_six_case_review.sum() == 5 and not extension.prior_six_case_review.any()
    outputs = {'source_frame.csv': original, 'target_editions.csv': targets,
               'source_frame_2023_unqualified.csv': extension,
               'target_editions_2023_unqualified.csv': extension_targets}
    for filename, frame in outputs.items():
        frame.to_csv(OUT / filename, index=False, lineterminator='\n')
    summary = []
    for year in (2021, 2022, 2023):
        pairs = original.loc[original.season.eq(year)] if year < 2023 else extension
        target_subset = targets.loc[targets.season.eq(year)] if year < 2023 else extension_targets
        summary.append(dict(season=year, target_editions=len(target_subset),
            target_matches=int(target_subset.target_matches.sum()), candidate_pairs=len(pairs),
            distinct_candidate_source_editions=int(pairs.source_tourney_id.nunique()),
            target_editions_with_candidate_pairs=int(pairs.target_tourney_id.nunique()),
            target_editions_without_candidate_pairs=target_subset.loc[
                target_subset.candidate_source_count.eq(0), 'target_tourney_id'].tolist(),
            reviewed_pilot_pairs=int(pairs.prior_six_case_review.sum()),
            newly_adjudicated_pairs=0, admitted_for_prediction=0))
    review = dict(status='independent_metadata_pass_with_source_provenance_limits',
        reviewed_on='2026-11-28', network_retrievals_by_reproduction_script=0,
        models_fitted=0, history_cutoffs_changed=False, final_year_raw_accessed=False,
        input_columns=FIELDS, note='Hashing verifies the full file; only six metadata columns are parsed.',
        independent_date_bound_qualified=sum(c['date_bound_qualified'] for c in checked),
        independent_new_source_editions=sum(c['newly_available_under_evidence_assumptions'] for c in checked),
        independent_pilot_target_matches=544, independent_new_direct_target_matches=204,
        cases=checked, frame_by_season=summary,
        frame_rule='source_nominal > target_nominal - 21 days AND source_nominal <= target_nominal; exclude the target edition itself',
        frame_limits=[
            'Overinclusive nominal-date sampling frame; same-day and overlapping events remain candidates, never automatically available.',
            'Only the original six purposive cases have completion-source review; the broader frame is a next-round worklist.',
            'Rome 2021 to Roland-Garros is outside the new-information frame: nominal dates 2021-05-08 and 2021-05-31 are 23 days apart, already qualifying under T-minus-21.',
            '2021-499 has no candidate source in the defined window and remains in target_editions.csv.',
            'Source match IDs refer to the original completed/ranked Tour population, not every match in the official draw.',
            'Nominal dates are not certified first-match times; source/target opening corrections must be audited before later admission.',
            '2023 is a separate unqualified same-rule extension with zero reviewed pairs; no prediction or fit is authorized.'],
        publication_review=dict(
            five_dated_completion_reports_reopened=True,
            independent_historic_first_publication_snapshots_obtained=False,
            halle_2022=dict(url='https://www.youtube.com/watch?v=jULQZp74uQY',
                independent_reopen_status='fetch_failed_and_exact_video_id_search_returned_no_results',
                interpretation='Original indexed-primary publication date retained as conditional evidence, not independently reverified.',
                official_draw_limit='Draw corroborates finalists/champion, not completion or first-publication day by itself.'),
            target_date_sources='RG 2021/2022 notices, Bastad 2021/2022 edition pages, Wimbledon 2021/2022 schedules and 2022 guide were reopened.',
            schedule_limit='Current Day 1 PDFs contain results; the 2022 PDF carries a 5 July update. They corroborate the opening date, not an as-of publication snapshot.',
            source_access_limit='No further network search is performed by this artifact task; the blocked ATP yearbook was not accessed.'),
        next_round_prerequisites=[
            'Freeze the whole source-edition worklist and evidence criteria before predicting; retain failed/missing sources.',
            'Record completion and first-publication evidence separately, with source URL, retrieved-content hash and retrieval time.',
            'When both bounds are established, require max(completion_upper, publication_upper) < target_start_lower; otherwise disclose the conditional evidence tier.',
            'Initially admit result identities/outcomes only; technical-statistic publication and corrections remain unverified.',
            'Keep all target rows, exact T-minus-21 fallback and matched stale/fresh rating/update rules; no error-driven source selection.'],
        hashes=dict(targets=digest(target_path), original_ledger=digest(ledger_path),
            original_audit_code=digest(ROOT / 'scripts/audit_completion_round10.py'),
            independent_audit_code=digest(Path(__file__)),
            review_document=digest(ROOT / 'docs/research/round10_completion_review.md'),
            output_csvs={filename: digest(OUT / filename) for filename in outputs}))
    (OUT / 'independent_review.json').write_text(json.dumps(review, indent=2) + '\n')
    print(json.dumps({'status': review['status'], 'frame_by_season': summary}, indent=2))


if __name__ == '__main__':
    main()
