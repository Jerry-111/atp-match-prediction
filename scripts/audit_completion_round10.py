"""Six prespecified event transitions: metadata-only completion feasibility.

No model, outcome, score, market quote or final-year raw file is loaded. Dates
are manually adjudicated primary-source evidence, not inferred from targets.
This script checks their conservative time bounds and participant coverage;
it does not turn a current webpage into an archived publication timestamp.
"""
from __future__ import annotations

import json

import pandas as pd

from scripts.train_models import ROOT, sha256

TARGET_SHA256 = '9bcb3ad4b0fbe745b23b776a750256f92ab9acc9fd1cb57aed0a16cef86435d2'
COLUMNS = ['match_id', 'season', 'tourney_id', 'tourney_date', 'player_a_id', 'player_b_id']

# The six edition pairs were specified before target-label or prediction use.
# The descriptions below are paraphrases, not copied articles or draw tables.
CASES = [
    dict(source='2021-416', target='2021-520', source_name='Rome', target_name='Roland-Garros',
         completion_day='2021-05-16', target_main_start='2021-05-30',
         completion_sources=[dict(
             url='https://novakdjokovic.com/en/news/tennis/novak-with-runner-up-finish-in-rome/',
             evidence='Participant official report dated 16 May 2021 describes the completed final.',
             inspection='full_text')],
         target_sources=[dict(
             url='https://www.rolandgarros.com/en-us/article/new-dates-tournament-2021-postponed-by-one-week',
             evidence='Official revised schedule distinguishes qualifying from main draw beginning 30 May.',
             inspection='full_text')]),
    dict(source='2021-500', target='2021-540', source_name='Halle', target_name='Wimbledon',
         completion_day='2021-06-20', target_main_start='2021-06-28',
         completion_sources=[dict(
             url='https://www.fft.fr/actualites/humbert-un-bleu-gagnant-sur-le-vert',
             evidence='French federation report dated 20 June 2021 describes Humbert winning the final.',
             inspection='full_text')],
         target_sources=[dict(
             url='https://www.itftennis.com/en/tournament/wimbledon/gbr/2021/m-sl-gbr-01a-2021/draws-and-results/',
             evidence='ITF edition dates begin 28 June 2021.', inspection='indexed_primary_metadata'),
             dict(url='https://www.wimbledon.com/en_GB/scores/2021/schedule/pdf/schedulePDF8.pdf',
                  evidence='Official schedule labels Monday 28 June as Day 1.',
                  inspection='indexed_primary_pdf_text')]),
    dict(source='2021-540', target='2021-316', source_name='Wimbledon', target_name='Bastad',
         completion_day='2021-07-11', target_main_start='2021-07-12',
         completion_sources=[dict(
             url='https://novakdjokovic.com/en/news/tennis/novak-wins-historic-title-at-wimbledon-claims-20th-grand-slam-trophy/',
             evidence='Participant official report dated 11 July 2021 describes the completed final.',
             inspection='full_text')],
         target_sources=[dict(url='https://www.tennistv.com/tournaments/316_2021/bastad',
             evidence='Official ATP streaming edition lists Monday 12 July as opening date.',
             inspection='indexed_primary_metadata')]),
    dict(source='2022-416', target='2022-520', source_name='Rome', target_name='Roland-Garros',
         completion_day='2022-05-15', target_main_start='2022-05-22',
         completion_sources=[dict(
             url='https://novakdjokovic.com/en/news/tennis/nole-overpowers-tsitsipas-to-win-sixth-rome-title/',
             evidence='Participant official report dated 15 May 2022 describes the completed final.',
             inspection='full_text')],
         target_sources=[dict(
             url='https://www.rolandgarros.com/en-us/article/roland-garros-2022-opening-official-ticketing',
             evidence='Official advance ticket notice separates qualifying 16-20 May from main draw on 22 May.',
             inspection='full_text')]),
    dict(source='2022-500', target='2022-540', source_name='Halle', target_name='Wimbledon',
         completion_day='2022-06-19', target_main_start='2022-06-27',
         completion_sources=[dict(
             url='https://www.youtube.com/watch?v=jULQZp74uQY',
             evidence='Verified official Tennis TV final highlights have displayed publication date 19 June 2022.',
             inspection='indexed_primary_publication_metadata'),
             dict(url='https://www.terrawortmann-open.de/fileadmin/Gerry_Weber_Open/Ergebnisse/Rueckblick/Hauptfeld_Einzel.pdf',
                  evidence='Official final draw visually confirms Hurkacz winner and Medvedev finalist; planned date range alone is not completion proof.',
                  inspection='downloaded_and_visually_inspected_page_1',
                  sha256='fd6e9190af0dbae5fa5ff45dbe8c1e3c29afe057414ada26873da6de0aa735d9')],
         target_sources=[dict(
             url='https://www.wimbledon.com/pdf/CH_2022_Easy_Access_Guide_v8_DIGI.pdf',
             evidence='Official access guide dates the Championships from Monday 27 June 2022.',
             inspection='indexed_primary_pdf_text'),
             dict(url='https://www.wimbledon.com/en_GB/scores/2022/schedule/pdf/schedulePDF8.pdf',
                  evidence='Official schedule labels Monday 27 June as Day 1; later update is not an as-of publication snapshot.',
                  inspection='indexed_primary_pdf_text')]),
    dict(source='2022-540', target='2022-316', source_name='Wimbledon', target_name='Bastad',
         completion_day='2022-07-10', target_main_start='2022-07-11',
         completion_sources=[dict(
             url='https://novakdjokovic.com/en/news/tennis/impressive-novak-wins-fourth-consecutive-wimbledon-title/',
             evidence='Participant official report dated 10 July 2022 describes the completed final.',
             inspection='full_text')],
         target_sources=[dict(url='https://www.tennistv.com/tournaments/316_2022/bastad',
             evidence='Official ATP streaming edition lists Monday 11 July as opening date.',
             inspection='full_text')]),
]


def date_bounds(completion_day, target_start):
    """End of date at UTC-12 versus beginning of date at UTC+14.

    These deliberately loose bounds do not assume venue timezones or actual
    match hours. Equality is rejected. Adjacent dates therefore cannot qualify.
    """
    finish = pd.Timestamp(completion_day, tz='UTC') + pd.Timedelta(hours=36)
    start = pd.Timestamp(target_start, tz='UTC') - pd.Timedelta(hours=14)
    return finish, start, finish < start


def audit_case(case, rows):
    source = rows.loc[rows.tourney_id.eq(case['source'])]
    target = rows.loc[rows.tourney_id.eq(case['target'])]
    assert len(source) and len(target)
    assert source.season.eq(int(case['source'][:4])).all()
    assert target.season.eq(int(case['target'][:4])).all()
    assert source.tourney_date.nunique() == target.tourney_date.nunique() == 1
    source_date = pd.Timestamp(source.tourney_date.iloc[0], tz='UTC')
    nominal_target_date = pd.Timestamp(target.tourney_date.iloc[0], tz='UTC')
    finish, start, qualified = date_bounds(case['completion_day'], case['target_main_start'])
    assert source_date <= pd.Timestamp(case['completion_day'], tz='UTC')
    # A corrected Sunday opening may precede the repository's Monday date.
    assert abs((nominal_target_date - pd.Timestamp(case['target_main_start'], tz='UTC')).days) <= 1
    stale_eligible = source_date <= nominal_target_date - pd.Timedelta(days=21)
    participants = set(source.player_a_id) | set(source.player_b_id)
    a = target.player_a_id.isin(participants)
    b = target.player_b_id.isin(participants)
    newly_available = qualified and not stale_eligible
    return dict(**case, source_pool_matches=len(source), target_matches=len(target),
        source_nominal_date=source_date.date().isoformat(),
        target_nominal_date=nominal_target_date.date().isoformat(),
        completion_utc_upper=finish.isoformat(), target_start_utc_lower=start.isoformat(),
        conservative_gap_hours=(start-finish).total_seconds()/3600,
        date_bound_qualified=bool(qualified),
        rejected_reason=None if qualified else 'Date-only UTC bounds overlap; no actual-time certificate.',
        already_available_under_T_minus_21=bool(stale_eligible),
        newly_available_under_evidence_assumptions=bool(newly_available),
        direct_intersection_either_player=int((a|b).sum()),
        direct_intersection_both_players=int((a&b).sum()),
        newly_covered_either_player=int((a|b).sum()) if newly_available else 0,
        newly_covered_both_players=int((a&b).sum()) if newly_available else 0,
        source_match_ids=source.match_id.tolist(), target_match_ids=target.match_id.tolist(),
        newly_covered_target_ids=target.loc[(a|b) & newly_available,'match_id'].tolist())


def main():
    path = ROOT/'results/round3/development_targets.csv'
    assert sha256(path) == TARGET_SHA256
    rows = pd.read_csv(path, usecols=COLUMNS)
    assert not rows.match_id.duplicated().any() and len(rows) == 36421
    # Metadata projection contains no a_won/outcome; reject final years anyway.
    assert rows.season.between(2009,2023).all()
    rows = rows[rows.season.isin([2021,2022])].copy()
    cases = [audit_case(case, rows) for case in CASES]
    assert len(set(c['target'] for c in cases)) == len(cases)
    output = dict(status='source_feasibility_only_with_publication_provenance_limits',
        reviewed_on='2026-11-28', prespecified_transition_count=6,
        date_bound_qualified=sum(c['date_bound_qualified'] for c in cases),
        distinct_new_source_editions=sum(c['newly_available_under_evidence_assumptions'] for c in cases),
        target_matches_in_pilot=sum(c['target_matches'] for c in cases),
        new_direct_participant_coverage=sum(c['newly_covered_either_player'] for c in cases),
        cases=cases,
        assumptions=[
            'Displayed dated primary reports and publication metadata correctly describe completed public finals; historic page revisions are not independently archived.',
            'Earlier knockout outcomes were public by the completed final. This does not timestamp later-corrected technical statistics.',
            'Earliest main-draw date precedes every target match. Qualifying dates are not substituted for main draw.',
            'Global date bounds cover UTC offsets from -12 to +14; actual venue zones/hours are not inferred.'],
        limitations=[
            'Mixed evidence strength is recorded per source; indexed metadata is not full-page or audiovisual inspection.',
            'Six purposive transitions do not estimate full-season coverage or model gains.',
            'Direct participant intersections are not the total influence of opponent-adjusted ratings.',
            'No predictive fit, changed lag, final-test access or new 2023 score is authorized by this file.',
            'The ATP 2023 season yearbook was located by search but its PDF returned 403; it is not an inspected completion source.',
            'The source PDF remains a local temporary cache and is not redistributed.'],
        model_fitting_authorized=False,
        hashes=dict(targets=sha256(path), code=sha256(ROOT/'scripts/audit_completion_round10.py')),
        input_columns=COLUMNS)
    out = ROOT/'results/round10/completion'
    out.mkdir(parents=True,exist_ok=True)
    (out/'source_feasibility.json').write_text(json.dumps(output,indent=2)+'\n')
    print(json.dumps({k:v for k,v in output.items() if k in
        ['status','prespecified_transition_count','date_bound_qualified',
         'distinct_new_source_editions','target_matches_in_pilot','new_direct_participant_coverage']},indent=2))
    for c in cases:
        print(c['source'],'->',c['target'],c['date_bound_qualified'],
              c['target_matches'],c['newly_covered_either_player'])


if __name__ == '__main__':
    main()
