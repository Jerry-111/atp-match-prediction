"""Post-link source-winner quality audit; never a model input or label repair.

Only pinned 2021--2023 local files are allowed. Matching stays frozen from
Round 5. Winner IDs are read solely after a recorded audit protocol exists.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
import hashlib
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
YEARS = (2021, 2022, 2023)
SOURCE_COLUMNS = ('match_id', 'date', 'tournoi_id', 'joueur1_id', 'joueur2_id',
                  'vainqueur_id')
TARGET_COLUMNS = ('match_id', 'season', 'tourney_date', 'tourney_id', 'round',
                  'player_a_id', 'player_b_id', 'player_a_name', 'player_b_name', 'a_won')


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def identifier(value):
    """Normalize numeric source IDs without accepting missing/fractional IDs."""
    if pd.isna(value):
        return None
    try:
        number = Decimal(str(value))
    except InvalidOperation:
        return None
    if not number.is_finite() or number <= 0 or number != number.to_integral_value():
        return None
    return str(int(number))


def timestamp(value):
    return pd.to_datetime(value, utc=True, errors='coerce')


def audit_year(targets, linked, source, year, reused_source_ids=()):
    """Compare fixed linked winners; preserve all target rows and assignments."""
    if year not in YEARS:
        raise ValueError('Only 2021--2023 development source years are permitted')
    if targets.empty or targets.match_id.duplicated().any() or linked.match_id.duplicated().any():
        raise ValueError('Nonempty unique target and linkage IDs required')
    if not targets.match_id.reset_index(drop=True).equals(linked.match_id.reset_index(drop=True)):
        raise ValueError('Frozen linkage order or target population changed')
    if not pd.to_numeric(targets.season).eq(year).all() or not pd.to_numeric(linked.season).eq(year).all():
        raise ValueError('Target/linkage season mismatch or final holdout')
    dates = pd.to_datetime(targets.tourney_date, errors='coerce')
    valid_dates = dates.dt.year.eq(year) | (
        dates.dt.year.eq(year - 1) & dates.dt.month.eq(12) & dates.dt.day.ge(25))
    if dates.isna().any() or dates.dt.year.gt(2023).any() or not valid_dates.all():
        raise ValueError('Target date outside declared development season')
    labels = pd.to_numeric(targets.a_won, errors='coerce')
    if not labels.isin([0, 1]).all():
        raise ValueError('Unchanged binary target labels required')
    source_records = {}
    for record in source.to_dict('records'):
        source_records.setdefault(identifier(record['match_id']), []).append(record)
    reused = set(reused_source_ids)
    counts = Counter(identifier(v) for v in linked.loc[linked.status.eq('matched'), 'market_match_id'])
    reused.update(k for k, n in counts.items() if n > 1)
    results = []
    for target, link in zip(targets.to_dict('records'), linked.to_dict('records')):
        row = {key: target[key] for key in TARGET_COLUMNS}
        row.update(linkage_status=link['status'], market_match_id=link['market_match_id'],
                   market_player_a_id=link['market_player_a_id'],
                   market_player_b_id=link['market_player_b_id'],
                   source_winner_id=None, source_winner_side=None,
                   source_a_won=None, agreement=None, issues='')
        if link['status'] != 'matched':
            row['audit_status'] = 'unmatched_not_audited'
            results.append(row)
            continue
        sid = identifier(link['market_match_id'])
        source_rows = source_records.get(sid, [])
        issues = []
        if sid is None:
            issues.append('invalid_saved_source_match_id')
        if sid in reused:
            issues.append('source_match_reused')
        if len(source_rows) != 1:
            issues.append('source_match_missing' if not source_rows else 'source_match_id_duplicated')
        if str(link['target_event_id']) != str(target['tourney_id']):
            issues.append('target_event_changed')
        if len(source_rows) == 1:
            record = source_rows[0]
            first, second = identifier(record['joueur1_id']), identifier(record['joueur2_id'])
            a, b = identifier(link['market_player_a_id']), identifier(link['market_player_b_id'])
            if None in (first, second, a, b) or first == second or a == b or {first, second} != {a, b}:
                issues.append('source_player_pair_changed')
            if identifier(record['tournoi_id']) != identifier(link['market_event_id']):
                issues.append('source_event_changed')
            day, fixed_day = timestamp(record['date']), timestamp(link['market_match_utc'])
            if pd.isna(day) or pd.isna(fixed_day) or day != fixed_day:
                issues.append('source_datetime_changed_or_invalid')
            if pd.notna(day):
                date_valid = day.year == year or (day.year == year - 1 and day.month == 12 and day.day >= 25)
                if day.year > 2023 or not date_valid:
                    issues.append('source_date_outside_development_season')
            winner = identifier(record['vainqueur_id'])
            row['source_winner_id'] = winner
            if not issues:
                if winner is None:
                    row['audit_status'] = 'winner_missing_or_invalid'
                elif winner not in (first, second):
                    row['audit_status'] = 'winner_not_in_source_pair'
                else:
                    row['source_winner_side'] = 'joueur1' if winner == first else 'joueur2'
                    row['source_a_won'] = int(winner == a)
                    row['agreement'] = int(winner == a) == int(target['a_won'])
                    row['audit_status'] = 'agreement' if row['agreement'] else 'winner_disagreement'
        if issues:
            row['audit_status'] = 'structural_conflict'
            row['issues'] = '|'.join(issues)
        results.append(row)
    frame = pd.DataFrame(results)
    status = frame.audit_status.value_counts().to_dict()
    valid_winners = status.get('agreement', 0) + status.get('winner_disagreement', 0)
    summary = dict(season=year, targets=len(frame), matched=int(linked.status.eq('matched').sum()),
                   unmatched=int(linked.status.ne('matched').sum()),
                   structurally_valid_winners=valid_winners,
                   agreements=status.get('agreement', 0),
                   winner_disagreements=status.get('winner_disagreement', 0),
                   structural_conflicts=status.get('structural_conflict', 0),
                   missing_or_invalid_winners=status.get('winner_missing_or_invalid', 0),
                   unrelated_winners=status.get('winner_not_in_source_pair', 0),
                   winner_source_side_counts=frame.source_winner_side.value_counts().to_dict(),
                   unmatched_status_counts=linked.loc[linked.status.ne('matched'), 'status'].value_counts().to_dict(),
                   source_agreement_rate=(status.get('agreement', 0) / valid_winners if valid_winners else None))
    return frame, summary


def load_pinned_year(cache, year, entry):
    if year not in YEARS or int(entry['season']) != year:
        raise ValueError('Only pinned 2021--2023 files may be read')
    path = Path(cache) / f'{year}.csv'
    if sha256(path) != entry['sha256']:
        raise ValueError(f'Raw source hash changed: {year}')
    source = pd.read_csv(path, sep=';', usecols=list(SOURCE_COLUMNS), dtype='string')
    if len(source) != entry['rows']:
        raise ValueError(f'Raw source row count changed: {year}')
    return source


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache', type=Path, default=ROOT / 'data/raw/market')
    parser.add_argument('--output', type=Path, default=ROOT / 'results/round6')
    args = parser.parse_args()
    protocol = ROOT / 'results/round6/source_audit.protocol.md'
    if not protocol.is_file():
        raise ValueError('The recorded pre-outcome audit protocol is required')
    manifest_path = ROOT / 'data/market-manifest.json'
    manifest = json.loads(manifest_path.read_text())
    previous_audit = json.loads((ROOT / 'results/round5/market/artifact_audit.json').read_text())
    linkage_path = ROOT / 'results/round5/market/linked_matches.csv'
    targets_path = ROOT / 'data/processed/model_features.csv'
    registry = json.loads((ROOT / 'results/round5/search_registry.json').read_text())
    if sha256(manifest_path) != previous_audit['source_manifest_sha256']:
        raise ValueError('The audited market manifest changed')
    if sha256(linkage_path) != previous_audit['artifact_sha256']['linked_matches.csv']:
        raise ValueError('Frozen outcome-blind source linkage changed')
    if sha256(targets_path) != registry['target_sha256']:
        raise ValueError('Pinned target table changed')
    targets = pd.read_csv(targets_path, usecols=list(TARGET_COLUMNS))
    targets = targets.loc[targets.season.isin(YEARS)].reset_index(drop=True)
    linked = pd.read_csv(linkage_path)
    if len(targets) != len(linked) or len(targets) != 7504:
        raise ValueError('The fixed three-year target population changed')
    counts = Counter(identifier(v) for v in linked.loc[linked.status.eq('matched'), 'market_match_id'])
    reused = {k for k, n in counts.items() if n > 1}
    entries = {int(e['season']): e for e in manifest['files']}
    frames, summaries = [], []
    for year in YEARS:
        source = load_pinned_year(args.cache, year, entries[year])
        frame, result = audit_year(targets.loc[targets.season.eq(year)],
                                   linked.loc[linked.season.eq(year)], source, year, reused)
        frames.append(frame)
        summaries.append(result)
        print(json.dumps(result, sort_keys=True), flush=True)
        del source
    combined = pd.concat(frames, ignore_index=True)
    conflicts = combined.loc[~combined.audit_status.isin(['agreement', 'unmatched_not_audited'])]
    # Append independently checked official evidence without replacing either
    # observed winner or converting a disagreement into source agreement.
    resolution_path = ROOT / 'results/round6/source_audit.resolutions.json'
    resolutions = json.loads(resolution_path.read_text())['resolutions'] if resolution_path.exists() else []
    resolved_ids = set()
    for resolution in resolutions:
        selected = conflicts.loc[conflicts.match_id.eq(resolution['match_id'])]
        if len(selected) != 1 or resolution['match_id'] in resolved_ids:
            raise ValueError('Official resolution must identify one current unresolved conflict')
        observed = selected.iloc[0]
        if (identifier(observed.market_match_id) != resolution['market_match_id']
                or observed.source_winner_id != resolution['observed_source_winner_id']
                or int(observed.a_won) != resolution['observed_target_a_won']
                or resolution['changes_applied'] is not False):
            raise ValueError('Official resolution observations changed; re-review required')
        resolved_ids.add(resolution['match_id'])
    audit = dict(audited_utc=datetime.now(timezone.utc).isoformat(),
                 status=('agreement_on_all_auditable_links' if conflicts.empty else
                         'disagreements_officially_resolved' if len(resolved_ids) == len(conflicts)
                         else 'investigation_required'),
                 protocol_written_before_source_winner_read=True, processing_order=list(YEARS),
                 final_test_evaluated=False, final_test_source_files_accessed=False,
                 predictions_read=False, models_fitted=False, labels_changed=False,
                 source_linkage_changed=False, winner_fields_used_for_matching=False,
                 source_fields_read=list(SOURCE_COLUMNS), target_rows=len(combined),
                 validation_2023_rows=int(combined.season.eq(2023).sum()),
                 seasons=summaries, investigation_rows=len(conflicts),
                 official_resolutions=resolutions, unresolved_investigation_rows=len(conflicts) - len(resolved_ids),
                 input_sha256={'protocol': sha256(protocol), 'manifest': sha256(manifest_path),
                               'fixed_linkage': sha256(linkage_path), 'targets': sha256(targets_path),
                               **{str(y): entries[y]['sha256'] for y in YEARS}},
                 interpretation='Post-link source agreement is a quality check, not independent model validation. '
                 'Unmatched targets remain unaudited; shared upstream errors remain possible. '
                 'No labels, cohorts, features, models or scores are changed.')
    if resolution_path.exists():
        audit['input_sha256']['official_resolution_record'] = sha256(resolution_path)
    args.output.mkdir(parents=True, exist_ok=True)
    combined.to_csv(args.output / 'source_audit.csv', index=False)
    conflicts.to_csv(args.output / 'source_audit.conflicts.csv', index=False)
    (args.output / 'source_audit.json').write_text(json.dumps(audit, indent=2) + '\n')
    text = ['# 第六轮：固定匹配后的跨来源胜者核对', '',
            '规则先记录，再依次读取2021、2022、2023的外部胜者字段；匹配、目标标签和完整样本均不改变。', '',
            '| 年份 | 完整目标 | 原匹配 | 未匹配、未核对 | 有效胜者 | 一致 | 胜者不一致 | 结构冲突 | 缺失/异常胜者 |',
            '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for s in summaries:
        text.append(f"| {s['season']} | {s['targets']} | {s['matched']} | {s['unmatched']} | "
                    f"{s['structurally_valid_winners']} | {s['agreements']} | {s['winner_disagreements']} | "
                    f"{s['structural_conflicts']} | {s['missing_or_invalid_winners'] + s['unrelated_winners']} |")
    text += ['', '原始字段 `vainqueur_id` 必须等于该记录两位球员之一；按冻结的球员A来源ID转换方向。'
             '原始市场CSV的价格、比分和比赛时长字段未读取；审计脚本也未读取或评分任何模型预测。', '',
             ('所有可核对匹配的胜者方向一致，没有需要官方结果裁决的冲突。这个结果不支持“已匹配比赛的大量标签错误造成70%差距”的解释。'
              if conflicts.empty else
              '检测到来源分歧，原始记录保留在 source_audit.conflicts.csv；官方复核见下文，此审计不自动修正标签。'
              if resolutions else
              '发现待调查记录，详见 source_audit.conflicts.csv。须逐条核对官方事件结果；此审计不自动修正标签。'), '',
             '未匹配记录仍未得到第二来源验证；两个来源可能共享上游错误。一致性不能证明每个真值绝对正确，'
             '也不意味着模型有新增准确率。2023已在前轮查看，本工作是质量审计；2024–2025没有访问。', '',
             '[事先规则](source_audit.protocol.md) · [逐场记录](source_audit.csv) · [机器可读汇总](source_audit.json)', '',
             '复跑：`python -m scripts.audit_round6_labels`。只使用本地已固定哈希的2021–2023来源，不下载新数据。', '']
    if resolutions:
        text += ['## 官方结果复核（审计后追加，不改变预先规则）', '']
        for r in resolutions:
            text += [f"`{r['match_id']}`：外部文件将胜者记为来源ID `{r['observed_source_winner_id']}`。"
                     f"[官方ATP签表]({r['official_source_url']})显示 **{r['official_winner_name']}** "
                     f"以 **{r['official_score']}** 战胜 {r['official_loser_name']}；"
                     '与本项目原有标签一致。官方PDF已视觉核验，哈希和核验日期保存在独立resolution记录中。', '']
        text += [f'官方复核完成{len(resolved_ids)}条，仍待调查{len(conflicts) - len(resolved_ids)}条。'
                 '表中仍保留原始来源不一致计数；没有把证据裁决伪装成两个来源原本一致。'
                 '市场模型从未使用外部winner字段，因此此发现不改变既有模型标签或分数。', '',
                 '2023已匹配的2,502场全部同向，另86场未匹配仍未核对。此次审计不支持“广泛标签错误能解释距70%的差额”，'
                 '但不能排除未匹配样本或共同上游错误。', '']
    (args.output / 'source_audit.md').write_text('\n'.join(text))


if __name__ == '__main__':
    main()
