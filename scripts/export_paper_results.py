#!/usr/bin/env python3
"""Export anonymous aggregate tables at the precision displayed in a manuscript."""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
from pathlib import Path
import re

# Explicit schemas avoid exporting task examples, source records or paper prose.
TABLES = {
 'tab:gaia-dataset-stats': ('main/dataset_profile', 'layer,statistics'),
 'tab:native-augmented-eval': ('main/three_facet_headline', 'model,parameters,node_f1,edge_f1_native,edge_f1_augmented,edge_f1_delta,tool_f1,param_f1,param_value_f1,normalized_em,token_f1,accepted_rate'),
 'tab:multi-order-score-lift': ('main/multi_order_score_lift', 'subset,metric,native,augmented,delta,relative_delta_percent,delta_ci95,models_with_positive_delta'),
 'tab:reference-budget-sensitivity': ('main/reference_budget', 'non_native_budget,edge_f1,relative_delta,delta_ci95,gain_pairs_percent'),
 'tab:planning-answer-toolclean': ('main/planning_answer_toolclean', 'model,tasks,pearson_r,ci95'),
 'tab:web-source-reference': ('main/web_source', 'slice,tasks,edge_f1_native,edge_f1_augmented,edge_f1_delta,delta_ci95'),
 'tab:best-match': ('appendix/reference_robustness/best_match_distribution', 'scope,pairs,gain_pairs,tie_pairs,native_zero_percent,augmented_zero_percent'),
 'tab:alignment-threshold-current': ('appendix/reference_robustness/alignment_threshold_s1', 'threshold,all_edge_delta,original_rich_edge_delta,all_rank_spearman,original_rich_rank_spearman'),
 'tab:tool-threshold': ('appendix/planning_answer/tool_clean_thresholds', 'ats,tisr,pairs,correct,mean_r_native,mean_r_augmented'),
 'tab:appendix-cross-benchmark-taskbench-ultratool': ('appendix/cross_benchmark/planning_s1', 'model,taskbench_node_f1,taskbench_edge_f1,taskbench_planning_score,ultratool_node_f1,ultratool_edge_f1,ultratool_planning_score'),
}

def braced(text, pos):
    if text[pos] != '{':
        raise ValueError('Expected braced TeX argument')
    depth = 1
    for end in range(pos + 1, len(text)):
        if text[end] in '{}' and text[end - 1] != '\\':
            depth += 1 if text[end] == '{' else -1
            if depth == 0:
                return text[pos + 1:end], end + 1
    raise ValueError('Unclosed TeX argument')

def clean(value):
    value = value.replace('{,}', ',')
    for command in ['model', 'metric', 'textbf', 'mathbf', 'emph', 'mathrm', 'text', 'operatorname', 'textrm']:
        token = '\\' + command
        while token + '{' in value:
            start = value.index(token + '{')
            inner, end = braced(value, start + len(token))
            value = value[:start] + inner + value[end:]
    replacements = {'\\geq': '>=', '\\rightarrow': 'to', '\\to': 'to', '\\Delta': 'Delta',
                    '\\%': '%', '\\_': '_', '\\&': '&', '\\,': ' ', '\\;': ' ', '\\ ': ' '}
    for old, new in replacements.items():
        value = value.replace(old, new)
    value = re.sub(r'\^\{[^{}]*\}', '', value)
    value = value.replace('$', '').replace('{', '').replace('}', '')
    value = re.sub(r'\s+', ' ', value).strip()
    if '\\' in value:
        raise ValueError('Unrecognized markup in aggregate cell: ' + value)
    return value

def split_top_level(text, separator):
    depth, start, pos, values = 0, 0, 0, []
    while pos < len(text):
        if depth == 0 and text.startswith(separator, pos) and (pos == 0 or text[pos - 1] != '\\'):
            values.append(text[start:pos])
            pos += len(separator)
            start = pos
            continue
        if text[pos] in '{}' and (pos == 0 or text[pos - 1] != '\\'):
            depth += 1 if text[pos] == '{' else -1
        pos += 1
    values.append(text[start:])
    return values

def numeric_tokens(value):
    return re.findall(r'[-+]?\d+(?:[.,]\d+)*', value.replace('{,}', ','))

def parse_table(table, columns):
    start = re.search(r'\\begin\{(tabularx?)\}', table)
    if not start:
        raise ValueError('Missing table body')
    pos = start.end()
    for _ in range(2 if start.group(1) == 'tabularx' else 1):
        while table[pos].isspace():
            pos += 1
        _, pos = braced(table, pos)
    body = table[pos:table.index('\\end{' + start.group(1) + '}', pos)]
    body = body.split('\\midrule', 1)[1]
    group, rows, fingerprints = '', [], []
    for raw in split_top_level(body, '\\\\'):
        raw = re.sub(r'\\(?:toprule|midrule|bottomrule)', '', raw)
        raw = re.sub(r'\\(?:cmidrule(?:\([^)]*\))?\{[^}]+\}|addlinespace(?:\[[^]]*\])?)', '', raw).strip()
        if not raw:
            continue
        if raw.startswith('\\multicolumn'):
            at = len('\\multicolumn')
            for _ in range(3):
                while raw[at].isspace():
                    at += 1
                inner, at = braced(raw, at)
            if '\\footnotesize' in inner:
                continue
            group = clean(inner)
            continue
        cells = split_top_level(raw, '&')
        if len(cells) != len(columns):
            raise ValueError(f'Expected {len(columns)} cells, got {len(cells)}: {raw}')
        normalized = [clean(cell) for cell in cells]
        for before, after in zip(cells, normalized):
            if numeric_tokens(before) != numeric_tokens(after):
                raise ValueError('Numeric content changed during normalization')
        rows.append({'group': group, **dict(zip(columns, normalized))})
        fingerprints.append([numeric_tokens(cell) for cell in normalized])
    if not rows:
        raise ValueError('Empty aggregate export')
    # Carry an explicitly displayed multirow scope into subsequent blank cells.
    if columns[0] in {'subset'}:
        previous = ''
        for row in rows:
            previous = row[columns[0]] or previous
            row[columns[0]] = previous
    return rows, fingerprints

def export(main_path, appendix_path, output):
    output.mkdir(parents=True, exist_ok=False)
    entries, sources, seen, exported_rows = [], [], set(), {}
    for manuscript, path in [('main', main_path), ('appendix', appendix_path)]:
        data = path.read_bytes()
        sources.append({'manuscript': manuscript, 'sha256': hashlib.sha256(data).hexdigest()})
        text = re.sub(r'(?<!\\)%[^\n]*', '', data.decode())
        for match in re.finditer(r'\\begin\{table\*?\}.*?\\end\{table\*?\}', text, re.S):
            table = match.group()
            labels = re.findall(r'\\label\{([^}]+)\}', table)
            selected = [label for label in labels if label in TABLES]
            if not selected:
                continue
            label = selected[0]
            relative, columns = TABLES[label]
            columns = columns.split(',')
            rows, fingerprints = parse_table(table, columns)
            if label in seen:
                if exported_rows[label] != rows:
                    raise ValueError('Conflicting duplicate table label: ' + label)
                continue
            seen.add(label)
            exported_rows[label] = rows
            destination = output / (relative + '.csv')
            destination.parent.mkdir(parents=True, exist_ok=True)
            with destination.open('w', newline='') as handle:
                writer = csv.DictWriter(handle, fieldnames=['group', *columns])
                writer.writeheader()
                writer.writerows(rows)
            caption = ''
            cap = re.search(r'\\caption\{', table)
            if cap:
                caption, _ = braced(table, cap.end() - 1)
                # Captions retain TeX reference labels for an exact table map.
                caption = re.sub(r'\s+', ' ', caption).strip()
            entries.append({'file': str(destination.relative_to(output)), 'manuscript': manuscript,
                            'table_label': label, 'caption_tex': caption, 'rows': len(rows),
                            'columns': ['group', *columns], 'sha256': hashlib.sha256(destination.read_bytes()).hexdigest(),
                            'numeric_cell_fidelity_checked': True,
                            'numeric_fingerprint_sha256': hashlib.sha256(json.dumps(fingerprints).encode()).hexdigest()})
    if seen != set(TABLES):
        raise ValueError('Missing expected aggregate tables: ' + ', '.join(sorted(set(TABLES) - seen)))
    manifest = {'title': 'WWW 2027 manuscript-aligned core aggregate results',
                'precision': 'Exact displayed manuscript values; no unreported precision is inferred.',
                'primary_planning_stage': 'S3', 'auxiliary_planning_stage': 'S1',
                'open_weight_configurations': 9, 'closed_model_configurations': 11,
                'tasks': 165, 'native_references': 165, 'sampled_candidate_orderings': 1786,
                'candidate_orderings': 1429, 'total_references': 1594,
                'tasks_with_retained_candidates': 64, 'retained_rich_tasks': 56,
                'original_rich_cohort_tasks': 63,
                'replay_filters_main_references': True, 'manuscript_snapshots': sources,
                'raw_questions_answers_traces_or_identifiers_included': False,
                'retention_policy': 'At least one complete, error/mock/empty-free, gold-equivalent executor replay; all Native chains retained',
                'tables': entries}
    (output / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n')
    return manifest

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--main', type=Path, required=True)
    parser.add_argument('--appendix', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    manifest = export(args.main, args.appendix, args.output)
    print(json.dumps({'tables': len(manifest['tables']), 'rows': sum(t['rows'] for t in manifest['tables']),
                      'primary_planning_stage': 'S3', 'precision': 'manuscript_display'}))
