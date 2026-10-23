#!/usr/bin/env python3
"""Check exported aggregates, cross-table accounting and diagram metadata offline."""
import csv
import hashlib
import json
from pathlib import Path
import re
from PIL import Image
import pytesseract

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / 'results/www2027'

def rows(relative):
    with (RESULTS / relative).open(newline='') as handle:
        return list(csv.DictReader(handle))

def main():
    manifest = json.loads((RESULTS / 'manifest.json').read_text())
    checks = []
    def check(name, passed):
        if not passed:
            raise AssertionError(name)
        checks.append(name)
    for table in manifest['tables']:
        path = RESULTS / table['file']
        check('checksum:' + table['file'], hashlib.sha256(path.read_bytes()).hexdigest() == table['sha256'])
        check('rows:' + table['file'], len(rows(table['file'])) == table['rows'])
    headline = rows('main/three_facet_headline.csv')
    models = [r for r in headline if not r['model'].endswith('mean')]
    check('twenty_configurations', len(models) == 20)
    check('closed_eleven_open_nine', sum(r['group']=='Closed-model' for r in models)==11 and
          sum(r['group']=='Open-weight' for r in models)==9)
    check('main_is_s3', manifest['primary_planning_stage']=='S3')
    check('all_reference_counts', manifest['native_references']==165 and manifest['candidate_orderings']==1786)
    check('no_replay_filter', manifest['replay_filters_main_references'] is False)
    check('open_accepted_count_383', sum(round(float(r['accepted_rate'])*165) for r in models
          if r['group']=='Open-weight') == 383)
    closed = {r['model']:r for r in rows('appendix/closed_models/planning_headline_s3.csv')}
    for row in models:
        check('nondecreasing_edge:' + row['model'], float(row['edge_f1_augmented'])>=float(row['edge_f1_native']))
        if row['group']=='Closed-model':
            for metric in ['node_f1','edge_f1_native','edge_f1_augmented']:
                check('closed_table_agreement:' + row['model'] + ':' + metric,
                      abs(float(row[metric])-float(closed[row['model']][metric]))<=.00055)
    transitions = rows('appendix/replay/outcome_transitions.csv')
    coverage = rows('appendix/replay/coverage_and_outcomes.csv')
    for key, record in zip(['llama4','gemma4'],coverage):
        check('replay_total:' + key, sum(int(r[key]) for r in transitions)==1786)
        check('replay_same_outcome:' + key, sum(int(r[key]) for r in transitions[:3])==int(record['same_outcome']))
    human = rows('appendix/human_dependency_audit/pair_majorities.csv')
    check('human_majority_accounting', all(int(r['present'])+int(r['absent'])+int(r['no_majority'])==int(r['pairs']) for r in human))
    raters = rows('appendix/human_dependency_audit/anonymous_rater_counts.csv')
    check('three_anonymous_raters_422_each', len(raters)==3 and all(
        sum(int(r[k]) for k in r if k not in {'rater','group'})==422 for r in raters))
    cross = rows('appendix/cross_benchmark/planning_s1.csv')
    check('cross_benchmark_shared_five_models', sum(r['tasks']=='1,000' for r in cross)==10)
    for row in cross:
        check('cross_planning_mean:' + row['group'] + ':' + row['model'],
              abs((float(row['node_f1'])+float(row['edge_f1']))/2-float(row['planning_score']))<=.00055)
    facts = json.loads((ROOT/'docs/assets/framework_overview.facts.json').read_text())
    check('figure_reference_accounting', facts['native_references']+facts['non_native_candidates']==facts['total_references']==1951)
    check('figure_primary_s3', facts['primary_planning_stage']=='S3')
    check('figure_catalogs', facts['open_weight_catalog_actions']==14 and facts['closed_model_catalog_actions']==16)
    check('figure_semantic_records', facts['artifact_dependency_records']+facts['state_dependency_records']==1334)
    image = Image.open(ROOT/'docs/assets/framework_overview.png')
    check('png_no_text_metadata', all(k in {'dpi'} for k in image.info))
    pipeline = Image.open(ROOT/'docs/assets/dependency_synthesis_pipeline.png')
    check('pipeline_no_text_metadata', all(k in {'dpi'} for k in pipeline.info))
    visible = pytesseract.image_to_string(image) + '\n' + pytesseract.image_to_string(image,config='--psm 11')
    normalized_visible = visible.replace(' ', '').replace('\n','')
    check('figure_current_counts', all(s in normalized_visible for s in ['1,786','1,951','1,207']))
    check('figure_policy_metadata', facts['replay_filters_main_reference_set'] is False)
    check('figure_old_counts_removed', not any(s in visible for s in ['1,771','1,522','1,357','414 dropped']))
    report={'status':'passed','checks':len(checks),'tables':len(manifest['tables']),
            'aggregate_rows':sum(t['rows'] for t in manifest['tables']),
            'scope':'Displayed-value fidelity, checksums, aggregate accounting and asset metadata; no claim of independently rerunning all experiments.',
            'api_calls':0,'gpu_used':False}
    print(json.dumps(report))
    return report

if __name__=='__main__':
    main()
