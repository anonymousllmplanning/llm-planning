#!/usr/bin/env python3
"""Check final displayed aggregates, reference identities, and release assets offline."""
import csv
import hashlib
import json
from pathlib import Path
import sys
import zipfile
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.evaluation.ordering_references import load_ordering_schedule
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
    check('ten_final_tables', len(manifest['tables']) == 10)
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
    check('final_reference_accounting', manifest['native_references']==165 and
          manifest['sampled_candidate_orderings']==1786 and manifest['candidate_orderings']==1429 and
          manifest['total_references']==1594 and manifest['replay_filters_main_references'] is True)
    check('open_accepted_count_383', sum(round(float(r['accepted_rate'])*165) for r in models
          if r['group']=='Open-weight') == 383)
    for row in models:
        check('nondecreasing_edge:' + row['model'], float(row['edge_f1_augmented'])>=float(row['edge_f1_native']))
    cross = rows('appendix/cross_benchmark/planning_s1.csv')
    check('cross_five_models_and_mean', len(cross)==6)
    for row in cross:
        for benchmark in ['taskbench','ultratool']:
            check('cross_planning_mean:' + benchmark + ':' + row['model'],
                  abs((float(row[benchmark+'_node_f1'])+float(row[benchmark+'_edge_f1']))/2-
                      float(row[benchmark+'_planning_score']))<=.00055)
    refs = ROOT/'references'
    with (refs/'reference_set_1594.csv').open(encoding='utf-8-sig',newline='') as handle:
        csv_rows = list(csv.DictReader(handle))
    schedule = load_ordering_schedule(refs/'ordering_schedule.json')
    check('schedule_task_counts', len(schedule)==165 and
          sum(len(v)>1 for v in schedule.values())==64 and sum(len(v)>2 for v in schedule.values())==56)
    schedule_map={(tid,r['kind'],r['candidate_rank']):r['ordering'] for tid,values in schedule.items() for r in values}
    csv_map={(r['sample_id'],r['kind'],int(r['candidate_rank'])):json.loads(r['ordering']) for r in csv_rows}
    check('schedule_matches_csv', len(csv_rows)==1594 and len(csv_map)==1594 and schedule_map==csv_map)
    check('native_and_retained_counts', sum(r['kind']=='native' for r in csv_rows)==165 and
          sum(r['kind']=='candidate' for r in csv_rows)==1429)
    check('complete_permutations', all(sorted(json.loads(r['ordering']))==list(range(int(r['n_nodes']))) for r in csv_rows))
    check('recorded_precedence_flags', all(int(r['compiled_precedence_violations'])==0 for r in csv_rows))
    retained=json.loads((refs/'replay_retained_orderings.json').read_text())
    decisions={(r['task_id'],'candidate',r['candidate_rank']):r['ordering'] for r in retained['retained_candidates']}
    check('retained_decisions_match_csv', len(decisions)==1429 and decisions=={k:v for k,v in csv_map.items() if k[1]=='candidate'})
    check('retention_rule', retained['retention_criteria']['required_executor_successes']==1 and
          all(r['supporting_executors'] for r in retained['retained_candidates']))
    check('overview_pdf_present', (ROOT/'docs/assets/overview_figure.pdf').read_bytes().startswith(b'%PDF-'))
    for name in ['overview_figure.png','pipeline_figure.png']:
        image=Image.open(ROOT/'docs/assets'/name)
        check('png_metadata:'+name,all(k in {'dpi'} for k in image.info))
    report={'status':'passed','checks':len(checks),'tables':len(manifest['tables']),
            'aggregate_rows':sum(t['rows'] for t in manifest['tables']),
            'scope':'Final displayed-value hashes and accounting; reference identity/permutation consistency; asset checks. No live inference or independent semantic answer adjudication.',
            'api_calls':0,'gpu_used':False}
    print(json.dumps(report))
    return report

if __name__=='__main__':
    main()
