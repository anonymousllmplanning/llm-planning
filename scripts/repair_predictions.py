#!/usr/bin/env python3
"""Normalize saved S5 responses and run at most one technical repair per case."""
from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.inference.technical_repair import defects, gate, normalize_saved_s5


def read_rows(path):
    with Path(path).open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_rows(path, records):
    with Path(path).open('x') as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + '\n')


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def run_child(command, log_path, environment):
    secrets = {v for k, v in environment.items() if k.endswith('API_KEY') and len(v) > 10}
    failure = None
    with log_path.open('x') as log:
        process = subprocess.Popen(command, cwd=ROOT, env=environment, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                   text=True, bufsize=1)
        for line in process.stdout:
            for secret in secrets:
                line = line.replace(secret, '[REDACTED]')
            log.write(line)
            log.flush()
            if '[ERROR] API call failed after' in line:
                failure = 'exhausted_model_transport'
                process.terminate()
        return process.wait(), failure


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True, help='Saved unified predictions in JSONL')
    parser.add_argument('--output-dir', type=Path, required=True, help='A new directory; existing evidence is never overwritten')
    parser.add_argument('--model', help='Exact OpenAI-compatible model ID, required for execution')
    parser.add_argument('--select-only', action='store_true', help='Normalize and select defects without model calls')
    args = parser.parse_args()
    if not args.select_only and not args.model:
        parser.error('--model is required unless --select-only is used')
    source, output = args.input.resolve(strict=True), args.output_dir.resolve()
    records = read_rows(source)
    ids = [str(row['meta']['id']) for row in records]
    if len(ids) != len(set(ids)):
        parser.error('Input contains duplicate task IDs')
    output.mkdir(parents=True, exist_ok=False)
    shutil.copy2(source, output / 'baseline.jsonl')
    normalized, audits, cases = [], [], []
    for row in records:
        value, audit = normalize_saved_s5(row)
        normalized.append(value)
        audits.append(audit)
        route, reasons = defects(value)
        if route:
            cases.append({'sample_id': str(row['meta']['id']), 'route': route, 'reasons': reasons,
                          'eligibility_uses_gold_or_EM_or_score': False, 'fixed_new_attempts': 1})
    cases.sort(key=lambda value: ({'s5': 0, 's4': 1, 's3': 2, 's1': 3}[value['route']], value['sample_id']))
    checkpoint = output / 'normalized.jsonl'
    write_rows(checkpoint, normalized)
    write_json(output / 'parser_audit.json', audits)
    write_json(output / 'cases.json', cases)
    manifest = {'created_at': datetime.now(timezone.utc).isoformat(), 'model': args.model,
                'baseline_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
                'normalized_sha256': hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                'rows': len(records), 'selected': len(cases), 'fixed_attempts_per_case': 1,
                'selection_and_promotion_read_gold_or_scores': False, 'select_only': args.select_only,
                'stage1_dag_handoff': 'steps_only', 'S1_max_tokens': 8192, 'S3_max_tokens': 16384,
                'S5_max_tokens': 4096, 'S4_max_turns': 15, 'credentials_persisted': False}
    write_json(output / 'manifest.json', manifest)
    if args.select_only:
        print(json.dumps({'state': 'selected_without_inference', 'rows': len(records), 'selected': len(cases)}))
        return 0
    environment = dict(os.environ, PYTHONPATH=str(ROOT), CUDA_VISIBLE_DEVICES='',
                       LLM_API_TIMEOUT=os.environ.get('LLM_API_TIMEOUT', '900'))
    by_id = {str(row['meta']['id']): row for row in normalized}
    decisions = []
    for case in cases:
        attempt = output / 'attempts' / case['sample_id'] / 'attempt_0001'
        attempt.mkdir(parents=True, exist_ok=False)
        baseline = by_id[case['sample_id']]
        record = copy.deepcopy(baseline)
        record.pop('pred', None)
        inp, out = attempt / 'input.jsonl', attempt / 'predictions.jsonl'
        write_rows(inp, [record])
        command = [sys.executable, '-u', '-m', 'src.inference.runner', '--unified_path', str(inp),
                   '--output', str(out), '--model_name', args.model, '--backend', 'api', '--mode', 'answer',
                   '--dataset', 'gaia', '--tool_scope', 'record', '--skill_mode', 'no_skill',
                   '--max_turns', '15', '--save_trace', '--checkpoint_predictions', str(checkpoint),
                   '--resume_from_stage', case['route'], '--stage1_dag_handoff', 'steps_only']
        write_json(attempt / 'command.json', {'command': command, 'credentials_persisted': False})
        code, failure = run_child(command, attempt / 'run.log', environment)
        produced = read_rows(out) if out.exists() else []
        decision = (gate(produced[0], baseline, case['route']) if len(produced) == 1 else
                    {'sample_id': case['sample_id'], 'route': case['route'], 'promote': False,
                     'issues': ['no_single_prediction']})
        decision.update(returncode=code, operational_failure=failure)
        if code or failure:
            decision['promote'] = False
        if decision['promote']:
            by_id[case['sample_id']] = produced[0]
        write_json(attempt / 'technical_gate.json', decision)
        decisions.append(decision)
        write_json(output / 'decisions.json', decisions)
        if failure:
            break
    write_rows(output / 'predictions.jsonl', [by_id[sid] for sid in ids])
    write_json(output / 'completion.json', {'attempted': len(decisions),
        'promoted': sum(bool(r['promote']) for r in decisions),
        'stopped_transport_failure': any(r['operational_failure'] for r in decisions),
        'all_original_and_repaired_traces_retained': True, 'answer_correctness_not_a_promotion_gate': True})
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
