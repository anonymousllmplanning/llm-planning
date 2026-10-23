"""Outcome-blind normalization, selection and fixed-attempt promotion rules."""
from __future__ import annotations
import collections
import copy
import hashlib
import json
from pathlib import Path
from src.inference import answer_recovery as PARSER

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def dag_problems(graph):
    problems = []
    if not isinstance(graph, dict): return ["missing_graph"]
    nodes, edges = graph.get("nodes"), graph.get("edges")
    if not isinstance(nodes, list) or not nodes: return ["empty_nodes"]
    if not isinstance(edges, list): return ["missing_edges_array"]
    ids = [str(n.get("node_id", n.get("id", ""))) for n in nodes if isinstance(n, dict)]
    if len(ids) != len(nodes) or "" in ids: problems.append("invalid_node_id")
    if len(ids) != len(set(ids)): problems.append("duplicate_node_id")
    if any(not isinstance(n, dict) or not str(n.get("label") or n.get("description") or n.get("task") or "").strip() for n in nodes): problems.append("blank_node_label")
    neighbors = collections.defaultdict(list); indegree = {node: 0 for node in ids}
    for edge in edges:
        if not isinstance(edge, dict): problems.append("invalid_edge"); continue
        src = str(edge.get("source", edge.get("from", ""))); dst = str(edge.get("target", edge.get("to", "")))
        if src not in indegree or dst not in indegree: problems.append("dangling_edge"); continue
        neighbors[src].append(dst); indegree[dst] += 1
    queue = collections.deque(node for node, degree in indegree.items() if degree == 0); visited = 0
    while queue:
        node = queue.popleft(); visited += 1
        for child in neighbors[node]:
            indegree[child] -= 1
            if indegree[child] == 0: queue.append(child)
    if visited != len(indegree): problems.append("cycle")
    return sorted(set(problems))

def text(value):
    if value is None:
        return ''
    return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)

def normalize_saved_s5(record):
    """Use the same parser on every stored raw S5 response, without reading gold."""
    result = copy.deepcopy(record)
    prediction = result.get('pred') or {}
    trace = prediction.get('_trace') or {}
    previous = trace.get('stage5') or {}
    raw = previous.get('raw_output')
    before = text((prediction.get('final_answer') or {}).get('answer'))
    parsed = PARSER.parse_stage5_answer(raw if isinstance(raw, str) else '')
    after = text(parsed['final_answer'].get('answer'))
    updated = dict(previous)
    updated.update(parsed)
    updated['uniform_parser_version'] = 'uniform-technical-repair-v2'
    trace.setdefault('uniform_parser_history', []).append(dict(
        previous_parse_status=previous.get('parse_status'),
        previous_final_answer=copy.deepcopy(prediction.get('final_answer')),
        raw_output_sha256=hashlib.sha256((raw or '').encode()).hexdigest(),
        parser_sha256=sha(Path(PARSER.__file__))))
    trace['stage5'] = updated
    prediction['final_answer'] = copy.deepcopy(parsed['final_answer'])
    handoff = copy.deepcopy(prediction.get('_answer_handoff') or trace.get('answer_handoff') or {})
    handoff.update(stage5_answer=after, stage5_answer_present=bool(after.strip()),
                   stage5_parse_status=parsed['parse_status'], stage5_parse_error=parsed.get('parse_error'),
                   uniform_parser_applied=True)
    # Keep the native S4 submission as separate evidence; never replace S5 with it.
    native = text(handoff.get('native_submit_answer'))
    present = bool(handoff.get('submit_tool_present'))
    same = bool(native.strip() and after.strip() and native == after)
    if native.strip():
        status = 'native_submit_consistent' if not after.strip() or same else 'native_submit_stage5_disagree'
        source, effective = 'native_submit_tool', native
    elif after.strip():
        status = 'submit_tool_empty_stage5_backfill' if present else 'missing_submit_tool_stage5_backfill'
        source, effective = 'stage5_final_answer', after
    else:
        status = 'submit_tool_empty_and_stage5_missing' if present else 'missing_submit_and_stage5_answer'
        source, effective = 'missing', ''
    handoff.update(status=status, source=source, effective_submit_answer=effective,
                   submit_stage5_same=same if native.strip() and after.strip() else None,
                   reason='Native S4 submission and uniformly parsed S5 answer are recorded separately; answer scoring uses S5.')
    prediction['submit_answer'] = effective
    prediction['_answer_handoff'] = handoff
    trace['answer_handoff'] = copy.deepcopy(handoff)
    prediction['_trace'] = trace
    result['pred'] = prediction
    return result, dict(sample_id=record['meta']['id'], before_answer=before, after_answer=after,
                        answer_changed=before != after, recovered_empty=not before.strip() and bool(after.strip()),
                        rejected_prior_nonempty=bool(before.strip()) and not after.strip(),
                        parse_status=parsed['parse_status'], gold_or_scores_read=False)

def defects(record):
    prediction = record.get('pred') or {}
    trace = prediction.get('_trace') or {}
    s3 = trace.get('stage3_prediction') or {}
    source = prediction.get('_abs_plan_dag_source') or trace.get('abs_plan_dag_source')
    findings = []
    for issue in dag_problems(prediction.get('abs_plan_dag')):
        findings.append(('s1', 's1_' + issue))
    if source not in {'abs_plan_dag', 'abstract_plan_dag', 'plan_dag', 'dag'}:
        findings.append(('s1', 's1_fallback_or_missing_explicit_graph'))
    for issue in dag_problems(s3.get('plan_dag')):
        findings.append(('s3', 's3_' + issue))
    if trace.get('used_stage3_fallback') or s3.get('_stage3_used_fallback'):
        findings.append(('s3', 's3_fallback'))
    # Truncated planning generations are selected by metadata, never score magnitude.
    for stage, generation in [('s1', (trace.get('abstract_plan') or {}).get('_generation') or {}),
                              ('s3', s3.get('_stage3_generation') or {})]:
        if (generation.get('api_metadata') or {}).get('finish_reason') == 'length':
            findings.append((stage, stage + '_generation_truncated'))
    if prediction.get('_execution_error') or not isinstance(trace.get('stage4_decision_stream'), list):
        findings.append(('s4', 'execution_or_missing_saved_stream'))
    stream = trace.get('stage4_decision_stream') or []
    if any(event.get('status') == 'invalid_decision' for event in stream):
        findings.append(('s4', 'unparseable_s4_action'))
    if any(event.get('fallback_raw_output') is not None or
           event.get('last_resort_raw_output') is not None or event.get('last_resort_mode') is not None
           for event in stream):
        findings.append(('s4', 'model_specific_s4_recovery_policy'))
    if not text((prediction.get('final_answer') or {}).get('answer')).strip():
        findings.append(('s5', 'empty_or_ambiguous_saved_s5_after_uniform_parse'))
    if findings:
        priority = {'s1': 0, 's3': 1, 's4': 2, 's5': 3}
        route = min((stage for stage, _ in findings), key=priority.get)
    else:
        route = None
    return route, [reason for _, reason in findings]

def gate(repaired, baseline, route):
    """Promotion depends only on format, graph validity, and upstream preservation."""
    prediction = repaired.get('pred') or {}
    trace = prediction.get('_trace') or {}
    s3 = trace.get('stage3_prediction') or {}
    issues = []
    if repaired.get('meta', {}).get('id') != baseline['meta']['id']:
        issues.append('task_id_mismatch')
    if repaired.get('query') != baseline.get('query'):
        issues.append('query_or_attachments_changed')
    for stage, graph in [('s1', prediction.get('abs_plan_dag')), ('s3', s3.get('plan_dag'))]:
        issues.extend(stage + '_' + problem for problem in dag_problems(graph))
    for stage, generation in [('s1', (trace.get('abstract_plan') or {}).get('_generation') or {}),
                              ('s3', s3.get('_stage3_generation') or {})]:
        if (generation.get('api_metadata') or {}).get('finish_reason') == 'length':
            issues.append(stage + '_generation_still_truncated')
    if trace.get('used_stage3_fallback') or s3.get('_stage3_used_fallback'):
        issues.append('s3_fallback_remains')
    if (prediction.get('_abs_plan_dag_source') or trace.get('abs_plan_dag_source')) not in {'abs_plan_dag', 'abstract_plan_dag', 'plan_dag', 'dag'}:
        issues.append('s1_not_explicit_graph')
    for flag in ('_error', '_parse_error', '_execution_error'):
        if prediction.get(flag):
            issues.append(flag)
    if not text((prediction.get('final_answer') or {}).get('answer')).strip():
        issues.append('empty_final_s5')
    if not isinstance(trace.get('stage4_decision_stream'), list):
        issues.append('missing_s4_stream')
    else:
        if any(event.get('status') == 'invalid_decision' for event in trace['stage4_decision_stream']):
            issues.append('unparseable_s4_action_remains')
        if any(event.get('fallback_raw_output') is not None or event.get('last_resort_raw_output') is not None
               or event.get('last_resort_mode') is not None for event in trace['stage4_decision_stream']):
            issues.append('model_specific_s4_recovery_remains')
    old_pred = baseline['pred']
    old_trace = old_pred['_trace']
    if route in {'s3', 's4', 's5'}:
        for name in ('abstract_plan', 'abs_plan_dag', 'created_tools', 'creation_reasoning', 'creation_generation'):
            if trace.get(name) != old_trace.get(name):
                issues.append('saved_upstream_changed_' + name)
        if prediction.get('abs_plan_dag') != old_pred.get('abs_plan_dag'):
            issues.append('saved_prediction_s1_changed')
    if route in {'s4', 's5'} and s3 != old_trace.get('stage3_prediction'):
        issues.append('saved_s3_changed')
    if route == 's5':
        for name in ('tool_calls', 'tool_outputs'):
            if prediction.get(name) != old_pred.get(name):
                issues.append('saved_s4_changed_' + name)
        if trace.get('stage4_decision_stream') != old_trace.get('stage4_decision_stream'):
            issues.append('saved_s4_stream_changed')
        def upstream_only(value):
            result = copy.deepcopy(value)
            for name in ('final_answer', 'submit_answer', '_answer_handoff'):
                result.pop(name, None)
            saved_trace = result.get('_trace') or {}
            for name in ('stage5', 'answer_handoff', 'repair_provenance',
                         'stage5_repair_history', 'stage5_repair_api_generation_events'):
                saved_trace.pop(name, None)
            result['_trace'] = saved_trace
            return result
        if upstream_only(prediction) != upstream_only(old_pred):
            issues.append('s5_changed_fields_outside_explicit_allowlist')
    return dict(sample_id=baseline['meta']['id'], route=route, technical_pass=not issues,
                promote=not issues, issues=issues, gate_reads_EM_or_planning_scores=False)
