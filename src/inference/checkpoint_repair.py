"""Versioned stage checkpoint reuse for fixed, outcome-blind repair attempts."""
from __future__ import annotations
import copy
import hashlib
import json
from pathlib import Path
from typing import Any


def load_repair_checkpoints(path: Path | None) -> tuple[dict[str, dict], str | None]:
    if path is None:
        return {}, None
    raw = path.read_bytes()
    records = {}
    # JSONL records are separated by physical LF bytes. Unicode line/paragraph
    # separators (U+2028/U+2029) are valid inside JSON strings, not record breaks.
    for line in raw.decode().split('\n'):
        if not line.strip():
            continue
        rec = json.loads(line)
        sid = str(rec.get('meta', {}).get('id') or rec.get('task_id') or rec.get('id'))
        if sid == 'None' or sid in records:
            raise ValueError('Repair checkpoints need unique nonempty task IDs')
        records[sid] = rec
    return records, hashlib.sha256(raw).hexdigest()


def checkpoint_components(record: dict, checkpoint: dict, stage: str,
                          recover_top_level_graph: bool = False) -> dict[str, Any]:
    """Reuse original upstream outputs; never supply gold to an LLM prompt.

    Promotion of original top-level nodes is an envelope repair, not regeneration
    or a merge with the erroneous fallback chain. Original attempts stay intact.
    """
    if stage not in {'s3', 's4', 's5'}:
        raise ValueError('Checkpoint reuse is only defined for s3, s4 or s5')
    if record.get('query') != checkpoint.get('query'):
        raise ValueError('Checkpoint question/attachments differ from the input')
    pred = checkpoint.get('pred') or {}
    trace = pred.get('_trace') or {}
    abstract = trace.get('abstract_plan')
    if not isinstance(abstract, dict):
        raise ValueError('Checkpoint has no saved Stage 1 abstract plan')
    result = {'abstract': copy.deepcopy(abstract),
              'new_tools': copy.deepcopy(trace.get('created_tools') or []),
              'creation_reasoning': copy.deepcopy(trace.get('creation_reasoning')),
              'creation_generation': copy.deepcopy(trace.get('creation_generation')),
              'checkpoint_api_generation_events': copy.deepcopy(trace.get('api_generation_events') or []),
              'stage3_prediction': None,
              'checkpoint_stage1_dag_handoff': trace.get('stage1_dag_handoff', 'steps_only'),
              'recovered_top_level_graph': False,
              'original_stage3_fallback': bool(trace.get('used_stage3_fallback'))}
    if stage == 's5':
        if not isinstance(pred.get('tool_outputs'), list) or not isinstance(pred.get('tool_calls'), list):
            raise ValueError('S5 checkpoint needs saved Stage 4 tool outputs and calls')
        if not isinstance(trace.get('stage3_prediction'), dict) or not isinstance(trace.get('stage4_decision_stream'), list):
            raise ValueError('S5 checkpoint needs immutable S3 snapshot and S4 decision stream')
        if pred.get('_execution_error'):
            raise ValueError('S5 resume cannot reuse a failed execution checkpoint')
        result['saved_prediction'] = copy.deepcopy(pred)
        result['observations'] = copy.deepcopy(pred['tool_outputs'])
        result['actual_tool_calls'] = copy.deepcopy(pred['tool_calls'])
    if stage == 's4':
        snapshot = copy.deepcopy(trace.get('stage3_prediction'))
        if not isinstance(snapshot, dict):
            raise ValueError('Checkpoint has no pre-execution Stage 3 snapshot')
        if recover_top_level_graph:
            nodes, edges = snapshot.get('nodes'), snapshot.get('edges', [])
            if not isinstance(nodes, list) or not nodes or not isinstance(edges, list):
                raise ValueError('Requested top-level graph recovery has no saved graph')
            snapshot['plan_dag'] = {'nodes': copy.deepcopy(nodes), 'edges': copy.deepcopy(edges)}
            snapshot.pop('nodes', None)
            snapshot.pop('edges', None)
            # Regenerate planned calls from the preserved tool nodes. Do not
            # retain the empty calls created for the erroneous fallback chain.
            snapshot['tool_calls'] = []
            snapshot['_stage3_used_fallback'] = False
            result['recovered_top_level_graph'] = True
        result['stage3_prediction'] = snapshot
    return result


def replace_stage5_only(saved_prediction: dict, stage5_result: dict,
                        answer_handoff: dict, provenance: dict,
                        api_generation_events: list | None = None) -> dict:
    """Replace S5 only, retaining every upstream value and original raw output.

    Callers must read saved_prediction from an immutable checkpoint. No S1-S4
    normalizer, graph repair or executor is invoked by this function.
    """
    pred = copy.deepcopy(saved_prediction)
    trace = pred.get('_trace')
    if not isinstance(trace, dict):
        raise ValueError('S5 replacement requires a saved trace')
    history = trace.setdefault('stage5_repair_history', [])
    history.append({'stage5': copy.deepcopy(trace.get('stage5')),
                    'answer_handoff': copy.deepcopy(trace.get('answer_handoff')),
                    'final_answer': copy.deepcopy(pred.get('final_answer')),
                    'repair_provenance': copy.deepcopy(trace.get('repair_provenance'))})
    pred['final_answer'] = copy.deepcopy(stage5_result.get('final_answer') or {'answer': '', 'reasoning': ''})
    pred['submit_answer'] = answer_handoff.get('effective_submit_answer', '')
    pred['_answer_handoff'] = copy.deepcopy(answer_handoff)
    trace['stage5'] = copy.deepcopy(stage5_result)
    trace['answer_handoff'] = copy.deepcopy(answer_handoff)
    trace['repair_provenance'] = copy.deepcopy(provenance)
    trace['stage5_repair_api_generation_events'] = copy.deepcopy(api_generation_events or [])
    return pred
