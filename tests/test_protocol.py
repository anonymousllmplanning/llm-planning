"""Synthetic protocol regressions; no external inference or benchmark data."""
import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from src.inference.answer_recovery import parse_stage5_answer
from src.inference.checkpoint_repair import checkpoint_components, replace_stage5_only
from src.inference.parsing import normalize_pred_structure, robust_json_parse
from src.inference.technical_repair import defects, gate, normalize_saved_s5


def record():
    graph = {'nodes': [{'node_id': 'n0', 'label': 'Calculate six times seven'}], 'edges': []}
    return {'meta': {'id': 'synthetic'}, 'query': {'user_query': 'Calculate six times seven'},
            'gold': {'final_answer': {'answer': '42'}},
            'pred': {'abs_plan_dag': copy.deepcopy(graph), '_abs_plan_dag_source': 'abs_plan_dag',
                     'final_answer': {'answer': '42', 'reasoning': ''},
                     'tool_calls': [], 'tool_outputs': [],
                     '_trace': {'abstract_plan': {'abstract_plan': ['Calculate six times seven']},
                                'stage3_prediction': {'plan_dag': copy.deepcopy(graph)},
                                'stage4_decision_stream': [],
                                'stage5': {'raw_output': '{"final_answer":{"answer":"42","reasoning":""}}'}}}}


class ProtocolTests(unittest.TestCase):
    def test_valid_json_keeps_literal_wrappers(self):
        for text in ['<think>literal content</think>', '```json\ninside a string\n```']:
            value = {'label': text}
            parsed, error = robust_json_parse(json.dumps(value))
            self.assertEqual(parsed, value)
            self.assertFalse(error)

    def test_root_graph_normalization_preserves_input(self):
        value = {'nodes': [{'node_id': 'n0', 'label': 'Calculate'}], 'edges': []}
        original = copy.deepcopy(value)
        parsed = normalize_pred_structure(value)
        self.assertEqual(value, original)
        self.assertEqual(parsed['plan_dag']['nodes'], original['nodes'])
        parsed['plan_dag']['nodes'][0]['label'] = 'changed'
        self.assertEqual(value, original)

    def test_selection_is_invariant_to_gold(self):
        left, right = record(), record()
        right['gold'] = {'final_answer': {'answer': 'unrelated'}, 'planning_score': -100}
        a, audit_a = normalize_saved_s5(left)
        b, audit_b = normalize_saved_s5(right)
        self.assertEqual(a['pred'], b['pred'])
        self.assertEqual(audit_a, audit_b)
        self.assertEqual(defects(a), defects(b))
        self.assertFalse(audit_a['gold_or_scores_read'])

    def test_native_submit_is_separate_from_s5(self):
        value = record()
        value['pred']['_answer_handoff'] = {'native_submit_answer': '41', 'submit_tool_present': True}
        normalized, _ = normalize_saved_s5(value)
        self.assertEqual(normalized['pred']['final_answer']['answer'], '42')
        self.assertEqual(normalized['pred']['submit_answer'], '41')
        self.assertEqual(normalized['pred']['_answer_handoff']['status'], 'native_submit_stage5_disagree')

    def test_s5_replacement_copies_and_preserves_upstream(self):
        saved = record()['pred']
        before = copy.deepcopy(saved)
        replacement = replace_stage5_only(saved, parse_stage5_answer('{"final_answer":{"answer":"43"}}'),
                                           {'effective_submit_answer': '43'}, {'route': 's5'})
        self.assertEqual(saved, before)
        for key in ['abstract_plan', 'stage3_prediction', 'stage4_decision_stream']:
            self.assertEqual(replacement['_trace'][key], before['_trace'][key])
        baseline, repaired = record(), record()
        repaired['pred'] = replacement
        self.assertTrue(gate(repaired, baseline, 's5')['promote'])
        repaired['pred']['_trace']['stage4_decision_stream'].append({'status': 'executed'})
        self.assertFalse(gate(repaired, baseline, 's5')['promote'])

    def test_checkpoint_rejects_changed_query(self):
        baseline = record()
        changed = copy.deepcopy(baseline)
        changed['query']['user_query'] = 'A different task'
        with self.assertRaisesRegex(ValueError, 'differ'):
            checkpoint_components(changed, baseline, 's5')

    def test_transport_or_parse_failure_is_not_promoted(self):
        baseline, repaired = record(), record()
        repaired['pred']['_trace']['stage4_decision_stream'] = [{'status': 'invalid_decision'}]
        self.assertFalse(gate(repaired, baseline, 's4')['promote'])

    def test_offline_selector_preserves_baseline_and_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source, output = root / 'baseline.jsonl', root / 'selection'
            source.write_text(json.dumps(record()) + '\n')
            command = [sys.executable, 'scripts/repair_predictions.py', '--input', str(source),
                       '--output-dir', str(output), '--select-only']
            result = subprocess.run(command, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual((output / 'baseline.jsonl').read_bytes(), source.read_bytes())
            self.assertFalse((output / 'attempts').exists())
            self.assertFalse(json.loads((output / 'manifest.json').read_text())['selection_and_promotion_read_gold_or_scores'])
            saved = (output / 'normalized.jsonl').read_bytes()
            self.assertNotEqual(subprocess.run(command, capture_output=True).returncode, 0)
            self.assertEqual((output / 'normalized.jsonl').read_bytes(), saved)


if __name__ == '__main__':
    unittest.main()
