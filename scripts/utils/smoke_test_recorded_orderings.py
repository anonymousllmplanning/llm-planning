#!/usr/bin/env python3
"""Offline checks for recorded-chain references and Stage-4 plan handoff."""

from __future__ import annotations

import contextlib
import copy
import io
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

os.environ["LLM_PLANNING_NODE_MATCH_BACKEND"] = "token"
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.evaluation import metrics
from src.evaluation.ordering_references import load_ordering_schedule, ordering_references
from src.evaluation.runner import evaluate_file
from src.inference import runner


def native_record():
    nodes = [{"node_id": f"n{i}", "step_index": i, "label": label}
             for i, label in enumerate(("Find source", "Read values", "Calculate answer"))]
    return {"meta": {"id": "synthetic-task", "dataset": "gaia", "has_answer": False},
            "gold": {"plan_dag": {"nodes": nodes, "edges": [
                {"source": "n0", "target": "n1"}, {"source": "n1", "target": "n2"}]}}}


def schedule_rows(orders):
    return [{"task_id": "synthetic-task", "kind": "native" if rank == 0 else "candidate",
             "candidate_rank": rank, "ordering": order} for rank, order in enumerate(orders)]


class RecordedOrderTests(unittest.TestCase):
    def setUp(self):
        self.record = native_record()
        self.evaluator = metrics.ASTEvaluationSystem(use_embeddings=False)

    def score(self, orders, prediction):
        refs = ordering_references(self.record, schedule_rows(orders))
        return self.evaluator.evaluate_record(self.record["gold"], {"plan_dag": prediction},
                                              {"dataset": "gaia", "plan_source": "stage3",
                                               "reference_plan_dags": refs})["plan"]

    def test_second_candidate_can_win_and_alignment_is_shared(self):
        orders = [[0, 1, 2], [1, 0, 2], [2, 1, 0]]
        refs = ordering_references(self.record, schedule_rows(orders))
        with patch.object(metrics, "_semantic_node_alignment", wraps=metrics._semantic_node_alignment) as align:
            scores = self.score(orders, refs[2]["dag"])
        self.assertEqual(scores["reference_selected"], "candidate_2")
        self.assertEqual(scores["planning_score"], 1.0)
        self.assertEqual(align.call_count, 1)
        self.assertIs(refs[2]["dag"]["nodes"], self.record["gold"]["plan_dag"]["nodes"])

    def test_native_wins_exact_ties(self):
        prediction = dict(self.record["gold"]["plan_dag"], edges=[])
        scores = self.score([[0, 1, 2], [2, 1, 0]], prediction)
        self.assertEqual(scores["reference_selected"], "chain")
        self.assertEqual(scores["augmented_best_planning_score"], scores["chain_only_planning_score"])

    def test_native_only_equals_native(self):
        prediction = self.record["gold"]["plan_dag"]
        self.assertEqual(self.score([[0, 1, 2]], prediction)["planning_score"],
                         self.evaluator.evaluate_plan_dag(prediction, prediction).planning_score)

    def test_stage3_snapshot_precedes_later_plan(self):
        native = self.record["gold"]["plan_dag"]
        pred = {"plan_dag": {"nodes": [], "edges": []},
                "_trace": {"stage3_prediction": {"plan_dag": native}}}
        scores = self.evaluator.evaluate_record(self.record["gold"], pred,
                                                {"dataset": "gaia", "plan_source": "stage3"})["plan"]
        self.assertEqual(scores["planning_score"], 1.0)

    def test_bad_permutation_duplicate_and_constraint_are_errors(self):
        for orders in ([[0, 1, 2], [0, 1, 1]], [[0, 1, 2], [0, 1, 2]], [[0, 1, 2], [0, 1]]):
            with self.subTest(orders=orders), self.assertRaises(ValueError):
                ordering_references(self.record, schedule_rows(orders))
        self.record["ad_sd_v6"] = {"dag_edges": [{"from": 0, "to": 1}]}
        with self.assertRaisesRegex(ValueError, "constraint"):
            ordering_references(self.record, schedule_rows([[0, 1, 2], [1, 0, 2]]))

    def test_edgeless_requires_nonempty_semantic_alignment(self):
        gold = {"nodes": [{"node_id": "n0", "step_index": 0, "label": "Find source"}], "edges": []}
        empty = self.evaluator.evaluate_plan_dag(gold, {"nodes": [], "edges": []})
        self.assertEqual(empty.edge_f1, 0.0)
        pred = {"nodes": [{"node_id": "p0", "step_index": 0, "label": "Unrelated"}], "edges": []}
        with patch.object(metrics, "_semantic_node_similarity", return_value=0.0):
            unmatched = self.evaluator.evaluate_plan_dag(gold, pred)
        self.assertEqual(unmatched.edge_f1, 0.0)
        self.assertEqual(self.evaluator.evaluate_plan_dag(gold, gold).edge_f1, 1.0)

    def test_schedule_loader_and_runner_keep_all_candidates(self):
        orders = [[0, 1, 2], [1, 0, 2], [2, 1, 0]]
        rows = schedule_rows(orders)
        pred = copy.deepcopy(self.record)
        pred["pred"] = {"plan_dag": ordering_references(self.record, rows)[2]["dag"]}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "schedule.json").write_text(json.dumps(rows))
            (root / "predictions.jsonl").write_text(json.dumps(pred) + "\n")
            schedule = load_ordering_schedule(root / "schedule.json")
            with contextlib.redirect_stdout(io.StringIO()):
                results, _ = evaluate_file(root / "predictions.jsonl", use_embeddings=False,
                                            gaia_ordering_schedule=schedule, plan_source="stage3")
            self.assertEqual(results[0].reference_selected, "candidate_2")
            self.assertEqual(results[0].planning_score, 1.0)
            (root / "schedule.json").write_text(json.dumps(rows + [rows[1]]))
            with self.assertRaises(ValueError):
                load_ordering_schedule(root / "schedule.json")

    def test_stage4_prompt_preserves_labels_arguments_outputs_and_dependencies(self):
        prompts = []
        def local_response(**kwargs):
            prompts.append(kwargs["prompt"])
            return json.dumps({"tool_id": "submit_final_answer", "arguments": {"answer": "42"}})
        record = {"meta": {"id": "synthetic-task"}, "query": {"user_query": "Calculate six times seven"}}
        plan = {"plan_dag": {
            "nodes": [
                {"node_id": "microsoft", "tool_id": "web_search", "label": "Search Microsoft annual revenue",
                 "arguments": {"query": "Microsoft annual revenue 2025"}, "output_vars": ["microsoft_result"]},
                {"node_id": "apple", "tool_id": "web_search", "label": "Search Apple annual revenue",
                 "description": "Use the Apple annual filing", "task": "Read Apple revenue",
                 "arguments": {"query": "Apple annual revenue 2025"}, "output_vars": ["apple_result"]}],
            "edges": [{"source": "microsoft", "target": "apple", "edge_type": "control_dep"}]},
            "tool_calls": [{"node_id": "apple", "tool_id": "web_search",
                            "arguments": {"query": "Apple annual report 2025"}}]}
        original = copy.deepcopy(plan)
        args = SimpleNamespace(backend="api", model_name="offline-test", tool_scope="global", dataset="gaia_cat_A")
        with patch.object(runner, "generate_with_api", side_effect=local_response), \
             patch.object(runner, "execute_tool", return_value="42"), \
             contextlib.redirect_stdout(io.StringIO()):
            runner.run_stage_4_iterative_execution(record, plan, [], args, None, None, max_turns=1)
        self.assertTrue(prompts)
        for value in ("Search Microsoft annual revenue", "Search Apple annual revenue",
                      '"query": "Microsoft annual revenue 2025"', '"query": "Apple annual revenue 2025"',
                      "microsoft_result", "apple_result", '"node_id": "microsoft"',
                      '"source": "microsoft"', '"target": "apple"', '"edge_type": "control_dep"',
                      "Use the Apple annual filing", "Read Apple revenue", '"planned_tool_calls"',
                      "Apple annual report 2025"):
            self.assertIn(value, prompts[0])
        self.assertEqual(plan, original)
        self.assertIn("not observed evidence", prompts[0])


if __name__ == "__main__":
    unittest.main()
