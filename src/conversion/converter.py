#!/usr/bin/env python3
"""
Unified Dataset Converter for LLM Planning Evaluation

Consolidates conversion logic for all supported datasets:
1. Delta (new and old versions with different tool catalogs)
2. UltraTool
3. TaskBench (multimedia, huggingface, dailylifeapis)
4. GAIA

This script produces a unified JSONL format compatible with:
- run_agents.py (inference)
- ast_evaluation_system.py (evaluation)
- dual_facet_evaluation_runner.py (reporting)

Usage Examples:
    # Convert new delta dataset
    python convert_all_datasets.py --mode delta_new \
        --queries data/delta_new/new_queries_delta.json \
        --tools data/delta_new/new_tools_delta.json \
        --output data/delta_new/unified_delta_new.jsonl

    # Convert old delta dataset
    python convert_all_datasets.py --mode delta_old \
        --queries data/delta_old/old_queries_delta.json \
        --tools data/delta_old/old_tools_delta.json \
        --output data/delta_old/unified_delta_old.jsonl

    # Convert UltraTool
    python convert_all_datasets.py --mode ultratool \
        --ultratool_dir data/taskbench_ultratool/UltraTool/data/English-dataset/test_set \
        --output data/unified_ultratool.jsonl

    # Convert TaskBench (multiple subsets)
    python convert_all_datasets.py --mode taskbench \
        --taskbench_dirs data/taskbench/multimedia data/taskbench/huggingface \
        --output data/unified_taskbench.jsonl

    # Convert GAIA
    python convert_all_datasets.py --mode gaia \
        --gaia_path data/gaia/gaia_2023_val.jsonl \
        --output data/unified_gaia.jsonl

    # Convert all datasets together
    python convert_all_datasets.py --mode all \
        --queries data/delta_new/new_queries_delta.json \
        --tools data/delta_new/new_tools_delta.json \
        --ultratool_dir data/taskbench_ultratool/UltraTool/data/English-dataset/test_set \
        --taskbench_dirs data/taskbench/multimedia \
        --gaia_path data/gaia/gaia_2023_val.jsonl \
        --output data/unified_all.jsonl \
        --limit 50
"""

from __future__ import annotations
import json
import argparse
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Iterable, Set
from collections import Counter


# =============================================================================
# Common Utilities
# =============================================================================

def load_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    """Yield JSON objects from a .jsonl file."""
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def load_json(path: Path) -> Any:
    """Load a JSON file."""
    with path.open() as f:
        return json.load(f)


def write_jsonl(records: List[Dict[str, Any]], out_path: Path) -> int:
    """Write records to a JSONL file."""
    count = 0
    with out_path.open("w") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            count += 1
    return count


# =============================================================================
# Delta Dataset Conversion (Both New and Old)
# =============================================================================

def load_delta_tool_catalog(tools_path: Path) -> Tuple[Dict[str, Dict], List[Dict]]:
    """
    Load Delta tool catalog and build lookup structures.

    Returns:
        - tool_lookup: Dict mapping tool_id -> tool info with schema
        - flat_tools: List of all tools in unified format
    """
    data = load_json(tools_path)

    tool_lookup = {}
    flat_tools = []

    for server in data.get("servers", []):
        server_name = server.get("server_name", "")
        for tool in server.get("tools", []):
            tool_name = tool.get("tool_name", "")
            tool_id = f"{server_name}::{tool_name}"

            # Build arguments schema from input_schema
            input_schema = tool.get("input_schema", {}) or {}
            properties = input_schema.get("properties", {}) or {}
            required_fields = input_schema.get("required", []) or []

            arguments_schema = {}
            for prop_name, prop_spec in properties.items():
                if not isinstance(prop_spec, dict):
                    continue
                arguments_schema[prop_name] = {
                    "type": prop_spec.get("type", "string"),
                    "description": prop_spec.get("description", ""),
                    "required": prop_name in required_fields,
                }

            tool_info = {
                "tool_id": tool_id,
                "tool_name": tool_name,
                "server_name": server_name,
                "description": tool.get("tool_description", ""),
                "input_type": [],
                "output_type": [],
                "arguments_schema": arguments_schema,
            }

            tool_lookup[tool_id] = tool_info
            flat_tools.append(tool_info)

    return tool_lookup, flat_tools


def convert_delta(
    queries_path: Path,
    tools_path: Path,
    subset_name: str = "mcp_tools",
    limit: Optional[int] = None,
    is_old_delta: bool = False,
) -> List[Dict[str, Any]]:
    """
    Convert Delta MCP dataset to unified format.

    Delta's ground_truth_tools structure:
    [
        [tool_at_step_0],  # Step 0: one tool (no dependencies)
        [tool_at_step_1],  # Step 1: depends on step 0
        [tool_at_step_2_option_a, tool_at_step_2_option_b],  # Step 2: alternatives
    ]

    Each tool has:
    - tool_id: "server_name::tool_name"
    - dependencies: list of step indices this step depends on
    """
    # Load tool catalog
    tool_lookup, flat_tools = load_delta_tool_catalog(tools_path)
    print(f"[INFO] Loaded {len(flat_tools)} tools from Delta catalog")

    # Load queries (can be JSON array or JSONL)
    queries_data = load_json(queries_path)
    if isinstance(queries_data, dict):
        queries = [queries_data]
    elif isinstance(queries_data, list):
        queries = queries_data
    else:
        queries = list(load_jsonl(queries_path))

    records: List[Dict[str, Any]] = []

    for query_data in queries:
        if limit is not None and len(records) >= limit:
            break

        query_id = query_data.get("query_id")
        query_text = query_data.get("query", "")
        ground_truth_tools = query_data.get("ground_truth_tools", [])
        expected_tool_count = query_data.get("ground_truth_tools_count", 0)

        # Build plan_dag nodes and edges
        nodes = []
        edges = []
        tool_calls = []

        for step_idx, step_tools in enumerate(ground_truth_tools):
            if not step_tools:
                continue

            # Primary tool (first in list)
            primary_tool = step_tools[0]
            tool_id = primary_tool.get("tool_id", "")
            tool_name = primary_tool.get("tool_name", "")
            description = (primary_tool.get("description", "") or "")[:100]
            dependencies = primary_tool.get("dependencies", [])

            # Alternative tools (rest of list)
            alternative_tools = []
            for alt in step_tools[1:]:
                alt_id = alt.get("tool_id")
                if alt_id:
                    alternative_tools.append(alt_id)

            # Create node
            node = {
                "node_id": f"n{step_idx}",
                "step_index": step_idx,
                "label": description,
                "step_type": "tool",
                "tool_id": tool_id,
                "alternative_tools": alternative_tools,
                "arguments": {},  # Delta doesn't specify argument values
                "output_vars": [f"<n{step_idx}>"],
                "needs_tool": True,
                "needs_new_tool": False,
            }
            nodes.append(node)

            # Create edges from dependencies
            for dep_idx in dependencies:
                edges.append({
                    "source": f"n{dep_idx}",
                    "target": f"n{step_idx}",
                    "edge_type": "data_dep",
                })

            # Create tool_call
            tool_call = {
                "call_index": len(tool_calls),
                "node_id": f"n{step_idx}",
                "tool_id": tool_id,
                "alternative_tools": alternative_tools,
                "arguments": [],  # No argument values in Delta
            }
            tool_calls.append(tool_call)

        # Determine plan type based on structure
        if len(nodes) <= 1:
            plan_type = "single"
        elif all(len(gt) == 1 for gt in ground_truth_tools):
            plan_type = "chain"
        else:
            plan_type = "dag"

        # Build unified record
        record = {
            "meta": {
                "id": str(query_id),
                "dataset": "delta",
                "subset": subset_name if not is_old_delta else "old_delta",
                "split": "test",
                "plan_type": plan_type,
                "difficulty": None,
                "expected_tool_count": expected_tool_count,
                "has_arguments": False,  # Delta doesn't have argument values
            },
            "query": {
                "user_query": query_text,
                "extra_instruction": None,
                "attachments": [],
            },
            "tool_environment": {
                "tools": flat_tools,
                "tool_graph": {"nodes": [], "edges": []},
            },
            "gold": {
                "plan_dag": {"nodes": nodes, "edges": edges},
                "tool_calls": tool_calls,
                "final_answer": {
                    "answer_type": "none",
                    "answer": None,
                    "aliases": [],
                    "tolerance": 0.0,
                },
            },
        }

        records.append(record)

    return records


# =============================================================================
# UltraTool Dataset Conversion
# =============================================================================

def convert_ultratool(ut_dir: Path, limit: Optional[int] = None) -> List[Dict[str, Any]]:
    """Convert UltraTool dataset files into unified records."""
    test_path = ut_dir / "test.json"
    tool_usage_path = ut_dir / "tool_usage.json"
    tool_usage_awareness_path = ut_dir / "tool_usage_awareness.json"
    tool_creation_awareness_path = ut_dir / "tool_creation_awareness.json"

    # Check if files exist
    required_files = [test_path, tool_usage_path, tool_usage_awareness_path, tool_creation_awareness_path]
    for f in required_files:
        if not f.exists():
            print(f"[WARN] Required file not found: {f}")
            return []

    test_iter = load_jsonl(test_path)
    usage_iter = load_jsonl(tool_usage_path)
    usage_aw_iter = load_jsonl(tool_usage_awareness_path)
    creation_aw_iter = load_jsonl(tool_creation_awareness_path)

    records: List[Dict[str, Any]] = []

    for idx, (test_row, usage_row, usage_aw_row, creation_aw_row) in enumerate(
        zip(test_iter, usage_iter, usage_aw_iter, creation_aw_iter)
    ):
        if limit is not None and idx >= limit:
            break

        question = test_row.get("question", "").strip()
        tools_raw = test_row.get("tools", [])
        plan_steps = test_row.get("plan", [])

        # Map step text -> index for later lookup
        step_to_idx: Dict[str, int] = {s.get("step", ""): i for i, s in enumerate(plan_steps)}

        # needs_tool from tool_usage_awareness
        needs_tool_map: Dict[int, bool] = {i: False for i in range(len(plan_steps))}
        for ref in usage_aw_row.get("reference", []):
            s = ref.get("step", "")
            flag = ref.get("tool", "0")
            if s in step_to_idx:
                needs_tool_map[step_to_idx[s]] = (flag == "1")

        # needs_new_tool from tool_creation_awareness
        needs_new_tool_map: Dict[int, bool] = {i: False for i in range(len(plan_steps))}
        for ref in creation_aw_row.get("reference", []):
            s = ref.get("step", "")
            if s in step_to_idx:
                needs_new_tool_map[step_to_idx[s]] = True

        # tool parameters from tool_usage
        params_by_step: Dict[str, Dict[str, Any]] = {}
        tool_name_by_step: Dict[str, str] = {}
        for ref in usage_row.get("reference", []):
            s = ref.get("step", "")
            tool_name_by_step[s] = ref.get("tool")
            params_by_step[s] = ref.get("param", {}) or {}

        # Build tool catalog
        tool_catalog = []
        for t in tools_raw:
            arg_schema = {}
            props = t.get("arguments", {}).get("properties", {})
            for arg_name, arg_info in props.items():
                arg_schema[arg_name] = {
                    "type": arg_info.get("type", "string"),
                    "description": arg_info.get("description"),
                    "required": True,
                }
            tool_catalog.append({
                "tool_id": t.get("name", ""),
                "description": t.get("description", ""),
                "input_type": [],
                "output_type": [],
                "arguments_schema": arg_schema,
            })

        # Build plan_dag nodes
        dag_nodes = []
        for i, step in enumerate(plan_steps):
            raw_step = step.get("step", "")
            label = raw_step
            node_id = f"n{i}"
            step_type = "thought"
            tool_id = None

            # Prefer explicit tool from params mapping
            explicit_tool = tool_name_by_step.get(raw_step)
            if explicit_tool:
                step_type = "tool"
                tool_id = explicit_tool
            else:
                tool_field = step.get("tool")
                if tool_field and tool_field not in ("null", "No tool required", ""):
                    step_type = "tool"
                    tool_id = str(tool_field).split("(")[0].strip()

            params = params_by_step.get(raw_step, {})
            dag_nodes.append({
                "node_id": node_id,
                "step_index": i,
                "label": label,
                "step_type": step_type,
                "tool_id": tool_id,
                "alternative_tools": [],
                "arguments": params,
                "output_vars": [f"<{node_id}>"],
                "needs_tool": needs_tool_map.get(i, None),
                "needs_new_tool": needs_new_tool_map.get(i, None),
            })

        # Simple sequential edges
        dag_edges = []
        for i in range(1, len(dag_nodes)):
            dag_edges.append({
                "source": dag_nodes[i - 1]["node_id"],
                "target": dag_nodes[i]["node_id"],
                "edge_type": "control_dep",
            })

        # Build tool_calls
        tool_calls = []
        for node in dag_nodes:
            if node["step_type"] != "tool" or not node["tool_id"]:
                continue
            step_text = node["label"]
            params = params_by_step.get(step_text, {})
            args_list = [{"name": k, "value": v} for k, v in params.items()]
            tool_calls.append({
                "call_index": len(tool_calls),
                "node_id": node["node_id"],
                "tool_id": node["tool_id"],
                "alternative_tools": [],
                "arguments": args_list,
            })

        record = {
            "meta": {
                "dataset": "ultratool",
                "subset": "test",
                "split": "test",
                "id": str(idx),
                "plan_type": "chain",
                "difficulty": None,
                "has_arguments": bool(tool_calls and any(tc.get("arguments") for tc in tool_calls)),
            },
            "query": {
                "user_query": question,
                "extra_instruction": None,
                "attachments": [],
            },
            "tool_environment": {
                "tools": tool_catalog,
                "tool_graph": {"nodes": [], "edges": []},
            },
            "gold": {
                "plan_dag": {"nodes": dag_nodes, "edges": dag_edges},
                "tool_calls": tool_calls,
                "final_answer": {
                    "answer_type": "none",
                    "answer": None,
                    "aliases": [],
                    "tolerance": 0.0,
                },
            },
        }
        records.append(record)

    return records


# =============================================================================
# TaskBench Dataset Conversion
# =============================================================================

def build_taskbench_tool_env(tb_dir: Path) -> Dict[str, Any]:
    """Build tool environment from TaskBench directory."""
    tool_desc_path = tb_dir / "tool_desc.json"
    graph_desc_path = tb_dir / "graph_desc.json"

    if not tool_desc_path.exists() or not graph_desc_path.exists():
        return {"tools": [], "tool_graph": {"nodes": [], "edges": []}}

    tool_desc = load_json(tool_desc_path)
    graph_desc = load_json(graph_desc_path)

    tools = []
    for n in tool_desc.get("nodes", []):
        tools.append({
            "tool_id": n.get("id", ""),
            "description": n.get("desc", ""),
            "input_type": n.get("input-type", []),
            "output_type": n.get("output-type", []),
            "arguments_schema": {},
        })

    tool_graph = {
        "nodes": [n.get("id", "") for n in graph_desc.get("nodes", [])],
        "edges": [
            {
                "source": link.get("source", ""),
                "target": link.get("target", ""),
                "edge_type": link.get("type", ""),
            }
            for link in graph_desc.get("links", [])
            if isinstance(link, dict) and "source" in link and "target" in link
        ],
    }

    return {"tools": tools, "tool_graph": tool_graph}


def convert_taskbench(
    tb_dir: Path,
    subset_name: str = "multimedia",
    limit: Optional[int] = None,
    use_alignment_ids: bool = True,
) -> List[Dict[str, Any]]:
    """Convert TaskBench split into unified records."""
    tool_env = build_taskbench_tool_env(tb_dir)

    # Load user requests
    user_requests: Dict[str, str] = {}
    user_requests_path = tb_dir / "user_requests.json"
    if not user_requests_path.exists():
        user_requests_path = tb_dir / "user_requests.jsonl"
    if user_requests_path.exists():
        for row in load_jsonl(user_requests_path):
            user_requests[row.get("id", "")] = row.get("user_request", "")

    # Alignment ids
    allowed_ids: Optional[Set[str]] = None
    alignment_path = tb_dir / "alignment_ids.json"
    if use_alignment_ids and alignment_path.exists():
        alignment = load_json(alignment_path)
        both = alignment.get("both_node_link_alignment_id", {})
        allowed_ids = set(
            both.get("single", []) +
            both.get("chain", []) +
            both.get("dag", [])
        )

    records: List[Dict[str, Any]] = []
    data_path = tb_dir / "data.json"

    if not data_path.exists():
        print(f"[WARN] data.json not found in {tb_dir}")
        return []

    for row in load_jsonl(data_path):
        sample_id = row.get("id", "")

        # Alignment filter
        if allowed_ids is not None and sample_id not in allowed_ids:
            continue

        # Only keep chain/dag types
        plan_type = row.get("type", "unknown")
        if plan_type not in ("chain", "dag"):
            continue

        if limit is not None and len(records) >= limit:
            break

        instruction = row.get("instruction") or row.get("user_request", "")
        user_query = user_requests.get(sample_id, instruction)

        # Parse tool_nodes (handles various formats)
        tool_nodes_raw = row.get("tool_nodes", [])
        if not tool_nodes_raw:
            tool_nodes_raw = row.get("task_nodes", [])
        if isinstance(tool_nodes_raw, str):
            try:
                tool_nodes_raw = json.loads(tool_nodes_raw)
            except json.JSONDecodeError:
                continue
        if isinstance(tool_nodes_raw, dict):
            tool_nodes_raw = [tool_nodes_raw]
        if not isinstance(tool_nodes_raw, list):
            continue

        parsed_nodes = []
        for j, item in enumerate(tool_nodes_raw):
            if isinstance(item, str):
                try:
                    item = json.loads(item)
                except json.JSONDecodeError:
                    continue
            if isinstance(item, dict) and "task" in item:
                parsed_nodes.append(item)

        if not parsed_nodes:
            continue

        tool_nodes_raw = parsed_nodes

        # Parse tool_links
        tool_links_raw = row.get("tool_links", [])
        if not tool_links_raw:
            tool_links_raw = row.get("task_links", [])
        if isinstance(tool_links_raw, str):
            try:
                tool_links_raw = json.loads(tool_links_raw)
            except json.JSONDecodeError:
                tool_links_raw = []
        if isinstance(tool_links_raw, dict):
            tool_links_raw = [tool_links_raw]
        if not isinstance(tool_links_raw, list):
            tool_links_raw = []

        parsed_links = []
        for j, item in enumerate(tool_links_raw):
            if isinstance(item, str):
                try:
                    item = json.loads(item)
                except json.JSONDecodeError:
                    continue
            if isinstance(item, dict) and "source" in item and "target" in item:
                parsed_links.append(item)

        tool_links_raw = parsed_links

        # Parse tool_steps
        tool_steps = row.get("tool_steps", [])
        if isinstance(tool_steps, str):
            try:
                tool_steps = json.loads(tool_steps)
            except json.JSONDecodeError:
                tool_steps = []

        # Build nodes
        name_to_nodeid: Dict[str, str] = {}
        dag_nodes = []
        for i, n in enumerate(tool_nodes_raw):
            node_id = f"n{i}"
            tool_name = n.get("task", "")
            args = n.get("arguments", [])
            name_to_nodeid[tool_name] = node_id

            label = tool_steps[i] if i < len(tool_steps) else tool_name

            # Handle different argument formats
            arguments: Dict[str, Any] = {}
            if isinstance(args, dict):
                arguments = args
            elif isinstance(args, list):
                for k, v in enumerate(args):
                    if isinstance(v, dict) and "name" in v:
                        arguments[v["name"]] = v.get("value")
                    else:
                        arguments[f"arg{k}"] = v
            else:
                arguments["arg0"] = args

            dag_nodes.append({
                "node_id": node_id,
                "step_index": i,
                "label": label,
                "step_type": "tool",
                "tool_id": tool_name,
                "alternative_tools": [],
                "arguments": arguments,
                "output_vars": [f"<{node_id}>"],
                "needs_tool": True,
                "needs_new_tool": False,
            })

        # Build edges
        dag_edges = []
        for link in tool_links_raw:
            src_name = link.get("source", "")
            tgt_name = link.get("target", "")
            src_id = name_to_nodeid.get(src_name)
            tgt_id = name_to_nodeid.get(tgt_name)
            if src_id and tgt_id:
                dag_edges.append({
                    "source": src_id,
                    "target": tgt_id,
                    "edge_type": "data_dep",
                })

        # Build tool calls
        tool_calls = []
        for i, node in enumerate(dag_nodes):
            args_dict = node.get("arguments", {})
            args_list = [{"name": k, "value": v} for k, v in args_dict.items()]
            tool_calls.append({
                "call_index": i,
                "node_id": node["node_id"],
                "tool_id": node["tool_id"],
                "alternative_tools": [],
                "arguments": args_list,
            })

        record = {
            "meta": {
                "dataset": "taskbench",
                "subset": subset_name,
                "split": "test",
                "id": sample_id,
                "plan_type": plan_type,
                "difficulty": None,
                "has_arguments": bool(tool_calls and any(tc.get("arguments") for tc in tool_calls)),
            },
            "query": {
                "user_query": user_query,
                "extra_instruction": instruction if instruction != user_query else None,
                "attachments": [],
            },
            "tool_environment": tool_env,
            "gold": {
                "plan_dag": {"nodes": dag_nodes, "edges": dag_edges},
                "tool_calls": tool_calls,
                "final_answer": {
                    "answer_type": "none",
                    "answer": None,
                    "aliases": [],
                    "tolerance": 0.0,
                },
            },
        }
        records.append(record)

    return records


# =============================================================================
# GAIA Dataset Conversion
# =============================================================================

def infer_answer_type(ans: Any) -> Tuple[str, Any]:
    """Infer answer type from GAIA final_answer."""
    if ans is None:
        return "none", None
    s = str(ans).strip()
    try:
        if "." in s:
            return "number", float(s)
        else:
            return "number", int(s)
    except ValueError:
        pass
    return "string", s


def convert_gaia(gaia_path: Path, limit: Optional[int] = None) -> List[Dict[str, Any]]:
    """Convert GAIA dataset to unified format."""
    records: List[Dict[str, Any]] = []

    for idx, row in enumerate(load_jsonl(gaia_path)):
        if limit is not None and idx >= limit:
            break

        ans_type, ans_val = infer_answer_type(row.get("final_answer"))

        record = {
            "meta": {
                "dataset": "gaia",
                "subset": "val_2023",
                "split": "val",
                "id": row.get("task_id", str(idx)),
                "plan_type": "unknown",
                "difficulty": row.get("level"),
                "has_arguments": False,
            },
            "query": {
                "user_query": row.get("input", ""),
                "extra_instruction": None,
                "attachments": [],
            },
            "tool_environment": {
                "tools": [],
                "tool_graph": {"nodes": [], "edges": []},
            },
            "gold": {
                "plan_dag": {"nodes": [], "edges": []},
                "tool_calls": [],
                "final_answer": {
                    "answer_type": ans_type,
                    "answer": ans_val,
                    "aliases": [],
                    "tolerance": 0.0,
                },
            },
        }
        records.append(record)

    return records


# =============================================================================
# Main CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Unified Dataset Converter for LLM Planning Evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Modes:
  delta_new   - Convert new Delta dataset (17 MCP tools)
  delta_old   - Convert old Delta dataset (2397 tools)
  ultratool   - Convert UltraTool dataset
  taskbench   - Convert TaskBench dataset(s)
  gaia        - Convert GAIA dataset
  all         - Convert all available datasets

Examples:
  # Convert new delta
  python convert_all_datasets.py --mode delta_new \\
      --queries data/delta_new/new_queries_delta.json \\
      --tools data/delta_new/new_tools_delta.json \\
      --output data/unified_delta_new.jsonl

  # Convert with limit
  python convert_all_datasets.py --mode ultratool \\
      --ultratool_dir data/UltraTool/test_set \\
      --output data/unified_ultratool_50.jsonl \\
      --limit 50
        """
    )

    # Mode selection
    parser.add_argument("--mode", type=str, required=True,
                        choices=["delta_new", "delta_old", "ultratool", "taskbench", "gaia", "all"],
                        help="Conversion mode / dataset type")

    # Delta arguments
    parser.add_argument("--queries", type=Path,
                        help="Path to Delta queries JSON file")
    parser.add_argument("--tools", type=Path,
                        help="Path to Delta tools JSON file")
    parser.add_argument("--delta_subset", type=str, default="mcp_tools",
                        help="Subset name for Delta dataset metadata")

    # UltraTool arguments
    parser.add_argument("--ultratool_dir", type=Path,
                        help="Directory containing UltraTool test files")

    # TaskBench arguments
    parser.add_argument("--taskbench_dirs", type=Path, nargs="+",
                        help="TaskBench directories (multiple allowed)")
    parser.add_argument("--no_alignment_filter", action="store_true",
                        help="Disable alignment_ids.json filtering for TaskBench")

    # GAIA arguments
    parser.add_argument("--gaia_path", type=Path,
                        help="Path to GAIA jsonl file")

    # Common arguments
    parser.add_argument("--output", type=Path, required=True,
                        help="Output JSONL path")
    parser.add_argument("--limit", type=int, default=None,
                        help="Maximum samples per dataset (for testing)")
    parser.add_argument("--verbose", action="store_true",
                        help="Print detailed progress")

    args = parser.parse_args()

    all_records: List[Dict[str, Any]] = []

    # Delta new
    if args.mode in ("delta_new", "all"):
        if args.queries and args.tools:
            print(f"[INFO] Converting Delta (new) from: {args.queries}")
            records = convert_delta(
                queries_path=args.queries,
                tools_path=args.tools,
                subset_name=args.delta_subset,
                limit=args.limit,
                is_old_delta=False,
            )
            print(f"[INFO]   -> {len(records)} records")
            all_records.extend(records)
        elif args.mode == "delta_new":
            print("[ERROR] --queries and --tools required for delta_new mode")
            return

    # Delta old
    if args.mode == "delta_old":
        if args.queries and args.tools:
            print(f"[INFO] Converting Delta (old) from: {args.queries}")
            records = convert_delta(
                queries_path=args.queries,
                tools_path=args.tools,
                subset_name="old_delta",
                limit=args.limit,
                is_old_delta=True,
            )
            print(f"[INFO]   -> {len(records)} records")
            all_records.extend(records)
        else:
            print("[ERROR] --queries and --tools required for delta_old mode")
            return

    # UltraTool
    if args.mode in ("ultratool", "all"):
        if args.ultratool_dir:
            print(f"[INFO] Converting UltraTool from: {args.ultratool_dir}")
            records = convert_ultratool(args.ultratool_dir, limit=args.limit)
            print(f"[INFO]   -> {len(records)} records")
            all_records.extend(records)
        elif args.mode == "ultratool":
            print("[ERROR] --ultratool_dir required for ultratool mode")
            return

    # TaskBench
    if args.mode in ("taskbench", "all"):
        if args.taskbench_dirs:
            for tb_dir in args.taskbench_dirs:
                subset_name = tb_dir.name.replace("data_", "")
                print(f"[INFO] Converting TaskBench/{subset_name} from: {tb_dir}")
                records = convert_taskbench(
                    tb_dir,
                    subset_name=subset_name,
                    limit=args.limit,
                    use_alignment_ids=not args.no_alignment_filter,
                )
                print(f"[INFO]   -> {len(records)} records")
                all_records.extend(records)
        elif args.mode == "taskbench":
            print("[ERROR] --taskbench_dirs required for taskbench mode")
            return

    # GAIA
    if args.mode in ("gaia", "all"):
        if args.gaia_path:
            print(f"[INFO] Converting GAIA from: {args.gaia_path}")
            records = convert_gaia(args.gaia_path, limit=args.limit)
            print(f"[INFO]   -> {len(records)} records")
            all_records.extend(records)
        elif args.mode == "gaia":
            print("[ERROR] --gaia_path required for gaia mode")
            return

    # Write output
    if not all_records:
        print("[WARN] No records to write!")
        return

    count = write_jsonl(all_records, args.output)
    print(f"\n[DONE] Written {count} total records to: {args.output}")

    # Print summary
    dataset_counts = Counter(r["meta"]["dataset"] for r in all_records)
    subset_counts = Counter(f"{r['meta']['dataset']}/{r['meta']['subset']}" for r in all_records)

    print(f"\nSummary by dataset:")
    for ds, cnt in sorted(dataset_counts.items()):
        print(f"  {ds}: {cnt}")

    print(f"\nSummary by subset:")
    for subset, cnt in sorted(subset_counts.items()):
        print(f"  {subset}: {cnt}")

    # Print argument stats
    has_args_count = sum(1 for r in all_records if r["meta"].get("has_arguments"))
    print(f"\nRecords with arguments: {has_args_count}/{len(all_records)}")


if __name__ == "__main__":
    main()
