from __future__ import annotations

import argparse
import csv
import difflib
import hashlib
import pickle
import sys
from functools import lru_cache
from pathlib import Path

import networkx as nx
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
PYLINGUAL_ROOT = REPO_ROOT / "pylingual"
DISTANCE_CACHE_DIR = REPO_ROOT / "results" / "cache" / "pyc_distance"
CACHE_FORMAT_VERSION = 1

if str(PYLINGUAL_ROOT) not in sys.path:
    sys.path.insert(0, str(PYLINGUAL_ROOT))

from utils.file_helpers import fetch_pyllmpatch_pyc_paths


CONTROL_FLOW_WEIGHT = 3
INSTRUCTION_GAP_COST = 2
CFG_BLOCK_NODE_COST = 3
CFG_EDGE_COST = 1
CFG_META_NODE_COST = 100
EXTRA_CODE_OBJECT_BASE_PENALTY = 2
MISSING_CODE_OBJECT_BASE_PENALTY = 4


def _load_bytecode_dependencies():
    import xdis.opcodes
    from xdis.load import load_module
    from pylingual.control_flow_reconstruction.cfg import CFG
    from pylingual.editable_bytecode import EditableBytecode, Inst
    from pylingual.editable_bytecode.bytecode_patches import (
        fix_indirect_jump,
        fix_unreachable,
        remove_extended_arg,
        remove_nop,
        replace_firstlno,
    )
    from pylingual.editable_bytecode.control_flow_graph import bytecode_to_control_flow_graph
    from pylingual.equivalence_check import is_control_flow_equivalent, matching_iter

    return {
        "xdis_opcodes": xdis.opcodes,
        "load_module": load_module,
        "CFG": CFG,
        "EditableBytecode": EditableBytecode,
        "Inst": Inst,
        "bytecode_to_control_flow_graph": bytecode_to_control_flow_graph,
        "is_control_flow_equivalent": is_control_flow_equivalent,
        "matching_iter": matching_iter,
        "patches": [remove_extended_arg, remove_nop, fix_indirect_jump, fix_unreachable, remove_extended_arg, replace_firstlno],
    }


_load_bytecode_dependencies = lru_cache(maxsize=1)(_load_bytecode_dependencies)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Measure distances between corresponding code objects in two .pyc files using pylingual."
    )
    parser.add_argument("pyc_a", type=Path, help="Path to the first .pyc file")
    parser.add_argument("pyc_b", type=Path, help="Path to the second .pyc file")
    parser.add_argument(
        "--file-hash",
        type=str,
        default=None,
        help="Optional dataset file hash to include in the summary CSV.",
    )
    parser.add_argument(
        "--csv-out",
        type=Path,
        default=None,
        help="Optional path to write the per-code-object results as CSV",
    )
    return parser


def validate_input(path: Path) -> Path:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"File does not exist: {path}")
    if path.suffix != ".pyc":
        raise ValueError(f"Expected a .pyc file: {path}")
    return path


def load_editable_bytecode_from_pyc(path: Path):
    deps = _load_bytecode_dependencies()
    source_tuple = deps["load_module"](str(path))
    if len(source_tuple) < 4:
        raise ValueError(f"Unexpected xdis load result with {len(source_tuple)} fields")

    version = source_tuple[0]
    code = source_tuple[3]
    opcode = getattr(deps["xdis_opcodes"], f"opcode_{version[0]}{version[1]}")

    bytecode = deps["EditableBytecode"](code, opcode, version)
    bytecode.apply_patches(deps["patches"])
    return bytecode


def normalize_argval(inst):
    argval = getattr(inst, "argval", None)

    if hasattr(argval, "co_name"):
        return ("code", getattr(argval, "co_name", None))
    if inst.is_jump:
        return ("jump", inst.opname)
    if isinstance(argval, (str, int, float, bytes, type(None), bool)):
        return argval
    return repr(argval)


def instruction_signature(inst):
    return (
        inst.opname,
        inst.opcode,
        inst.optype,
        inst.real_size,
        inst.has_arg,
        inst.has_extended_arg,
        inst.is_jump_target,
        normalize_argval(inst),
    )


def instruction_edit_cost(inst_a, inst_b) -> int:
    """Penalties: exact=0, opcode-only/arg-only match=1, full mismatch=2."""
    if inst_a == inst_b:
        return 0

    opcode_matches = inst_a[1] == inst_b[1]
    arg_matches = inst_a[-1] == inst_b[-1]

    if opcode_matches ^ arg_matches:
        return 1
    return 2


def levenshtein_distance(seq_a, seq_b) -> int:
    """Weighted edit distance: insert/delete=2, exact=0, partial substitution=1, full substitution=2."""
    if len(seq_a) < len(seq_b):
        seq_a, seq_b = seq_b, seq_a

    previous = [j * INSTRUCTION_GAP_COST for j in range(len(seq_b) + 1)]
    for i, item_a in enumerate(seq_a, start=1):
        current = [i * INSTRUCTION_GAP_COST]
        for j, item_b in enumerate(seq_b, start=1):
            substitution_cost = instruction_edit_cost(item_a, item_b)
            current.append(
                min(
                    previous[j] + INSTRUCTION_GAP_COST,
                    current[j - 1] + INSTRUCTION_GAP_COST,
                    previous[j - 1] + substitution_cost,
                )
            )
        previous = current
    return previous[-1]


def build_block_graph(bytecode):
    deps = _load_bytecode_dependencies()
    cfg = deps["bytecode_to_control_flow_graph"](bytecode)
    return deps["CFG"].from_graph(cfg, bytecode, iterate=False)


def cfg_complexity_basis(block_graph) -> int:
    block_nodes = [
        node
        for node, attrs in block_graph.nodes(data=True)
        if attrs.get("category", _node_category(node)) == "block"
    ]
    block_edge_count = sum(
        1
        for src, dst in block_graph.edges
        if block_graph.nodes[src].get("category", _node_category(src)) == "block"
        and block_graph.nodes[dst].get("category", _node_category(dst)) == "block"
    )
    return max(len(block_nodes) + block_edge_count, 1)


def _node_category(node) -> str:
    name = type(node).__name__
    if name == "MetaTemplate":
        meta_name = getattr(node, "name", "")
        if meta_name == "start":
            return "start"
        if meta_name == "end":
            return "end"
        return "meta"
    return "block"


def _node_offset_rank_map(graph) -> dict:
    ordered = sorted(
        graph.nodes(data="offset", default=(float("inf"),)),
        key=lambda item: min(item[1]),
    )
    return {node: rank for rank, (node, _) in enumerate(ordered)}


def _opcode_sequence_similarity(seq_a, seq_b) -> float:
    if not seq_a and not seq_b:
        return 1.0
    return difflib.SequenceMatcher(a=seq_a, b=seq_b).ratio()


def _annotate_cfg_for_edit_distance(graph: nx.DiGraph) -> nx.DiGraph:
    annotated = nx.DiGraph()
    rank_map = _node_offset_rank_map(graph)

    for node in graph.nodes:
        instructions = list(node.get_instructions()) if hasattr(node, "get_instructions") else []
        opcode_seq = tuple(getattr(inst, "opname", "") for inst in instructions)
        category = _node_category(node)
        annotated.add_node(
            node,
            category=category,
            opcode_seq=opcode_seq,
            inst_count=len(opcode_seq),
            out_degree=int(graph.out_degree(node)),
            starts_with=opcode_seq[0] if opcode_seq else None,
            ends_with=opcode_seq[-1] if opcode_seq else None,
            rank=rank_map[node],
        )

    for src, dst, edge_data in graph.edges(data=True):
        kind = edge_data.get("kind")
        kind_value = getattr(kind, "value", str(kind))
        annotated.add_edge(src, dst, kind=kind_value)

    return annotated


def _serialize_block_graph(graph: nx.DiGraph) -> nx.DiGraph:
    serialized = nx.DiGraph()
    node_ids = {node: idx for idx, node in enumerate(graph.nodes)}

    for node, attrs in graph.nodes(data=True):
        instructions = list(node.get_instructions()) if hasattr(node, "get_instructions") else []
        opcode_seq = tuple(getattr(inst, "opname", "") for inst in instructions)
        offset = attrs.get("offset", (float("inf"),))
        serialized.add_node(
            node_ids[node],
            offset=tuple(offset),
            category=_node_category(node),
            opcode_seq=opcode_seq,
            inst_count=len(opcode_seq),
            starts_with=opcode_seq[0] if opcode_seq else None,
            ends_with=opcode_seq[-1] if opcode_seq else None,
        )

    for src, dst, edge_data in graph.edges(data=True):
        kind = edge_data.get("kind")
        serialized.add_edge(
            node_ids[src],
            node_ids[dst],
            kind=getattr(kind, "value", str(kind)),
        )

    return serialized


def _annotate_serialized_cfg_for_edit_distance(graph: nx.DiGraph) -> nx.DiGraph:
    annotated = nx.DiGraph()
    rank_map = _node_offset_rank_map(graph)

    for node, attrs in graph.nodes(data=True):
        annotated.add_node(
            node,
            category=attrs["category"],
            opcode_seq=attrs["opcode_seq"],
            inst_count=attrs["inst_count"],
            out_degree=int(graph.out_degree(node)),
            starts_with=attrs["starts_with"],
            ends_with=attrs["ends_with"],
            rank=rank_map[node],
        )

    for src, dst, edge_data in graph.edges(data=True):
        annotated.add_edge(src, dst, kind=edge_data.get("kind"))

    return annotated


def _cfg_node_subst_cost(attrs_a: dict, attrs_b: dict) -> float:
    category_a = attrs_a["category"]
    category_b = attrs_b["category"]

    if category_a != category_b:
        return CFG_META_NODE_COST

    if category_a in {"start", "end", "meta"}:
        return 0.0

    if attrs_a["opcode_seq"] == attrs_b["opcode_seq"] and attrs_a["out_degree"] == attrs_b["out_degree"]:
        return 0.0

    cost = 0
    if attrs_a["starts_with"] != attrs_b["starts_with"]:
        cost += 1
    if attrs_a["ends_with"] != attrs_b["ends_with"]:
        cost += 1
    if abs(attrs_a["inst_count"] - attrs_b["inst_count"]) >= 2:
        cost += 1
    if attrs_a["out_degree"] != attrs_b["out_degree"]:
        cost += 1
    if _opcode_sequence_similarity(attrs_a["opcode_seq"], attrs_b["opcode_seq"]) < 0.6:
        cost += 1
    if abs(attrs_a["rank"] - attrs_b["rank"]) > 1:
        cost += 1

    return float(min(cost, CFG_BLOCK_NODE_COST))


def _cfg_node_del_cost(attrs: dict) -> float:
    return float(CFG_META_NODE_COST if attrs["category"] in {"start", "end", "meta"} else CFG_BLOCK_NODE_COST)


def _cfg_node_ins_cost(attrs: dict) -> float:
    return float(CFG_META_NODE_COST if attrs["category"] in {"start", "end", "meta"} else CFG_BLOCK_NODE_COST)


def _cfg_edge_subst_cost(attrs_a: dict, attrs_b: dict) -> float:
    return 0.0 if attrs_a.get("kind") == attrs_b.get("kind") else float(CFG_EDGE_COST)


def _cfg_edge_del_cost(_attrs: dict) -> float:
    return float(CFG_EDGE_COST)


def _cfg_edge_ins_cost(_attrs: dict) -> float:
    return float(CFG_EDGE_COST)


def _block_node_ids(graph: nx.DiGraph) -> list[int]:
    return [node for node, attrs in graph.nodes(data=True) if attrs["category"] == "block"]


def _edge_kind_histogram(graph: nx.DiGraph) -> dict[str, int]:
    histogram: dict[str, int] = {}
    for src, dst, edge_data in graph.edges(data=True):
        if graph.nodes[src]["category"] != "block" or graph.nodes[dst]["category"] != "block":
            continue
        kind = edge_data.get("kind", "unknown")
        histogram[kind] = histogram.get(kind, 0) + 1
    return histogram


def _successor_mismatch_count(graph_a: nx.DiGraph, graph_b: nx.DiGraph) -> int:
    ordered_blocks_a = sorted(
        (node for node in graph_a.nodes if graph_a.nodes[node]["category"] == "block"),
        key=lambda node: min(graph_a.nodes[node]["offset"]),
    )
    ordered_blocks_b = sorted(
        (node for node in graph_b.nodes if graph_b.nodes[node]["category"] == "block"),
        key=lambda node: min(graph_b.nodes[node]["offset"]),
    )

    node_id_mapping = {
        node_a: node_b
        for node_a, node_b in zip(ordered_blocks_a, ordered_blocks_b)
    }

    mismatch_count = 0
    for node_a, node_b in node_id_mapping.items():
        mapped_successors_a = {
            node_id_mapping[dest]
            for dest in graph_a.successors(node_a)
            if dest in node_id_mapping and graph_a.nodes[dest]["category"] == "block"
        }
        successors_b = {
            dest
            for dest in graph_b.successors(node_b)
            if graph_b.nodes[dest]["category"] == "block"
        }
        mismatch_count += len(mapped_successors_a.symmetric_difference(successors_b))

    return mismatch_count


def _heuristic_control_flow_distance(block_graph_a: nx.DiGraph, block_graph_b: nx.DiGraph) -> int:
    block_nodes_a = _block_node_ids(block_graph_a)
    block_nodes_b = _block_node_ids(block_graph_b)

    block_count_diff = abs(len(block_nodes_a) - len(block_nodes_b))
    edge_count_diff = abs(
        sum(1 for src, dst in block_graph_a.edges if block_graph_a.nodes[src]["category"] == "block" and block_graph_a.nodes[dst]["category"] == "block")
        - sum(1 for src, dst in block_graph_b.edges if block_graph_b.nodes[src]["category"] == "block" and block_graph_b.nodes[dst]["category"] == "block")
    )
    branch_count_diff = abs(
        sum(1 for node in block_nodes_a if block_graph_a.out_degree(node) > 1)
        - sum(1 for node in block_nodes_b if block_graph_b.out_degree(node) > 1)
    )

    edge_kind_hist_a = _edge_kind_histogram(block_graph_a)
    edge_kind_hist_b = _edge_kind_histogram(block_graph_b)
    edge_kind_diff = sum(
        abs(edge_kind_hist_a.get(kind, 0) - edge_kind_hist_b.get(kind, 0))
        for kind in set(edge_kind_hist_a) | set(edge_kind_hist_b)
    )

    successor_mismatch = _successor_mismatch_count(block_graph_a, block_graph_b)

    return block_count_diff + edge_count_diff + branch_count_diff + edge_kind_diff + successor_mismatch


def _ged_control_flow_distance_from_graphs(block_graph_a, block_graph_b, annotated_a=None, annotated_b=None) -> int:
    if _is_control_flow_equivalent_serialized(block_graph_a, block_graph_b):
        return 0

    annotated_a = annotated_a or _annotate_serialized_cfg_for_edit_distance(block_graph_a)
    annotated_b = annotated_b or _annotate_serialized_cfg_for_edit_distance(block_graph_b)
    distance = nx.algorithms.similarity.graph_edit_distance(
        annotated_a,
        annotated_b,
        node_subst_cost=_cfg_node_subst_cost,
        node_del_cost=_cfg_node_del_cost,
        node_ins_cost=_cfg_node_ins_cost,
        edge_subst_cost=_cfg_edge_subst_cost,
        edge_del_cost=_cfg_edge_del_cost,
        edge_ins_cost=_cfg_edge_ins_cost,
    )

    if distance is None:
        return abs(len(block_graph_a) - len(block_graph_b)) + abs(block_graph_a.number_of_edges() - block_graph_b.number_of_edges())

    return int(round(distance))


def _is_control_flow_equivalent_serialized(graph_a: nx.DiGraph, graph_b: nx.DiGraph) -> bool:
    if len(graph_a) != len(graph_b):
        return False

    ordered_nodes_a = sorted(graph_a.nodes(data="offset", default=(float("inf"),)), key=lambda item: min(item[1]))
    ordered_nodes_b = sorted(graph_b.nodes(data="offset", default=(float("inf"),)), key=lambda item: min(item[1]))
    node_id_mapping = {node_a: node_b for (node_a, _), (node_b, _) in zip(ordered_nodes_a, ordered_nodes_b)}

    for node_a, edges_a in graph_a.adjacency():
        node_b = node_id_mapping[node_a]
        mapped_destinations_a = {node_id_mapping[dest] for dest in edges_a.keys()}
        destinations_b = set(graph_b[node_b].keys())
        if mapped_destinations_a != destinations_b:
            return False

    return True


def _control_flow_distance_from_graphs(block_graph_a, block_graph_b, annotated_a=None, annotated_b=None) -> int:
    if _is_control_flow_equivalent_serialized(block_graph_a, block_graph_b):
        return 0

    return _heuristic_control_flow_distance(block_graph_a, block_graph_b)

    # GED is kept here for later experimentation, but it is disabled in the active
    # path because it dominates runtime on dataset-scale runs.
    #
    # return _ged_control_flow_distance_from_graphs(block_graph_a, block_graph_b, annotated_a, annotated_b)


def control_flow_distance(bc_a, bc_b, block_graph_a=None, block_graph_b=None) -> int:
    block_graph_a = block_graph_a or build_block_graph(bc_a)
    block_graph_b = block_graph_b or build_block_graph(bc_b)
    return _control_flow_distance_from_graphs(block_graph_a, block_graph_b)


def unmatched_code_object_penalty(bytecode, kind: str) -> tuple[int, int, int, dict]:
    block_graph = build_block_graph(bytecode)
    block_nodes = [node for node in block_graph.nodes if _node_category(node) == "block"]
    block_count = len(block_nodes)
    branch_count = sum(1 for node in block_nodes if block_graph.out_degree(node) > 1)
    try:
        block_graph._create_dominator_tree()
        loop_header_count = sum(1 for node in block_nodes if block_graph.is_loop_header(node))
    except Exception:
        loop_header_count = 0
    exception_edge_count = sum(
        1
        for src, dst, edge_data in block_graph.edges(data=True)
        if _node_category(src) == "block"
        and _node_category(dst) == "block"
        and getattr(edge_data.get("kind"), "value", str(edge_data.get("kind"))) == "exception"
    )

    instruction_distance = len(bytecode) * INSTRUCTION_GAP_COST
    cfg_distance = block_count + branch_count + loop_header_count + exception_edge_count
    unmatched_penalty = EXTRA_CODE_OBJECT_BASE_PENALTY if kind == "extra" else MISSING_CODE_OBJECT_BASE_PENALTY

    return instruction_distance, cfg_distance, unmatched_penalty, {
        "block_count": block_count,
        "branch_count": branch_count,
        "loop_header_count": loop_header_count,
        "exception_edge_count": exception_edge_count,
        "cfg_complexity_basis": cfg_complexity_basis(block_graph),
    }


def combined_distance(instruction_distance: int, cfg_distance: int, unmatched_penalty: int = 0) -> tuple[int, int]:
    """Combined score = instruction + 3*cfg + dynamic interaction + unmatched penalty."""
    interaction_penalty = min(instruction_distance, cfg_distance, CONTROL_FLOW_WEIGHT) if instruction_distance > 0 and cfg_distance > 0 else 0
    score = instruction_distance + (CONTROL_FLOW_WEIGHT * cfg_distance) + interaction_penalty + unmatched_penalty
    return score, interaction_penalty


def _cache_file_for_pyc(path: Path, cache_dir: Path | None = None) -> Path:
    cache_dir = (cache_dir or DISTANCE_CACHE_DIR).expanduser().resolve()
    digest = hashlib.sha256(str(path).encode("utf-8")).hexdigest()
    return cache_dir / f"{digest}.pkl"


def _serialize_code_object(bytecode) -> dict:
    block_graph = build_block_graph(bytecode)
    serialized_block_graph = _serialize_block_graph(block_graph)
    annotated_block_graph = _annotate_serialized_cfg_for_edit_distance(serialized_block_graph)
    block_nodes = [node for node, attrs in serialized_block_graph.nodes(data=True) if attrs["category"] == "block"]
    branch_count = sum(1 for node in block_nodes if serialized_block_graph.out_degree(node) > 1)
    try:
        block_graph._create_dominator_tree()
        loop_header_count = sum(1 for node in block_graph.nodes if _node_category(node) == "block" and block_graph.is_loop_header(node))
    except Exception:
        loop_header_count = 0
    exception_edge_count = sum(
        1
        for src, dst, edge_data in serialized_block_graph.edges(data=True)
        if serialized_block_graph.nodes[src]["category"] == "block"
        and serialized_block_graph.nodes[dst]["category"] == "block"
        and edge_data.get("kind") == "exception"
    )
    unmatched_instruction_distance = len(bytecode) * INSTRUCTION_GAP_COST
    unmatched_cfg_distance = len(block_nodes) + branch_count + loop_header_count + exception_edge_count
    unmatched_metrics = {
        "block_count": len(block_nodes),
        "branch_count": branch_count,
        "loop_header_count": loop_header_count,
        "exception_edge_count": exception_edge_count,
        "cfg_complexity_basis": cfg_complexity_basis(serialized_block_graph),
    }

    return {
        "name": bytecode.name,
        "instruction_signatures": [instruction_signature(inst) for inst in bytecode],
        "inst_count": len(bytecode),
        "block_graph": serialized_block_graph,
        "annotated_block_graph": annotated_block_graph,
        "cfg_complexity_basis": unmatched_metrics["cfg_complexity_basis"],
        "unmatched_instruction_distance": unmatched_instruction_distance,
        "unmatched_cfg_distance": unmatched_cfg_distance,
        "unmatched_metrics": unmatched_metrics,
    }


def _build_prepared_pyc(path: Path) -> dict:
    prepared_root = load_editable_bytecode_from_pyc(path)
    code_objects = [_serialize_code_object(bytecode) for bytecode in prepared_root.iter_bytecodes()]
    stat = path.stat()
    return {
        "cache_format_version": CACHE_FORMAT_VERSION,
        "source_path": str(path),
        "source_size": stat.st_size,
        "source_mtime_ns": stat.st_mtime_ns,
        "code_objects": code_objects,
    }


def prepare_pyc(path: Path, cache_dir: Path | None = None) -> dict:
    path = validate_input(path)
    cache_file = _cache_file_for_pyc(path, cache_dir=cache_dir)
    cache_file.parent.mkdir(parents=True, exist_ok=True)

    stat = path.stat()
    if cache_file.exists():
        try:
            with cache_file.open("rb") as handle:
                cached = pickle.load(handle)
            if (
                cached.get("cache_format_version") == CACHE_FORMAT_VERSION
                and cached.get("source_path") == str(path)
                and cached.get("source_size") == stat.st_size
                and cached.get("source_mtime_ns") == stat.st_mtime_ns
            ):
                return cached
        except Exception:
            pass

    prepared = _build_prepared_pyc(path)
    with cache_file.open("wb") as handle:
        pickle.dump(prepared, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return prepared


def _iter_matched_code_objects(prepared_gt: dict, prepared_derived: dict):
    gt_code_objects = prepared_gt["code_objects"]
    derived_code_objects = prepared_derived["code_objects"]
    sm = difflib.SequenceMatcher(
        a=[obj["name"] for obj in gt_code_objects],
        b=[obj["name"] for obj in derived_code_objects],
    )
    i_gt = 0
    i_derived = 0
    for block in sm.get_matching_blocks():
        while i_gt < block.a:
            yield gt_code_objects[i_gt], None
            i_gt += 1
        while i_derived < block.b:
            yield None, derived_code_objects[i_derived]
            i_derived += 1
        for i in range(block.size):
            yield gt_code_objects[i_gt + i], derived_code_objects[i_derived + i]
        i_gt += block.size
        i_derived += block.size
    while i_gt < len(gt_code_objects):
        yield gt_code_objects[i_gt], None
        i_gt += 1
    while i_derived < len(derived_code_objects):
        yield None, derived_code_objects[i_derived]
        i_derived += 1


def compare_code_object_distances(gt_pyc: Path, derived_pyc: Path):
    prepared_gt = prepare_pyc(gt_pyc) if isinstance(gt_pyc, Path) else gt_pyc
    prepared_derived = prepare_pyc(derived_pyc) if isinstance(derived_pyc, Path) else derived_pyc

    results = []
    gt_code_objects = len(prepared_gt["code_objects"])
    derived_code_objects = len(prepared_derived["code_objects"])

    for gt_bc, derived_bc in _iter_matched_code_objects(prepared_gt, prepared_derived):
        gt_name = gt_bc["name"] if gt_bc is not None else None
        derived_name = derived_bc["name"] if derived_bc is not None else None
        gt_len = gt_bc["inst_count"] if gt_bc is not None else 0
        derived_len = derived_bc["inst_count"] if derived_bc is not None else 0

        if gt_bc is None:
            instruction_distance = derived_bc["unmatched_instruction_distance"]
            cfg_distance = derived_bc["unmatched_cfg_distance"]
            unmatched_penalty = EXTRA_CODE_OBJECT_BASE_PENALTY
            unmatched_metrics = derived_bc["unmatched_metrics"]
            score, interaction_penalty = combined_distance(instruction_distance, cfg_distance, unmatched_penalty)
            instruction_basis = max(derived_len, 1)
            cfg_basis = max(unmatched_metrics["cfg_complexity_basis"], 1)
            status = "extra"
        elif derived_bc is None:
            instruction_distance = gt_bc["unmatched_instruction_distance"]
            cfg_distance = gt_bc["unmatched_cfg_distance"]
            unmatched_penalty = MISSING_CODE_OBJECT_BASE_PENALTY
            unmatched_metrics = gt_bc["unmatched_metrics"]
            score, interaction_penalty = combined_distance(instruction_distance, cfg_distance, unmatched_penalty)
            instruction_basis = max(gt_len, 1)
            cfg_basis = max(unmatched_metrics["cfg_complexity_basis"], 1)
            status = "missing"
        else:
            gt_seq = gt_bc["instruction_signatures"]
            derived_seq = derived_bc["instruction_signatures"]
            instruction_distance = levenshtein_distance(gt_seq, derived_seq)
            gt_block_graph = gt_bc["block_graph"]
            derived_block_graph = derived_bc["block_graph"]
            cfg_distance = _control_flow_distance_from_graphs(
                gt_block_graph,
                derived_block_graph,
                gt_bc["annotated_block_graph"],
                derived_bc["annotated_block_graph"],
            )
            unmatched_penalty = 0
            unmatched_metrics = None
            score, interaction_penalty = combined_distance(instruction_distance, cfg_distance)
            instruction_basis = max(gt_len, derived_len, 1)
            cfg_basis = max(gt_bc["cfg_complexity_basis"], derived_bc["cfg_complexity_basis"], 1)
            status = "matched"

        normalized_instruction_distance = instruction_distance / max(instruction_basis, 1)
        normalized_cfg_distance = cfg_distance / max(cfg_basis, 1)
        normalized_interaction_penalty = interaction_penalty / max(min(instruction_basis, cfg_basis), 1)
        normalized_unmatched_penalty = unmatched_penalty / max(instruction_basis + cfg_basis, 1)
        normalized_combined_distance = (
            normalized_instruction_distance
            + (CONTROL_FLOW_WEIGHT * normalized_cfg_distance)
            + normalized_interaction_penalty
            + normalized_unmatched_penalty
        )

        results.append(
            {
                "code_object": gt_name if gt_name == derived_name else f"{gt_name or 'None'} <-> {derived_name or 'None'}",
                "gt_name": gt_name,
                "derived_name": derived_name,
                "status": status,
                "gt_inst_count": gt_len,
                "derived_inst_count": derived_len,
                "instruction_distance": instruction_distance,
                "control_flow_distance": cfg_distance,
                "control_flow_weight": CONTROL_FLOW_WEIGHT,
                "interaction_penalty": interaction_penalty,
                "unmatched_penalty": unmatched_penalty,
                "unmatched_metrics": unmatched_metrics,
                "combined_distance": score,
                "instruction_basis": instruction_basis,
                "cfg_basis": cfg_basis,
                "normalized_instruction_distance": normalized_instruction_distance,
                "normalized_cfg_distance": normalized_cfg_distance,
                "normalized_interaction_penalty": normalized_interaction_penalty,
                "normalized_unmatched_penalty": normalized_unmatched_penalty,
                "normalized_combined_distance": normalized_combined_distance,
            }
        )

    setattr(compare_code_object_distances, "_last_code_object_counts", (gt_code_objects, derived_code_objects))
    return results


def summarize_results(results):
    gt_code_objects, derived_code_objects = getattr(compare_code_object_distances, "_last_code_object_counts", (None, None))
    total_combined_distance = sum(result["combined_distance"] for result in results)
    total_instruction_distance = sum(result["instruction_distance"] for result in results)
    total_control_flow_distance = sum(result["control_flow_distance"] for result in results)
    total_interaction_penalty = sum(result["interaction_penalty"] for result in results)
    total_unmatched_penalty = sum(result["unmatched_penalty"] for result in results)
    sum_normalized_instruction_distance = sum(result["normalized_instruction_distance"] for result in results)
    mean_normalized_instruction_distance = sum(result["normalized_instruction_distance"] for result in results) / max(len(results), 1)
    mean_normalized_cfg_distance = sum(result["normalized_cfg_distance"] for result in results) / max(len(results), 1)
    mean_normalized_interaction_penalty = sum(result["normalized_interaction_penalty"] for result in results) / max(len(results), 1)
    mean_normalized_unmatched_penalty = sum(result["normalized_unmatched_penalty"] for result in results) / max(len(results), 1)
    mean_normalized_combined_distance = sum(result["normalized_combined_distance"] for result in results) / max(len(results), 1)

    return {
        "code_object": "__TOTAL__",
        "gt_name": None,
        "derived_name": None,
        "status": "summary",
        "gt_code_object_count": gt_code_objects,
        "derived_code_object_count": derived_code_objects,
        "gt_inst_count": sum(result["gt_inst_count"] for result in results),
        "derived_inst_count": sum(result["derived_inst_count"] for result in results),
        "instruction_distance": total_instruction_distance,
        "control_flow_distance": total_control_flow_distance,
        "interaction_penalty": total_interaction_penalty,
        "unmatched_penalty": total_unmatched_penalty,
        "sum_normalized_instruction_distance": sum_normalized_instruction_distance,
        "control_flow_weight": CONTROL_FLOW_WEIGHT,
        "combined_distance": total_combined_distance,
        "normalized_instruction_distance": mean_normalized_instruction_distance,
        "normalized_cfg_distance": mean_normalized_cfg_distance,
        "normalized_interaction_penalty": mean_normalized_interaction_penalty,
        "normalized_unmatched_penalty": mean_normalized_unmatched_penalty,
        "normalized_combined_distance": mean_normalized_combined_distance,
    }


def print_results(results) -> None:
    print("code_object\tstatus\tgt_inst_count\tderived_inst_count\tinstruction_distance\tcontrol_flow_distance\tinteraction_penalty\tunmatched_penalty\tcombined_distance\tnormalized_combined_distance")
    for result in results:
        print(
            "\t".join(
                (
                    result["code_object"],
                    result["status"],
                    str(result["gt_inst_count"]),
                    str(result["derived_inst_count"]),
                    str(result["instruction_distance"]),
                    str(result["control_flow_distance"]),
                    str(result["interaction_penalty"]),
                    str(result["unmatched_penalty"]),
                    str(result["combined_distance"]),
                    f"{result['normalized_combined_distance']:.6f}",
                )
            )
        )
    summary = summarize_results(results)
    print(
        "\t".join(
            (
                summary["code_object"],
                summary["status"],
                str(summary["gt_inst_count"]),
                str(summary["derived_inst_count"]),
                str(summary["instruction_distance"]),
                str(summary["control_flow_distance"]),
                str(summary["interaction_penalty"]),
                str(summary["unmatched_penalty"]),
                str(summary["combined_distance"]),
                f"{summary['normalized_combined_distance']:.6f}",
            )
        )
    )


def save_results_csv(results, path: Path, file_hash: str | None = None) -> None:
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    summary = summarize_results(results)

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "file_hash",
                "status",
                "gt_code_object_count",
                "derived_code_object_count",
                "total_gt_inst_count",
                "total_derived_inst_count",
                "sum_instruction_distance",
                "sum_control_flow_distance",
                "sum_interaction_penalty",
                "sum_unmatched_penalty",
                "sum_code_object_normalized_instruction_distance",
                "mean_code_object_normalized_instruction_distance",
                "mean_code_object_normalized_cfg_distance",
                "mean_code_object_normalized_interaction_penalty",
                "mean_code_object_normalized_unmatched_penalty",
                "control_flow_weight",
                "sum_combined_distance",
                "mean_code_object_normalized_combined_distance",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "file_hash": file_hash,
                "status": summary["status"],
                "gt_code_object_count": summary["gt_code_object_count"],
                "derived_code_object_count": summary["derived_code_object_count"],
                "total_gt_inst_count": summary["gt_inst_count"],
                "total_derived_inst_count": summary["derived_inst_count"],
                "sum_instruction_distance": summary["instruction_distance"],
                "sum_control_flow_distance": summary["control_flow_distance"],
                "sum_interaction_penalty": summary["interaction_penalty"],
                "sum_unmatched_penalty": summary["unmatched_penalty"],
                "sum_code_object_normalized_instruction_distance": f"{summary['sum_normalized_instruction_distance']:.6f}",
                "mean_code_object_normalized_instruction_distance": f"{summary['normalized_instruction_distance']:.6f}",
                "mean_code_object_normalized_cfg_distance": f"{summary['normalized_cfg_distance']:.6f}",
                "mean_code_object_normalized_interaction_penalty": f"{summary['normalized_interaction_penalty']:.6f}",
                "mean_code_object_normalized_unmatched_penalty": f"{summary['normalized_unmatched_penalty']:.6f}",
                "control_flow_weight": summary["control_flow_weight"],
                "sum_combined_distance": summary["combined_distance"],
                "mean_code_object_normalized_combined_distance": f"{summary['normalized_combined_distance']:.6f}",
            }
        )


def validate_semantic_error_pyc_paths(dataset_path: Path | None = None) -> dict:
    dataset_path = (dataset_path or (REPO_ROOT / "dataset" / "pyllmpatch_dataset.csv")).expanduser().resolve()
    df = pd.read_csv(dataset_path)
    semantic_df = df[df["error_type"] == "semantic_error"].copy()

    missing_rows = []
    for row in semantic_df.itertuples(index=False):
        original_pyc, indented_pyc = fetch_pyllmpatch_pyc_paths(row.file_hash, row.source)
        if original_pyc is None or indented_pyc is None:
            missing_rows.append(
                {
                    "file_hash": row.file_hash,
                    "source": row.source,
                    "original_pyc": str(original_pyc) if original_pyc else None,
                    "indented_pyc": str(indented_pyc) if indented_pyc else None,
                }
            )

    return {
        "dataset_path": str(dataset_path),
        "semantic_error_count": int(len(semantic_df)),
        "all_resolved": len(missing_rows) == 0,
        "missing_count": len(missing_rows),
        "missing_rows": missing_rows,
    }


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    pyc_a = validate_input(args.pyc_a)
    pyc_b = validate_input(args.pyc_b)

    results = compare_code_object_distances(pyc_a, pyc_b)
    print_results(results)

    if args.csv_out is not None:
        save_results_csv(results, args.csv_out, file_hash=args.file_hash)
        print(f"\nSaved CSV to: {args.csv_out.expanduser().resolve()}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
