from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
PYLINGUAL_ROOT = REPO_ROOT / "pylingual"

if str(PYLINGUAL_ROOT) not in sys.path:
    sys.path.insert(0, str(PYLINGUAL_ROOT))

from utils.file_helpers import fetch_pyllmpatch_pyc_paths


CONTROL_FLOW_WEIGHT = 3


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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Measure distances between corresponding code objects in two .pyc files using pylingual."
    )
    parser.add_argument("pyc_a", type=Path, help="Path to the first .pyc file")
    parser.add_argument("pyc_b", type=Path, help="Path to the second .pyc file")
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


def levenshtein_distance(seq_a, seq_b) -> int:
    if len(seq_a) < len(seq_b):
        seq_a, seq_b = seq_b, seq_a

    previous = list(range(len(seq_b) + 1))
    for i, item_a in enumerate(seq_a, start=1):
        current = [i]
        for j, item_b in enumerate(seq_b, start=1):
            substitution_cost = 0 if item_a == item_b else 1
            current.append(
                min(
                    previous[j] + 1,
                    current[j - 1] + 1,
                    previous[j - 1] + substitution_cost,
                )
            )
        previous = current
    return previous[-1]


def build_block_graph(bytecode):
    deps = _load_bytecode_dependencies()
    cfg = deps["bytecode_to_control_flow_graph"](bytecode)
    return deps["CFG"].from_graph(cfg, bytecode, iterate=False)


def control_flow_distance(bc_a, bc_b) -> int:
    block_graph_a = build_block_graph(bc_a)
    block_graph_b = build_block_graph(bc_b)

    deps = _load_bytecode_dependencies()
    if deps["is_control_flow_equivalent"](block_graph_a, block_graph_b):
        return 0

    ordered_nodes_a = sorted(block_graph_a.nodes(data="offset", default=(float("inf"),)), key=lambda node: min(node[1]))
    ordered_nodes_b = sorted(block_graph_b.nodes(data="offset", default=(float("inf"),)), key=lambda node: min(node[1]))
    node_id_mapping = {node_a[0]: node_b[0] for node_a, node_b in zip(ordered_nodes_a, ordered_nodes_b)}

    edge_mismatch_count = 0
    for node_a, node_b in zip(ordered_nodes_a, ordered_nodes_b):
        edges_a = block_graph_a[node_a[0]]
        edges_b = block_graph_b[node_b[0]]

        destinations_a = {node_id_mapping[dest] for dest in edges_a.keys() if dest in node_id_mapping}
        destinations_b = set(edges_b.keys())
        edge_mismatch_count += len(destinations_a.symmetric_difference(destinations_b))

    return abs(len(block_graph_a) - len(block_graph_b)) + edge_mismatch_count


def compare_code_object_distances(pyc_a: Path, pyc_b: Path):
    prepared_a = load_editable_bytecode_from_pyc(pyc_a)
    prepared_b = load_editable_bytecode_from_pyc(pyc_b)
    deps = _load_bytecode_dependencies()

    results = []
    for bc_a, bc_b in deps["matching_iter"](prepared_a, prepared_b):
        name_a = bc_a.name if bc_a is not None else None
        name_b = bc_b.name if bc_b is not None else None
        len_a = len(bc_a) if bc_a is not None else 0
        len_b = len(bc_b) if bc_b is not None else 0

        if bc_a is None:
            instruction_distance = len_b
            cfg_distance = 1
            total_distance = instruction_distance + (CONTROL_FLOW_WEIGHT * cfg_distance)
            status = "extra"
        elif bc_b is None:
            instruction_distance = len_a
            cfg_distance = 1
            total_distance = instruction_distance + (CONTROL_FLOW_WEIGHT * cfg_distance)
            status = "missing"
        else:
            seq_a = [instruction_signature(inst) for inst in bc_a]
            seq_b = [instruction_signature(inst) for inst in bc_b]
            instruction_distance = levenshtein_distance(seq_a, seq_b)
            cfg_distance = control_flow_distance(bc_a, bc_b)
            total_distance = instruction_distance + (CONTROL_FLOW_WEIGHT * cfg_distance)
            status = "matched"

        results.append(
            {
                "code_object": name_a if name_a == name_b else f"{name_a or 'None'} <-> {name_b or 'None'}",
                "name_a": name_a,
                "name_b": name_b,
                "status": status,
                "inst_count_a": len_a,
                "inst_count_b": len_b,
                "instruction_distance": instruction_distance,
                "control_flow_distance": cfg_distance,
                "control_flow_weight": CONTROL_FLOW_WEIGHT,
                "distance": total_distance,
                "normalized_distance": total_distance / max(len_a, len_b, 1),
            }
        )

    return results


def summarize_results(results):
    total_weighted_distance = sum(result["distance"] for result in results)
    total_instruction_distance = sum(result["instruction_distance"] for result in results)
    total_control_flow_distance = sum(result["control_flow_distance"] for result in results)
    total_denominator = sum(max(result["inst_count_a"], result["inst_count_b"], 1) for result in results)

    return {
        "code_object": "__TOTAL__",
        "name_a": None,
        "name_b": None,
        "status": "summary",
        "inst_count_a": sum(result["inst_count_a"] for result in results),
        "inst_count_b": sum(result["inst_count_b"] for result in results),
        "instruction_distance": total_instruction_distance,
        "control_flow_distance": total_control_flow_distance,
        "control_flow_weight": CONTROL_FLOW_WEIGHT,
        "distance": total_weighted_distance,
        "normalized_distance": total_weighted_distance / max(total_denominator, 1),
    }


def print_results(results) -> None:
    print("code_object\tstatus\tinst_count_a\tinst_count_b\tinstruction_distance\tcontrol_flow_distance\tweighted_distance\tnormalized_distance")
    for result in results:
        print(
            "\t".join(
                (
                    result["code_object"],
                    result["status"],
                    str(result["inst_count_a"]),
                    str(result["inst_count_b"]),
                    str(result["instruction_distance"]),
                    str(result["control_flow_distance"]),
                    str(result["distance"]),
                    f"{result['normalized_distance']:.6f}",
                )
            )
        )
    summary = summarize_results(results)
    print(
        "\t".join(
            (
                summary["code_object"],
                summary["status"],
                str(summary["inst_count_a"]),
                str(summary["inst_count_b"]),
                str(summary["instruction_distance"]),
                str(summary["control_flow_distance"]),
                str(summary["distance"]),
                f"{summary['normalized_distance']:.6f}",
            )
        )
    )


def save_results_csv(results, path: Path) -> None:
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "code_object",
                "name_a",
                "name_b",
                "status",
                "inst_count_a",
                "inst_count_b",
                "instruction_distance",
                "control_flow_distance",
                "control_flow_weight",
                "distance",
                "normalized_distance",
            ],
        )
        writer.writeheader()
        for result in results:
            writer.writerow({**result, "normalized_distance": f"{result['normalized_distance']:.6f}"})
        summary = summarize_results(results)
        writer.writerow({**summary, "normalized_distance": f"{summary['normalized_distance']:.6f}"})


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
        save_results_csv(results, args.csv_out)
        print(f"\nSaved CSV to: {args.csv_out.expanduser().resolve()}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
