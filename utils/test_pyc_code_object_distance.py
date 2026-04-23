from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import xdis.opcodes
from xdis.load import load_module

REPO_ROOT = Path(__file__).resolve().parent.parent
PYLINGUAL_ROOT = REPO_ROOT / "pylingual"

if str(PYLINGUAL_ROOT) not in sys.path:
    sys.path.insert(0, str(PYLINGUAL_ROOT))

from pylingual.editable_bytecode import EditableBytecode, Inst
from pylingual.editable_bytecode.bytecode_patches import fix_indirect_jump, fix_unreachable, remove_extended_arg, remove_nop, replace_firstlno
from pylingual.equivalence_check import matching_iter


PATCHES = [remove_extended_arg, remove_nop, fix_indirect_jump, fix_unreachable, remove_extended_arg, replace_firstlno]


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


def load_editable_bytecode_from_pyc(path: Path) -> EditableBytecode:
    source_tuple = load_module(str(path))
    if len(source_tuple) < 4:
        raise ValueError(f"Unexpected xdis load result with {len(source_tuple)} fields")

    version = source_tuple[0]
    code = source_tuple[3]
    opcode = getattr(xdis.opcodes, f"opcode_{version[0]}{version[1]}")

    bytecode = EditableBytecode(code, opcode, version)
    bytecode.apply_patches(PATCHES)
    return bytecode


def normalize_argval(inst: Inst):
    argval = getattr(inst, "argval", None)

    if hasattr(argval, "co_name"):
        return ("code", getattr(argval, "co_name", None))
    if inst.is_jump:
        return ("jump", inst.opname)
    if isinstance(argval, (str, int, float, bytes, type(None), bool)):
        return argval
    return repr(argval)


def instruction_signature(inst: Inst):
    return (inst.opname, normalize_argval(inst))


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


def compare_code_object_distances(pyc_a: Path, pyc_b: Path):
    prepared_a = load_editable_bytecode_from_pyc(pyc_a)
    prepared_b = load_editable_bytecode_from_pyc(pyc_b)

    results = []
    for bc_a, bc_b in matching_iter(prepared_a, prepared_b):
        name_a = bc_a.name if bc_a is not None else None
        name_b = bc_b.name if bc_b is not None else None
        len_a = len(bc_a) if bc_a is not None else 0
        len_b = len(bc_b) if bc_b is not None else 0

        if bc_a is None:
            distance = len_b
            status = "extra"
        elif bc_b is None:
            distance = len_a
            status = "missing"
        else:
            seq_a = [instruction_signature(inst) for inst in bc_a]
            seq_b = [instruction_signature(inst) for inst in bc_b]
            distance = levenshtein_distance(seq_a, seq_b)
            status = "matched"

        results.append(
            {
                "code_object": name_a if name_a == name_b else f"{name_a or 'None'} <-> {name_b or 'None'}",
                "name_a": name_a,
                "name_b": name_b,
                "status": status,
                "inst_count_a": len_a,
                "inst_count_b": len_b,
                "distance": distance,
                "normalized_distance": distance / max(len_a, len_b, 1),
            }
        )

    return results


def print_results(results) -> None:
    print("code_object\tstatus\tinst_count_a\tinst_count_b\tdistance\tnormalized_distance")
    for result in results:
        print(
            "\t".join(
                (
                    result["code_object"],
                    result["status"],
                    str(result["inst_count_a"]),
                    str(result["inst_count_b"]),
                    str(result["distance"]),
                    f"{result['normalized_distance']:.6f}",
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
                "distance",
                "normalized_distance",
            ],
        )
        writer.writeheader()
        for result in results:
            writer.writerow({**result, "normalized_distance": f"{result['normalized_distance']:.6f}"})


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
