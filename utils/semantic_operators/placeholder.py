from __future__ import annotations

import ast
import textwrap

from utils.semantic_operators.leaf import _instructions, _round_trips

LINE_TOO_LONG_MARKER = (
    "Decompiler error: line too long for translation. "
    "Please decompile this statement manually."
)

_LITERAL_STORE_OPCODES = {"STORE_NAME", "STORE_FAST"}
# Broader set for "is this name assigned anywhere in the derived object?" — used only to
# rule a GT literal-store OUT as the deferred statement, so err on the side of inclusion.
_ANY_STORE_OPCODES = {"STORE_NAME", "STORE_FAST", "STORE_GLOBAL", "STORE_DEREF"}

# Names the compiler synthesizes into a module/class namespace; a "missing" one of these
# is never a dropped *source* assignment, so the dropped-assignment operator skips them.
_COMPILER_DUNDERS = frozenset({
    "__doc__", "__module__", "__qualname__", "__annotations__",
    "__static_attributes__", "__firstlineno__", "__dict__", "__weakref__",
})


def _argint(ins):
    a = getattr(ins, "arg", None)
    if isinstance(a, int):
        return a
    v = getattr(ins, "argval", None)
    return v if isinstance(v, int) else None


def _store_name(ins) -> str | None:
    v = getattr(ins, "argval", None)
    if isinstance(v, str) and v.isidentifier():
        return v
    r = (getattr(ins, "argrepr", "") or "").strip()
    return r if r.isidentifier() else None


def _gt_literal_store_targets(insts: list) -> list[tuple[str, object, str]]:
    seq = [i for i in insts if getattr(i, "opname", None) != "CACHE"]
    targets: list[tuple[str, object, str]] = []
    for idx, ins in enumerate(seq):
        if getattr(ins, "opname", None) not in _LITERAL_STORE_OPCODES:
            continue
        name = _store_name(ins)
        if name is None:
            continue
        # list-extend idiom (constant list display)
        if idx >= 3:
            build, load, extend = seq[idx - 3], seq[idx - 2], seq[idx - 1]
            if (build.opname == "BUILD_LIST" and _argint(build) == 0
                    and load.opname == "LOAD_CONST" and isinstance(getattr(load, "argval", None), tuple)
                    and extend.opname == "LIST_EXTEND" and _argint(extend) == 1):
                targets.append((name, getattr(load, "argval"), "list"))
                continue
        # scalar literal store: the whole RHS is a single LOAD_CONST. Review fix: a
        # conditional/short-circuit RHS (`K = a if c else 99`, `K = f() or 5`, `K = a and 3`)
        # ALSO ends in `LOAD_CONST; STORE` but is NOT a scalar assignment — the const-load is
        # the entry of a branch (a jump target), or the STORE is a branch-merge target. In
        # either case emitting `K = <const>` would drop the branch, so decline; a plain
        # `K = <const>` has neither instruction as a jump target.
        prev = seq[idx - 1] if idx >= 1 else None
        if (prev is not None and prev.opname == "LOAD_CONST"
                and not getattr(prev, "is_jump_target", False)
                and not getattr(ins, "is_jump_target", False)):
            targets.append((name, getattr(prev, "argval", None), "scalar"))
    return targets


def _derived_store_names(insts: list) -> set[str]:
    names: set[str] = set()
    for ins in insts:
        if getattr(ins, "opname", None) in _ANY_STORE_OPCODES:
            n = _store_name(ins)
            if n is not None:
                names.add(n)
    return names


def _is_annotated(insts: list, name: str) -> bool:
    for idx, ins in enumerate(insts):
        if getattr(ins, "opname", None) != "STORE_SUBSCR":
            continue
        window = insts[max(0, idx - 4):idx]
        has_key = any(w.opname == "LOAD_CONST" and getattr(w, "argval", None) == name for w in window)
        has_annotations = any(
            "__annotations__" in ((getattr(w, "argrepr", "") or "") + str(getattr(w, "argval", "") or ""))
            for w in window
        )
        if has_key and has_annotations:
            return True
    return False


def _marker_statement(fragment: str) -> ast.Expr | None:
    try:
        tree = ast.parse(textwrap.dedent(fragment))
    except SyntaxError:
        return None
    marker_consts = [
        n for n in ast.walk(tree)
        if isinstance(n, ast.Constant) and isinstance(n.value, str) and n.value == LINE_TOO_LONG_MARKER
    ]
    if len(marker_consts) != 1:
        return None
    target = marker_consts[0]
    for n in ast.walk(tree):
        if isinstance(n, ast.Expr) and n.value is target:
            return n
    return None


def _render_rhs(value, kind: str) -> str | None:
    if kind == "list":
        if not isinstance(value, tuple) or not _round_trips(value):
            return None
        items = list(value)
        rhs = repr(items)
    else:
        if not _round_trips(value):
            return None
        rhs = repr(value)
    try:
        if ast.literal_eval(rhs) != (list(value) if kind == "list" else value):
            return None
    except Exception:  # noqa: BLE001
        return None
    return rhs


def line_too_long_candidate(gt_code_object, derived_code_object, fragment: str,
                            repair_context: dict | None) -> dict | None:
    if not repair_context:
        return None
    failed = repair_context.get("pylingual_failed_result") or {}
    if failed.get("message") != "Different bytecode":
        return None

    stmt = _marker_statement(fragment)
    if stmt is None:
        return None

    gt = _instructions(gt_code_object)
    der = _instructions(derived_code_object)
    if not gt:
        return None

    der_names = _derived_store_names(der)
    aligned = [t for t in _gt_literal_store_targets(gt) if t[0] not in der_names]
    if len(aligned) != 1:
        return None
    name, value, kind = aligned[0]
    if not name.isidentifier():
        return None
    if _is_annotated(gt, name):
        return None

    rhs = _render_rhs(value, kind)
    if rhs is None:
        return None

    orig_lines = fragment.splitlines(keepends=True)
    li0, li1 = stmt.lineno - 1, stmt.end_lineno - 1
    if li0 < 0 or li1 >= len(orig_lines) or li0 > li1:
        return None
    first = orig_lines[li0]
    indent = first[:len(first) - len(first.lstrip())]
    trailing_nl = "\n" if orig_lines[li1].endswith("\n") else ""
    new_line = f"{indent}{name} = {rhs}{trailing_nl}"
    new_fragment = "".join(orig_lines[:li0] + [new_line] + orig_lines[li1 + 1:])
    if new_fragment == fragment:
        return None
    try:
        ast.parse(textwrap.dedent(new_fragment))
    except SyntaxError:
        return None

    preview = rhs if len(rhs) <= 40 else rhs[:37] + "..."
    return {
        "text": new_fragment,
        "operator": f"reconstruct {name} = {preview}",
        "opname": "STORE_NAME",
        "kind": "line_too_long",
        "confidence": "unique",
    }


def _ordered_store_names(insts: list) -> list[str]:
    out: list[str] = []
    for ins in insts:
        if getattr(ins, "opname", None) in _ANY_STORE_OPCODES:
            n = _store_name(ins)
            if n is not None:
                out.append(n)
    return out


def _unique_assign_target(tree: ast.AST, name: str) -> ast.Assign | None:
    matches: list[ast.Assign] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            if any(isinstance(t, ast.Name) and t.id == name for t in node.targets):
                matches.append(node)
    return matches[0] if len(matches) == 1 else None


def dropped_assignment_candidate(gt_code_object, derived_code_object, fragment: str,
                                 repair_context: dict | None) -> dict | None:
    if not repair_context:
        return None
    failed = repair_context.get("pylingual_failed_result") or {}
    if failed.get("message") != "Different bytecode":
        return None
    # Hand-off: a fragment still carrying the line-too-long marker is owned by
    # line_too_long_candidate (it overwrites the marker instead of inserting).
    if LINE_TOO_LONG_MARKER in fragment:
        return None

    gt = _instructions(gt_code_object)
    der = _instructions(derived_code_object)
    if not gt:
        return None

    der_names = _derived_store_names(der)
    absent = [
        t for t in _gt_literal_store_targets(gt)
        if t[2] == "scalar" and t[0] not in der_names and t[0] not in _COMPILER_DUNDERS
    ]
    if len(absent) != 1:
        return None
    name, value, _kind = absent[0]
    if not name.isidentifier():
        return None
    if _is_annotated(gt, name):
        return None
    if not _round_trips(value):
        return None

    try:
        tree = ast.parse(textwrap.dedent(fragment))
    except SyntaxError:
        return None

    # The one added judgement: locate the dropped store among GT's ordered stores and take
    # the immediately adjacent neighbour — prefer the preceding store (splice AFTER it);
    # if the drop is the very first store, fall back to the succeeding store (splice
    # BEFORE it). Either way the neighbour must map to a UNIQUE source assignment.
    ordered = _ordered_store_names(gt)
    if name not in ordered:
        return None
    pos = ordered.index(name)
    neighbor = None
    insert_after = True
    for j in range(pos - 1, -1, -1):
        if ordered[j] != name:
            neighbor, insert_after = ordered[j], True
            break
    if neighbor is None:
        for j in range(pos + 1, len(ordered)):
            if ordered[j] != name:
                neighbor, insert_after = ordered[j], False
                break
    if neighbor is None:
        return None

    anchor = _unique_assign_target(tree, neighbor)
    if anchor is None:
        return None
    if anchor.lineno is None or anchor.end_lineno is None:
        return None

    orig_lines = fragment.splitlines(keepends=True)
    ref_idx = anchor.lineno - 1
    if ref_idx < 0 or anchor.end_lineno > len(orig_lines):
        return None
    ref_line = orig_lines[ref_idx]
    indent = ref_line[:len(ref_line) - len(ref_line.lstrip())]
    rhs = repr(value)
    new_line = f"{indent}{name} = {rhs}\n"

    cut = anchor.end_lineno if insert_after else anchor.lineno - 1
    before = list(orig_lines[:cut])
    after = list(orig_lines[cut:])
    if before and not before[-1].endswith("\n"):
        before[-1] = before[-1] + "\n"
    new_fragment = "".join(before + [new_line] + after)
    if new_fragment == fragment:
        return None
    try:
        ast.parse(textwrap.dedent(new_fragment))
    except SyntaxError:
        return None

    preview = rhs if len(rhs) <= 40 else rhs[:37] + "..."
    return {
        "text": new_fragment,
        "operator": f"insert {name} = {preview}",
        "opname": "STORE_NAME",
        "kind": "dropped_assign",
        "confidence": "unique",
    }
