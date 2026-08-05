from __future__ import annotations

import re

from utils.semantic_operators.control_flow import (
    _HEADER_JUMPS,
    _bc_to_cft,
    _const_repr,
    _members,
    _recover_if_condition,
    _unwrap_body,
)
from utils.semantic_operators.leaf import _instructions
from utils.semantic_operators.structural import (
    _clean_name,
    _protected_call_parts,
    _real_try_blocks,
)

# Instructions with no bearing on statement content / value reconstruction.
_SKIP_OPS = {
    "CACHE", "RESUME", "PRECALL", "MAKE_CELL", "COPY_FREE_VARS", "EXTENDED_ARG",
    "NOP", "GEN_START", "RETURN_GENERATOR", "PUSH_NULL", "KW_NAMES",
}
_NAME_LOADS = {"LOAD_FAST", "LOAD_NAME", "LOAD_GLOBAL", "LOAD_DEREF", "LOAD_CLASSDEREF",
               "LOAD_FAST_CHECK"}
_STORE_OPS = {"STORE_FAST", "STORE_NAME", "STORE_GLOBAL", "STORE_DEREF"}
_ATTR_OPS = {"LOAD_ATTR", "LOAD_METHOD"}
_UNPACK_OPS = {"UNPACK_SEQUENCE", "UNPACK_EX"}

# Tree node class-name families are probed via their MEMBER keys (version-robust) rather
# than class names, but we still use the name to split "if" from a while-with-if pattern.
_MAX_DIRECTIVE_CLAUSES = 3


# Small instruction-stream helpers (never raise).
def _nonskip(insts: list) -> list:
    return [i for i in insts if getattr(i, "opname", None) not in _SKIP_OPS]


def _jump_target(inst) -> int | None:
    val = getattr(inst, "argval", None)
    if isinstance(val, int):
        return val
    m = re.search(r"to (\d+)", getattr(inst, "argrepr", "") or "")
    return int(m.group(1)) if m else None


def _stmt_groups(insts: list) -> list[list]:
    groups: list[list] = []
    for i in _nonskip(insts):
        ln = getattr(i, "starts_line", None)
        if ln is not None and (not groups or getattr(groups[-1][0], "starts_line", None) != ln):
            groups.append([i])
        elif groups:
            groups[-1].append(i)
        else:
            groups.append([i])
    return groups


def _first_group(insts: list) -> list:
    groups = _stmt_groups(insts)
    return groups[0] if groups else []


def _last_group(insts: list) -> list:
    groups = _stmt_groups(insts)
    return groups[-1] if groups else []


# Expression / statement rendering (best-effort, simple forms only).
def _simple_value(insts: list) -> str | None:
    stack: list[str] = []
    for i in _nonskip(insts):
        op = i.opname
        if op == "GET_ITER":
            continue
        if op in _NAME_LOADS:
            name = _clean_name(getattr(i, "argrepr", ""))
            if not name:
                return None
            stack.append(name)
        elif op == "LOAD_CONST":
            rep = _const_repr(getattr(i, "argval", None))
            if rep is None:
                return None
            stack.append(rep)
        elif op in _ATTR_OPS:
            if not stack:
                return None
            attr = _clean_name(getattr(i, "argrepr", ""))
            if not attr:
                return None
            stack[-1] = f"{stack[-1]}.{attr}"
        else:
            return None
    return stack[0] if len(stack) == 1 else None


def _leading_callee(insts: list) -> str | None:
    parts: list[str] = []
    for i in _nonskip(insts):
        op = i.opname
        if not parts:
            if op in _NAME_LOADS:
                name = _clean_name(getattr(i, "argrepr", ""))
                if not name:
                    return None
                parts = [name]
            else:
                return None
        elif op in _ATTR_OPS:
            attr = _clean_name(getattr(i, "argrepr", ""))
            if not attr:
                break
            parts.append(attr)
        else:
            break
    return ".".join(parts) if parts else None


def _loop_body_first_stmt(body_insts: list) -> str | None:
    stripped: list = []
    skipping = True
    for i in body_insts:
        op = getattr(i, "opname", None)
        if skipping and (op in _SKIP_OPS or op in _UNPACK_OPS or op in _STORE_OPS):
            continue
        skipping = False
        stripped.append(i)
    return _describe_statement(_first_group(stripped))


def _describe_statement(group: list) -> str | None:
    core = [i for i in _nonskip(group) if not i.opname.startswith("JUMP")]
    if not core:
        core = _nonskip(group)
    if not core:
        return None
    has_call = any(i.opname.startswith("CALL") for i in core)
    store_idxs = [k for k, i in enumerate(core) if i.opname in _STORE_OPS]
    last = core[-1]

    # augmented assignment: BINARY_OP whose symbol ends in '=' (e.g. '+=') + trailing store
    binop_aug_idx = next(
        (k for k, i in enumerate(core)
         if i.opname == "BINARY_OP" and (getattr(i, "argrepr", "") or "").endswith("=")),
        None,
    )
    if store_idxs and binop_aug_idx is not None:
        target = _clean_name(getattr(core[store_idxs[-1]], "argrepr", ""))
        symbol = (getattr(core[binop_aug_idx], "argrepr", "") or "").strip()
        operand = _simple_value(core[1:binop_aug_idx])  # skip the initial load of the target
        if target and symbol:
            return f"{target} {symbol} {operand}" if operand else f"{target} {symbol} ..."

    if last.opname in _STORE_OPS:  # plain assignment
        target = _clean_name(getattr(last, "argrepr", ""))
        rhs = core[:store_idxs[-1]] if store_idxs else core[:-1]
        if target:
            if any(i.opname.startswith("CALL") for i in rhs):
                callee = _leading_callee(rhs)
                return f"{target} = {callee}(...)" if callee else f"{target} = ..."
            val = _simple_value(rhs)
            return f"{target} = {val}" if val else f"{target} = ..."

    if last.opname == "RETURN_CONST":
        rep = _const_repr(getattr(last, "argval", None))
        return f"return {rep}" if rep is not None else "return ..."

    if last.opname == "RETURN_VALUE":
        rhs = core[:-1]
        if any(i.opname.startswith("CALL") for i in rhs):
            callee = _leading_callee(rhs)
            return f"return {callee}(...)" if callee else "return ..."
        val = _simple_value(rhs)
        return f"return {val}" if val else "return ..."

    if has_call:
        callee = _leading_callee(core)
        return f"{callee}(...)" if callee else None
    return None


# for-loop header recovery (target + iterable) from the raw stream.
def _for_target(full: list, f_idx: int) -> str | None:
    names: list[str] = []
    for i in full[f_idx + 1:]:
        op = getattr(i, "opname", None)
        if op in _SKIP_OPS or op in _UNPACK_OPS:
            continue
        if op in _STORE_OPS:
            name = _clean_name(getattr(i, "argrepr", ""))
            if name:
                names.append(name)
            continue
        break
    return ", ".join(names) if names else None


def _for_iterable(full: list, f_idx: int) -> str | None:
    gi = None
    for k in range(f_idx - 1, -1, -1):
        op = getattr(full[k], "opname", None)
        if op in _SKIP_OPS:
            continue
        if op == "GET_ITER":
            gi = k  # GET_ITER immediately precedes FOR_ITER (after any noise)
        break
    if gi is None:
        return None
    # statement group start: nearest earlier instruction carrying a line number
    gs = gi
    while gs > 0 and getattr(full[gs], "starts_line", None) is None:
        gs -= 1
    return _simple_value(full[gs:gi])  # exclude GET_ITER itself


# Construct records.  Each record: {kind, sig, cond, target, iterable, exc,
# body_desc, offset}.  sig is the diff/dedupe key.
def _if_record(cond: str, body_desc: str | None, offset: int) -> dict:
    return {"kind": "if", "sig": f"if::{cond}", "cond": cond, "body_desc": body_desc,
            "offset": offset}


def _for_record(target: str | None, iterable: str | None, body_desc: str | None, offset: int) -> dict:
    return {"kind": "for", "sig": f"for::{target or '?'}::{iterable or '?'}",
            "target": target, "iterable": iterable, "body_desc": body_desc, "offset": offset}


def _while_record(cond: str, body_desc: str | None, offset: int) -> dict:
    return {"kind": "while", "sig": f"while::{cond}", "cond": cond, "body_desc": body_desc,
            "offset": offset}


def _with_record(ctx: str | None, target: str | None, body_desc: str | None, offset: int) -> dict:
    return {"kind": "with", "sig": f"with::{ctx or '?'}::{target or '?'}", "ctx": ctx,
            "target": target, "body_desc": body_desc, "offset": offset}


def _try_record(dotted: str | None, exc: str | None, body_desc: str | None, offset: int) -> dict:
    return {"kind": "try", "sig": f"try::{dotted or exc or offset}", "exc": exc,
            "body_desc": body_desc, "offset": offset}


# --- bytecode-window extractors (robust; work on irreducible objects) -------
def _window_if_records(full: list) -> list[dict]:
    records: list[dict] = []
    for j, inst in enumerate(full):
        if getattr(inst, "opname", None) not in _HEADER_JUMPS:
            continue
        target = _jump_target(inst)
        off = getattr(inst, "offset", None)
        if target is None or off is None or target <= off:
            continue  # only forward guards; a back-edge is a loop, not an if
        gs = j
        while gs > 0 and getattr(full[gs], "starts_line", None) is None:
            gs -= 1
        cond = _recover_if_condition(full[gs:j + 1])
        if cond is None:
            continue
        # guarded body = fall-through range up to the branch target
        k = j + 1
        while k < len(full) and getattr(full[k], "offset", -1) < target:
            k += 1
        body_desc = _describe_statement(_first_group(full[j + 1:k]))
        records.append(_if_record(cond, body_desc, off))
    return records


def _window_for_records(full: list) -> list[dict]:
    records: list[dict] = []
    for j, inst in enumerate(full):
        if getattr(inst, "opname", None) != "FOR_ITER":
            continue
        off = getattr(inst, "offset", None)
        target = _for_target(full, j)
        iterable = _for_iterable(full, j)
        # body = statements after the target store, up to the loop back-edge
        b = j + 1
        while b < len(full) and getattr(full[b], "opname", None) in (_SKIP_OPS | _UNPACK_OPS | _STORE_OPS):
            b += 1
        end = len(full)
        for k in range(b, len(full)):
            op = getattr(full[k], "opname", None)
            if op and op.startswith("JUMP_BACKWARD"):
                end = k
                break
        body_desc = _describe_statement(_first_group(full[b:end]))
        records.append(_for_record(target, iterable, body_desc, off if off is not None else -1))
    return records


def _handler_exc_type(insts: list, target: int) -> str | None:
    window: list = []
    started = False
    for i in insts:
        if getattr(i, "offset", -1) == target:
            started = True
        if not started:
            continue
        window.append(i)
        if getattr(i, "opname", None) == "CHECK_EXC_MATCH":
            break
        if getattr(i, "opname", None) in ("RERAISE", "JUMP_FORWARD", "JUMP_BACKWARD"):
            return None
    if not window or window[-1].opname != "CHECK_EXC_MATCH":
        return None
    parts: list[str] = []
    for i in window[:-1]:
        op = getattr(i, "opname", None)
        if op in ("LOAD_GLOBAL", "LOAD_NAME", "LOAD_DEREF"):
            name = _clean_name(getattr(i, "argrepr", ""))
            if name:
                parts = [name]
        elif op in _ATTR_OPS and parts:
            attr = _clean_name(getattr(i, "argrepr", ""))
            if attr:
                parts.append(attr)
    return ".".join(parts) if parts else None


def _is_with_handler(insts: list, target: int) -> bool:
    started = False
    for i in insts:
        if getattr(i, "offset", -1) == target:
            started = True
        if not started:
            continue
        op = getattr(i, "opname", None)
        if op == "WITH_EXCEPT_START":
            return True
        if op in ("RERAISE", "JUMP_FORWARD", "JUMP_BACKWARD"):
            break
    return False


def _try_records(bc) -> list[dict]:
    records: list[dict] = []
    insts = _instructions(bc)
    for (start, end, target, _depth) in _real_try_blocks(bc):
        if start is None or end is None or target is None:
            continue
        if _is_with_handler(insts, target):
            continue
        parts = _protected_call_parts(insts, start, end)
        exc = _handler_exc_type(insts, target)
        dotted = ".".join(parts) if parts else None
        protected = [i for i in insts if start <= getattr(i, "offset", -1) < end]
        body_desc = (_describe_statement(_last_group(protected))
                     or _describe_statement(_first_group(protected))
                     or (f"{dotted}(...)" if dotted else None))
        records.append(_try_record(dotted, exc, body_desc, start))
    return records


# --- bc_to_cft tree extractor (primary structural source) -------------------
def _classify(node):
    name = type(node).__name__
    m = _members(node)
    if not isinstance(m, dict):
        return None
    keys = m.keys()
    if "for_iter" in keys:
        return ("for", m.get("for_iter"), m.get("for_body"))
    if "try_body" in keys:
        return None  # try handled via the exception table (more reliable)
    if "setup_with" in keys:
        return ("with", m.get("setup_with"), m.get("with_body"))
    if "while_header" in keys:
        return ("while", m.get("while_header"), m.get("while_body"))
    if "loop_header" in keys:
        return ("while", m.get("loop_header"), m.get("loop_body"))
    if "if_header" in keys:
        kind = "while" if "Loop" in name else "if"
        return (kind, m.get("if_header"), m.get("if_body"))
    return None


def _walk_tree(node, out: list, depth: int = 0):
    if node is None or depth > 60:
        return
    m = _members(node)
    if isinstance(m, list):  # BlockTemplate
        for child in m:
            _walk_tree(child, out, depth)
        return
    if isinstance(m, dict):
        info = _classify(node)
        if info is not None:
            kind, header, body = info
            out.append((kind, header, body, node))
        for v in m.values():
            if v is not None:
                _walk_tree(v, out, depth + 1)
        return
    # InstTemplate / MetaTemplate / leaf: stop.


def _node_insts(node) -> list:
    try:
        return node.get_instructions() if node is not None else []
    except Exception:  # noqa: BLE001
        return []


def _tree_records(bc, full: list) -> list[dict]:
    tree = _bc_to_cft(bc)
    if tree is None:
        return []
    root = _unwrap_body(tree)
    constructs: list[tuple] = []
    _walk_tree(root, constructs)
    records: list[dict] = []
    for kind, header, body, node in constructs:
        offset = getattr(node, "offset", -1)
        body_insts = _node_insts(body)
        if kind in ("if", "while"):
            cond = _recover_if_condition(_node_insts(header))
            if cond is None:
                continue
            desc = _describe_statement(_first_group(body_insts))
            records.append(_if_record(cond, desc, offset) if kind == "if"
                           else _while_record(cond, desc, offset))
        elif kind == "for":
            f_off = getattr(header, "offset", None)
            if f_off is None:
                continue
            f_idx = next((i for i, ins in enumerate(full)
                          if getattr(ins, "offset", None) == f_off), None)
            if f_idx is None:
                continue
            target = _for_target(full, f_idx)
            iterable = _for_iterable(full, f_idx)
            desc = _loop_body_first_stmt(body_insts)
            records.append(_for_record(target, iterable, desc, offset))
        elif kind == "with":
            ctx = _simple_value([i for i in _node_insts(header)
                                 if getattr(i, "opname", None) != "BEFORE_WITH"])
            target = None
            for i in body_insts:
                if getattr(i, "opname", None) in _STORE_OPS:
                    target = _clean_name(getattr(i, "argrepr", ""))
                    break
            desc = _describe_statement(_first_group(body_insts))
            records.append(_with_record(ctx, target, desc, offset))
    return records


def _constructs(bc) -> list[dict]:
    full = _instructions(bc)
    if not full:
        return []
    records = list(_tree_records(bc, full))
    records += _window_if_records(full)
    records += _window_for_records(full)
    records += _try_records(bc)
    deduped: dict[tuple, dict] = {}
    for rec in records:
        key = (rec["kind"], rec["sig"])
        if key not in deduped or (rec.get("body_desc") and not deduped[key].get("body_desc")):
            # first-seen wins, but a richer (body_desc-bearing) record upgrades it
            if key not in deduped:
                deduped[key] = rec
            elif rec.get("body_desc") and not deduped[key].get("body_desc"):
                deduped[key] = rec
    return list(deduped.values())


# Diff + directive rendering.
def _unmatched(gt_records: list[dict], der_records: list[dict]) -> list[dict]:
    pool: dict[tuple, int] = {}
    for r in der_records:
        pool[(r["kind"], r["sig"])] = pool.get((r["kind"], r["sig"]), 0) + 1
    out: list[dict] = []
    for r in sorted(gt_records, key=lambda x: (x.get("offset", 0))):
        key = (r["kind"], r["sig"])
        if pool.get(key, 0) > 0:
            pool[key] -= 1
        else:
            out.append(r)
    return out


def _clause(rec: dict) -> str | None:
    kind = rec["kind"]
    body = rec.get("body_desc")
    if kind == "if":
        cond = rec["cond"]
        if body:
            return (f"wrap the statements starting at `{body}` inside `if {cond}:` — "
                    "the ground-truth bytecode guards them by that condition, but the "
                    "decompiled source dropped or flattened that guard")
        return (f"reconstruct the `if {cond}:` conditional the ground-truth bytecode has "
                "but the decompiled source dropped or mis-nested")
    if kind == "while":
        cond = rec["cond"]
        tail = f" around `{body}`" if body else ""
        return (f"reconstruct the `while {cond}:` loop{tail} — the ground-truth runs "
                "that block inside a loop the decompiled source dropped")
    if kind == "for":
        target = rec.get("target")
        if not target:
            return None  # a for-loop with no recoverable loop variable is too vague — suppress
        iterable = rec.get("iterable") or "..."
        tail = f" wrapping `{body}`" if body else ""
        return (f"reconstruct the loop `for {target} in {iterable}:`{tail} — the "
                "ground-truth runs those statements inside a for-loop the decompiled "
                "source dropped")
    if kind == "with":
        ctx = rec.get("ctx") or "..."
        target = rec.get("target")
        as_clause = f" as {target}" if target else ""
        tail = f" (`{body}`)" if body else ""
        return (f"wrap the block{tail} in a `with {ctx}{as_clause}:` context manager the "
                "decompiled source dropped")
    if kind == "try":
        exc = rec.get("exc")
        handler = f"an `except {exc}:` handler" if exc else "an `except` handler"
        anchor = f" ending in `{body}`" if body else ""
        return (f"nest the block{anchor} inside a `try:` with {handler} — the decompiled "
                "source is missing that try/except")
    return None


def structural_repair_directive(gt_code_object, derived_code_object) -> str | None:
    try:
        gt_records = _constructs(gt_code_object)
        if not gt_records:
            return None
        der_records = _constructs(derived_code_object)
        unmatched = _unmatched(gt_records, der_records)
        if not unmatched:
            return None
        clauses: list[str] = []
        for rec in unmatched:
            clause = _clause(rec)
            if clause:
                clauses.append(clause)
            if len(clauses) >= _MAX_DIRECTIVE_CLAUSES:
                break
        if not clauses:
            return None
        if len(clauses) == 1:
            return "Structural fix: " + clauses[0] + "."
        body = "; ".join(f"({n + 1}) {c}" for n, c in enumerate(clauses))
        return (f"Structural fix ({len(clauses)} structural differences from ground truth): "
                + body + ".")
    except Exception:  # noqa: BLE001 - a directive is advisory; never break the caller.
        return None


def gt_structure_directive(gt_bytecode, max_clauses: int = 4) -> str:
    try:
        records = _constructs(gt_bytecode)
    except Exception:  # noqa: BLE001 - advisory only; never break the caller.
        return ""
    if not records:
        return ""
    clauses: list[str] = []
    for rec in sorted(records, key=lambda r: r.get("offset", 0)):
        clause = _clause(rec)
        if clause:
            clauses.append(clause)
        if len(clauses) >= max_clauses:
            break
    if not clauses:
        return ""
    if len(clauses) == 1:
        return "Ground-truth control flow to reconstruct: " + clauses[0] + "."
    body = "; ".join(f"({n + 1}) {c}" for n, c in enumerate(clauses))
    return f"Ground-truth control flow to reconstruct ({len(clauses)} constructs): " + body + "."


def structural_directive_or_empty(gt_code_object, derived_code_object) -> str:
    """Thin wrapper returning ``""`` instead of None, for easy prompt concatenation."""
    return structural_repair_directive(gt_code_object, derived_code_object) or ""
