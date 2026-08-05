from __future__ import annotations

import ast
import textwrap

from utils.semantic_operators.control_flow import (
    _NAME_LOADS,
    _SKIP_OPS,
    _const_repr,
    _covered_group_ordinals,
    _locate_body,
    _signature,
    _statement_groups,
    _unique_contiguous_match,
)
from utils.semantic_operators.structural import (
    _clean_name,
    _DropDead,
    _instructions,
    _match_indent_convention,
    _real_try_blocks,
)

# Scaffolding opcodes that never appear in a flat derived body, so a trailing one
# inside a GT protected range would break signature matching -> trim it.
_TRAILING_SCAFFOLD = {"JUMP_FORWARD", "JUMP_BACKWARD", "JUMP_ABSOLUTE", "NOP"}
_STORE_OPS = {"STORE_FAST", "STORE_NAME", "STORE_GLOBAL", "STORE_DEREF"}
_FALSE_JUMPS = {"POP_JUMP_IF_FALSE", "POP_JUMP_FORWARD_IF_FALSE",
                "POP_JUMP_BACKWARD_IF_FALSE"}
# Ops permitted between PUSH_EXC_INFO and the matched POP_TOP of a TRIVIAL handler
# (pure exception-type matching / stack shuffling — no real work).
_MATCH_OPS = {"CHECK_EXC_MATCH", "LOAD_GLOBAL", "LOAD_NAME", "LOAD_DEREF",
              "LOAD_ATTR", "COPY", "NOP"} | _FALSE_JUMPS


# Small bytecode helpers.
def _insts_in_range(insts: list, start, end) -> list:
    if start is None or end is None:
        return []
    return [i for i in insts if start <= getattr(i, "offset", -1) < end]


def _trim_trailing_scaffold(insts: list) -> list:
    out = list(insts)
    while out and out[-1].opname in _TRAILING_SCAFFOLD:
        out.pop()
    return out


def _raw_exception_entries(bc) -> list[tuple]:
    table = getattr(bc, "named_exception_table", None) or []
    return [(getattr(e, "start", None), getattr(e, "end", None),
             getattr(e, "target", None), getattr(e, "depth", None),
             getattr(e, "lasti", None)) for e in table]


# Handler classification.
def _except_kind(gt_insts: list, target) -> tuple[bool, str | None]:
    window: list = []
    started = False
    for i in gt_insts:
        if getattr(i, "offset", -2) == target:
            started = True
        if not started:
            continue
        window.append(i)
        if i.opname in ("POP_EXCEPT", "RERAISE"):
            break
    if len(window) < 3:
        return (False, None)
    if window[0].opname != "PUSH_EXC_INFO":
        return (False, None)
    if window[-1].opname != "POP_EXCEPT" or window[-2].opname != "POP_TOP":
        return (False, None)  # no clean matched-pop -> finally / content handler
    middle = window[1:-2]
    if any(i.opname not in _MATCH_OPS for i in middle):
        return (False, None)  # real work between PUSH_EXC_INFO and the pop
    exc_type = None
    if any(i.opname == "CHECK_EXC_MATCH" for i in middle):
        parts: list[str] = []
        for i in middle:
            if i.opname == "CHECK_EXC_MATCH":
                break
            if i.opname in ("LOAD_GLOBAL", "LOAD_NAME", "LOAD_DEREF"):
                nm = _clean_name(getattr(i, "argrepr", ""))
                if not nm:
                    return (False, None)
                parts = [nm]
            elif i.opname == "LOAD_ATTR":
                at = _clean_name(getattr(i, "argrepr", ""))
                if not at or not parts:
                    return (False, None)
                parts.append(at)
        if not parts:
            return (False, None)
        exc_type = ".".join(parts)
        try:
            ast.parse(exc_type, mode="eval")
        except (SyntaxError, ValueError):
            return (False, None)
    return (True, exc_type)


def _finally_body(gt_insts: list, target) -> list | None:
    window: list = []
    started = False
    for i in gt_insts:
        if getattr(i, "offset", -2) == target:
            started = True
        if not started:
            continue
        window.append(i)
        if i.opname == "RERAISE":
            break
    if len(window) < 3:
        return None
    if window[0].opname != "PUSH_EXC_INFO" or window[-1].opname != "RERAISE":
        return None
    body = window[1:-1]
    if any(i.opname in ("POP_EXCEPT", "CHECK_EXC_MATCH", "WITH_EXCEPT_START")
           for i in body):
        return None
    body = _trim_trailing_scaffold(body)
    return body or None


def _recover_ctx_load(insts: list) -> str | None:
    stack: list[str] = []
    for i in insts:
        op = i.opname
        if op in _SKIP_OPS:
            continue
        if op in ("BEFORE_WITH", "SETUP_WITH"):
            break
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
        elif op in ("LOAD_ATTR", "LOAD_METHOD"):
            if not stack:
                return None
            attr = _clean_name(getattr(i, "argrepr", ""))
            if not attr:
                return None
            stack[-1] = f"{stack[-1]}.{attr}"
        else:
            return None  # a call / operator / subscript is not a call-free load
    if len(stack) != 1:
        return None
    try:
        ast.parse(stack[0], mode="eval")
    except (SyntaxError, ValueError):
        return None
    return stack[0]


# Derived-side run location (mirrors control_flow: signature match -> statement
# ordinals). Returns (lo, hi) contiguous statement ordinals or None.
def _locate_run(der_bc, der_insts: list, needle: list[tuple]):
    if not needle:
        return None
    haystack = _signature(der_insts)
    start = _unique_contiguous_match(haystack, needle)
    if start is None:
        return None
    der_nonskip = [i for i in der_insts if i.opname not in _SKIP_OPS]
    matched = der_nonskip[start:start + len(needle)]
    matched_offsets = {getattr(i, "offset", None) for i in matched}
    groups = _statement_groups(der_bc)
    covered = _covered_group_ordinals(groups, matched_offsets)
    if not covered or covered != list(range(covered[0], covered[-1] + 1)):
        return None
    return covered[0], covered[-1]


# Shared finalization: DropDead placeholders -> unparse -> reindent -> re-parse
# guard -> result dict.
def _finalize(tree: ast.AST, fragment: str, opname: str, operator: str) -> dict | None:
    tree = _DropDead().visit(tree)
    ast.fix_missing_locations(tree)
    try:
        unparsed = ast.unparse(tree)
    except Exception:  # noqa: BLE001
        return None
    reindented = _match_indent_convention(fragment, unparsed)
    if reindented is None or reindented.strip() == fragment.strip():
        return None
    try:
        ast.parse(textwrap.dedent(reindented))
    except SyntaxError:
        return None
    return {
        "text": reindented,
        "operator": operator,
        "opname": opname,
        "kind": "exc_regions",
        "confidence": "unique",
    }


# WITH reconstruction.
def _reconstruct_with(gt, der, gt_insts, der_insts, fragment, dedented) -> dict | None:
    withs = [(idx, i) for idx, i in enumerate(gt_insts)
             if i.opname in ("BEFORE_WITH", "SETUP_WITH")]
    if len(withs) != 1:
        return None  # zero or multiple with-blocks -> not our clean single case
    bw_idx, bw_inst = withs[0]
    if bw_inst.opname == "SETUP_WITH":
        return None  # 3.10 block-stack form -> defer
    if not any(i.opname == "WITH_EXCEPT_START" for i in gt_insts):
        return None  # no real with cleanup -> not a with
    if any(i.opname in ("BEFORE_WITH", "SETUP_WITH") for i in der_insts):
        return None  # derived still has a with -> not a dropped-with case

    bw_off = getattr(bw_inst, "offset", None)
    prot = next((e for e in _raw_exception_entries(gt) if e[0] == (bw_off + 2)), None)
    if prot is None:
        return None
    body_end = prot[1]
    if bw_idx + 1 >= len(gt_insts):
        return None
    nxt = gt_insts[bw_idx + 1]
    as_target = None
    if nxt.opname in _STORE_OPS:
        as_target = _clean_name(getattr(nxt, "argrepr", ""))
        if not as_target:
            return None
    elif nxt.opname != "POP_TOP":
        return None  # unexpected enter-result handling -> defer

    body = _trim_trailing_scaffold(
        [i for i in gt_insts if getattr(nxt, "offset", -1) < getattr(i, "offset", -1) < body_end]
    )
    if not body:
        return None
    run = _locate_run(der, der_insts, _signature(body))
    if run is None:
        return None
    lo, hi = run

    tree = ast.parse(dedented)
    stmts, container = _locate_body(tree)
    if lo < 0 or hi >= len(stmts):
        return None

    if as_target is not None:
        # The decompiler flattens `with X as t:` to `t = X` + body; source the
        # context expression VERBATIM from that derived assignment (guaranteed
        # correct) and drop it, wrapping the present body in `with X as t:`.
        if lo - 1 < 0:
            return None
        prev = stmts[lo - 1]
        if not (isinstance(prev, ast.Assign) and len(prev.targets) == 1
                and isinstance(prev.targets[0], ast.Name)
                and prev.targets[0].id == as_target):
            return None
        ctx_node = prev.value
        optional_vars = ast.Name(id=as_target, ctx=ast.Store())
        with_node = ast.With(
            items=[ast.withitem(context_expr=ctx_node, optional_vars=optional_vars)],
            body=list(stmts[lo:hi + 1]),
        )
        new = stmts[:lo - 1] + [with_node] + stmts[hi + 1:]
        header = f"with <ctx> as {as_target}"
    else:
        # No `as` target: recover a call-free context expression from GT bytecode.
        ctx_insts: list = []
        j = bw_idx - 1
        while j >= 0:
            ctx_insts.insert(0, gt_insts[j])
            if getattr(gt_insts[j], "starts_line", None) is not None:
                break
            j -= 1
        ctx = _recover_ctx_load(ctx_insts)
        if ctx is None:
            return None
        try:
            ctx_node = ast.parse(ctx, mode="eval").body
        except (SyntaxError, ValueError):
            return None
        with_node = ast.With(
            items=[ast.withitem(context_expr=ctx_node, optional_vars=None)],
            body=list(stmts[lo:hi + 1]),
        )
        new = stmts[:lo] + [with_node] + stmts[hi + 1:]
        header = f"with {ctx}"

    container.body = new
    n = hi - lo + 1
    return _finalize(tree, fragment, "BEFORE_WITH",
                     f"wrap {n} statement{'s' if n != 1 else ''} in dropped `{header}:` from GT")


# TRY/EXCEPT and TRY/FINALLY reconstruction.
def _reconstruct_try(gt, der, gt_insts, der_insts, fragment, dedented) -> dict | None:
    gt_blocks = _real_try_blocks(gt)
    der_blocks = _real_try_blocks(der)
    # Scope: GT introduces exactly ONE new try around a fully-flat derived body.
    # More than one simultaneous try change is "multiple structural changes" -> defer.
    if len(gt_blocks) != 1 or len(der_blocks) != 0:
        return None
    s, e, tgt, _depth = gt_blocks[0]

    trivial, exc_type = _except_kind(gt_insts, tgt)
    if trivial:
        body = _trim_trailing_scaffold(_insts_in_range(gt_insts, s, e))
        run = _locate_run(der, der_insts, _signature(body))
        if run is None:
            return None
        lo, hi = run
        tree = ast.parse(dedented)
        stmts, container = _locate_body(tree)
        if lo < 0 or hi >= len(stmts):
            return None
        handler_type = ast.parse(exc_type, mode="eval").body if exc_type else None
        handler = ast.ExceptHandler(type=handler_type, name=None, body=[ast.Pass()])
        try_node = ast.Try(body=list(stmts[lo:hi + 1]), handlers=[handler],
                           orelse=[], finalbody=[])
        container.body = stmts[:lo] + [try_node] + stmts[hi + 1:]
        n = hi - lo + 1
        header = f"except {exc_type}" if exc_type else "except"
        return _finalize(tree, fragment, "PUSH_EXC_INFO",
                         f"wrap {n} statement{'s' if n != 1 else ''} in dropped "
                         f"try/`{header}: pass` from GT")

    # Not an except -> try/finally?
    fin_body = _finally_body(gt_insts, tgt)
    if fin_body is None:
        return None  # content-bearing except handler -> LLM territory
    try_body = _trim_trailing_scaffold(_insts_in_range(gt_insts, s, e))
    try_run = _locate_run(der, der_insts, _signature(try_body))
    fin_run = _locate_run(der, der_insts, _signature(fin_body))
    if try_run is None or fin_run is None:
        return None
    (t_lo, t_hi), (f_lo, f_hi) = try_run, fin_run
    if f_lo != t_hi + 1:
        return None  # finally body must be the run immediately after the try body

    tree = ast.parse(dedented)
    stmts, container = _locate_body(tree)
    if t_lo < 0 or f_hi >= len(stmts):
        return None
    try_node = ast.Try(body=list(stmts[t_lo:t_hi + 1]), handlers=[], orelse=[],
                       finalbody=list(stmts[f_lo:f_hi + 1]))
    container.body = stmts[:t_lo] + [try_node] + stmts[f_hi + 1:]
    nt, nf = t_hi - t_lo + 1, f_hi - f_lo + 1
    return _finalize(tree, fragment, "RERAISE",
                     f"wrap {nt} statement{'s' if nt != 1 else ''} + {nf} finally "
                     f"statement{'s' if nf != 1 else ''} in dropped try/finally from GT")


# Operator entry point.
def exc_regions_candidate(gt_code_object, derived_code_object, fragment: str,
                          repair_context: dict | None) -> dict | None:
    try:
        if not repair_context:
            return None
        failed = repair_context.get("pylingual_failed_result") or {}
        if failed.get("message") != "Different control flow":
            return None
        # Module / top-level repair is intentionally disabled (a `<module>` object
        # can't be fixed here) -> defer.
        co = getattr(gt_code_object, "codeobj", None) or getattr(gt_code_object, "code", None)
        if getattr(co, "co_name", None) == "<module>":
            return None

        gt_insts = _instructions(gt_code_object)
        der_insts = _instructions(derived_code_object)
        if not gt_insts:
            return None
        try:
            dedented = textwrap.dedent(fragment)
            ast.parse(dedented)
        except SyntaxError:
            return None

        out = _reconstruct_with(gt_code_object, derived_code_object,
                                gt_insts, der_insts, fragment, dedented)
        if out is not None:
            return out
        return _reconstruct_try(gt_code_object, derived_code_object,
                                gt_insts, der_insts, fragment, dedented)
    except Exception:  # noqa: BLE001 - a bad candidate defers; a raise would crash
        # the repair loop, so swallow everything. The oracle gate is the backstop.
        return None
