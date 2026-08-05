from __future__ import annotations

import ast
import textwrap

from utils.semantic_operators.leaf import _instructions
from utils.semantic_operators.structural import _clean_name, _match_indent_convention

# Instructions that carry no semantic weight for signature/condition purposes.
_SKIP_OPS = {
    "CACHE", "RESUME", "PRECALL", "MAKE_CELL", "COPY_FREE_VARS", "EXTENDED_ARG",
    "NOP", "GEN_START", "RETURN_GENERATOR", "PUSH_NULL",
}
_NAME_LOADS = {"LOAD_FAST", "LOAD_NAME", "LOAD_GLOBAL", "LOAD_DEREF", "LOAD_CLASSDEREF",
               "LOAD_FAST_CHECK"}
_STORE_OPS = {"STORE_FAST", "STORE_NAME", "STORE_GLOBAL", "STORE_DEREF"}
# Jump opcodes that terminate an ``if`` header (body runs on the fall-through).
_FALSE_JUMPS = {"POP_JUMP_IF_FALSE", "POP_JUMP_FORWARD_IF_FALSE", "POP_JUMP_BACKWARD_IF_FALSE"}
_TRUE_JUMPS = {"POP_JUMP_IF_TRUE", "POP_JUMP_FORWARD_IF_TRUE", "POP_JUMP_BACKWARD_IF_TRUE"}
_NONE_JUMPS = {"POP_JUMP_IF_NONE", "POP_JUMP_FORWARD_IF_NONE", "POP_JUMP_BACKWARD_IF_NONE"}
_NOT_NONE_JUMPS = {"POP_JUMP_IF_NOT_NONE", "POP_JUMP_FORWARD_IF_NOT_NONE",
                   "POP_JUMP_BACKWARD_IF_NOT_NONE"}
_HEADER_JUMPS = _FALSE_JUMPS | _TRUE_JUMPS | _NONE_JUMPS | _NOT_NONE_JUMPS
# Instructions that end a statement's control (so nothing after the wrapped body can
# fall into it) — makes leaving trailing statements un-wrapped equivalent to an ``else``.
_TERMINATORS = {"RETURN_VALUE", "RETURN_CONST", "RAISE_VARARGS", "RERAISE",
                "JUMP_FORWARD", "JUMP_BACKWARD", "JUMP_ABSOLUTE",
                "CONTINUE_LOOP", "BREAK_LOOP"}

_IF_CLASSES = {"IfThen", "IfElse", "IfJumpElse", "IfElseJump"}
_FLAT_CLASSES = {"BlockTemplate", "InstTemplate", "JumpTemplate", "NopTemplate"}


# CFT navigation helpers (use __dict__ to avoid ControlFlowTemplate.__getattr__,
# which recurses on nodes whose ``members`` attribute is absent, e.g. InstTemplate).
def _members(node):
    return node.__dict__.get("members")


def _get_member(node, key):
    m = _members(node)
    return m.get(key) if isinstance(m, dict) else None


def _blank_source(bc) -> list[str]:
    lines = [getattr(i, "starts_line", None) for i in _instructions(bc)]
    n = max([ln for ln in lines if ln is not None] + [0]) + 2
    return [""] * n


def _bc_to_cft(bc):
    try:
        from pylingual.control_flow_reconstruction.structure import bc_to_cft
    except Exception:  # noqa: BLE001 - pylingual missing / import side effects
        return None
    try:
        return bc_to_cft(bc, _blank_source(bc))
    except Exception:  # noqa: BLE001 - structuring can raise on odd bytecode
        return None


def _unwrap_body(cft):
    body = _get_member(cft, "body")
    return body if body is not None else cft


def _is_flat(cft) -> bool:
    node = _unwrap_body(cft)
    return type(node).__name__ in _FLAT_CLASSES


def _find_single_if(cft):
    node = _unwrap_body(cft)
    cls = type(node).__name__
    if cls in _IF_CLASSES:
        return node
    if cls == "BlockTemplate":
        members = _members(node) or []
        ifs = [m for m in members if type(m).__name__ in _IF_CLASSES]
        rest = [m for m in members if type(m).__name__ not in _IF_CLASSES]
        if len(ifs) == 1 and all(type(m).__name__ in _FLAT_CLASSES for m in rest):
            return ifs[0]
    return None


# Header (condition) recovery from GT bytecode.
def _last_statement_group(insts: list) -> list:
    groups: list[list] = []
    for i in insts:
        if i.opname in _SKIP_OPS:
            continue
        ln = getattr(i, "starts_line", None)
        if ln is not None and (not groups or getattr(groups[-1][0], "starts_line", None) != ln):
            groups.append([i])
        elif groups:
            groups[-1].append(i)
        else:
            groups.append([i])
    return groups[-1] if groups else []


def _const_repr(value) -> str | None:
    if isinstance(value, (str, bytes, int, float, bool, complex, type(None))):
        return repr(value)
    if isinstance(value, frozenset):
        return None
    if isinstance(value, tuple):
        parts = [_const_repr(v) for v in value]
        if any(p is None for p in parts):
            return None
        return "(" + ", ".join(parts) + ("," if len(parts) == 1 else "") + ")"
    return None


def _recover_if_condition(header_insts: list) -> str | None:
    group = _last_statement_group(header_insts)
    if not group:
        return None
    # split off the terminating header jump
    jump = group[-1]
    if jump.opname not in _HEADER_JUMPS:
        return None
    body = group[:-1]

    stack: list[str] = []
    for i in body:
        op = i.opname
        if op in _SKIP_OPS:
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
        elif op in ("LOAD_ATTR", "LOAD_METHOD"):
            if not stack:
                return None
            attr = _clean_name(getattr(i, "argrepr", ""))
            if not attr:
                return None
            stack[-1] = f"{stack[-1]}.{attr}"
        elif op == "COMPARE_OP":
            if len(stack) < 2:
                return None
            b, a = stack.pop(), stack.pop()
            stack.append(f"{a} {getattr(i, 'argrepr', '')} {b}")
        elif op == "IS_OP":
            if len(stack) < 2:
                return None
            b, a = stack.pop(), stack.pop()
            stack.append(f"{a} is not {b}" if getattr(i, "arg", 0) else f"{a} is {b}")
        elif op == "CONTAINS_OP":
            if len(stack) < 2:
                return None
            b, a = stack.pop(), stack.pop()
            stack.append(f"{a} not in {b}" if getattr(i, "arg", 0) else f"{a} in {b}")
        else:
            return None  # calls, arithmetic, boolean ops, etc. — not recoverable cleanly

    if len(stack) != 1:
        return None
    expr = stack[0]

    jop = jump.opname
    if jop in _FALSE_JUMPS:
        cond = expr
    elif jop in _NOT_NONE_JUMPS:
        cond = f"{expr} is None"
    elif jop in _NONE_JUMPS:
        cond = f"{expr} is not None"
    else:  # _TRUE_JUMPS — negated; inversion is error-prone, defer.
        return None

    try:
        ast.parse(cond, mode="eval")
    except (SyntaxError, ValueError):
        return None
    return cond


# Body identification: match the GT if-body signature in the derived bytecode.
def _signature(insts: list) -> list[tuple]:
    out = []
    for i in insts:
        op = i.opname
        if op in _SKIP_OPS:
            continue
        key = None
        if op in _NAME_LOADS or op in _STORE_OPS or op in ("LOAD_ATTR", "LOAD_METHOD",
                                                           "IMPORT_FROM", "IMPORT_NAME"):
            key = _clean_name(getattr(i, "argrepr", ""))
        elif op == "LOAD_CONST":
            try:
                key = repr(getattr(i, "argval", None))
            except Exception:  # noqa: BLE001
                key = None
        out.append((op, key))
    return out


def _unique_contiguous_match(haystack: list[tuple], needle: list[tuple]) -> int | None:
    if not needle or len(needle) > len(haystack):
        return None
    found = None
    n = len(needle)
    for s in range(len(haystack) - n + 1):
        if haystack[s:s + n] == needle:
            if found is not None:
                return None  # ambiguous
            found = s
    return found


def _statement_groups(bc) -> list[list]:
    groups: list[list] = []
    for i in _instructions(bc):
        if i.opname in _SKIP_OPS:
            continue
        ln = getattr(i, "starts_line", None)
        if ln is not None and (not groups or getattr(groups[-1][0], "starts_line", None) != ln):
            groups.append([i])
        elif groups:
            groups[-1].append(i)
        else:
            groups.append([i])
    return groups


def _covered_group_ordinals(groups: list[list], matched_offsets: set[int]) -> list[int]:
    out = []
    for idx, grp in enumerate(groups):
        offs = {getattr(i, "offset", None) for i in grp}
        if offs & matched_offsets:
            out.append(idx)
    return out


def _locate_body(tree: ast.AST):
    top = tree.body
    if len(top) == 1 and isinstance(top[0], (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        return top[0].body, top[0]
    return tree.body, tree


# Operator entry point.
def control_flow_candidate(gt_code_object, derived_code_object, fragment: str,
                           repair_context: dict | None) -> dict | None:
    try:
        return _control_flow_candidate(gt_code_object, derived_code_object, fragment, repair_context)
    except Exception:  # noqa: BLE001 - a bad candidate is rejected by the oracle gate; a
        # raised exception would crash the repair loop, so swallow everything.
        return None


def _control_flow_candidate(gt_code_object, derived_code_object, fragment: str, repair_context: dict | None) -> dict | None:
    if not repair_context:
        return None
    failed = repair_context.get("pylingual_failed_result") or {}
    if failed.get("message") != "Different control flow":
        return None

    gt_insts = _instructions(gt_code_object)
    der_insts = _instructions(derived_code_object)
    if not gt_insts or not der_insts:
        return None

    # 1. Structure both objects; require a clean single-if GT and a flat derived.
    gt_cft = _bc_to_cft(gt_code_object)
    der_cft = _bc_to_cft(derived_code_object)
    if gt_cft is None or der_cft is None:
        return None
    if not _is_flat(der_cft):
        return None
    if_node = _find_single_if(gt_cft)
    if if_node is None:
        return None
    header = _get_member(if_node, "if_header")
    if_body = _get_member(if_node, "if_body")
    if header is None or if_body is None:
        return None

    # 2. Recover the condition text verbatim from GT bytecode.
    try:
        header_insts = header.get_instructions()
        body_insts = if_body.get_instructions()
    except Exception:  # noqa: BLE001
        return None
    cond = _recover_if_condition(header_insts)
    if cond is None or not body_insts:
        return None

    # 3. Match the if-body signature as a UNIQUE contiguous run in the derived stream.
    needle = _signature(body_insts)
    haystack_sig = _signature(der_insts)
    start = _unique_contiguous_match(haystack_sig, needle)
    if start is None:
        return None

    # Recover the matched derived instruction offsets, then the covered statement
    # groups (ordinals), which map to derived AST statement ordinals.
    der_nonskip = [i for i in der_insts if i.opname not in _SKIP_OPS]
    matched = der_nonskip[start:start + len(needle)]
    matched_offsets = {getattr(i, "offset", None) for i in matched}
    groups = _statement_groups(derived_code_object)
    covered = _covered_group_ordinals(groups, matched_offsets)
    if not covered or covered != list(range(covered[0], covered[-1] + 1)):
        return None
    lo, hi = covered[0], covered[-1]

    # 4. Parse the fragment and wrap the corresponding statements.
    try:
        dedented = textwrap.dedent(fragment)
        tree = ast.parse(dedented)
    except SyntaxError:
        return None
    stmts, container = _locate_body(tree)
    if hi >= len(stmts) or lo < 0:
        return None
    # Soundness: if any statement TRAILS the wrapped run, leaving it un-wrapped is only
    # equivalent to a GT ``else`` when the wrapped body cannot fall through into it —
    # i.e. its last real instruction terminates control. Otherwise a true ``else`` may
    # be involved; defer rather than risk a wrong re-wrap.
    if hi < len(stmts) - 1:
        real = [i for i in body_insts if i.opname not in _SKIP_OPS]
        if not real or real[-1].opname not in _TERMINATORS:
            return None

    wrap = stmts[lo:hi + 1]
    try:
        test_node = ast.parse(cond, mode="eval").body
    except (SyntaxError, ValueError):
        return None
    if_stmt = ast.If(test=test_node, body=list(wrap), orelse=[])
    new_stmts = stmts[:lo] + [if_stmt] + stmts[hi + 1:]
    if isinstance(container, ast.Module):
        container.body = new_stmts
    else:
        container.body = new_stmts
    ast.fix_missing_locations(tree)

    try:
        unparsed = ast.unparse(tree)
    except Exception:  # noqa: BLE001
        return None
    reindented = _match_indent_convention(fragment, unparsed)
    if reindented is None or reindented.strip() == fragment.strip():
        return None
    # Never emit a fragment that doesn't parse.
    try:
        ast.parse(textwrap.dedent(reindented))
    except SyntaxError:
        return None

    n = hi - lo + 1
    return {
        "text": reindented,
        "operator": f"wrap {n} statement{'s' if n != 1 else ''} in dropped `if {cond}:` from GT",
        "opname": "POP_JUMP_IF_FALSE",
        "kind": "control_flow",
        "confidence": "unique",
    }


def _member_items(node):
    m = _members(node)
    if isinstance(m, dict):
        return list(m.items())
    if isinstance(m, list):
        return list(enumerate(m))
    return []
