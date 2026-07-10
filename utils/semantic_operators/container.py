"""Container-construction operator: regenerate an all-CONSTANT container literal
(dict / set / list) from ground-truth bytecode when the decompiler mangled the display.

Some decompiled containers come out structurally wrong — a flat literal rendered as a
nested one, a key/value pair shifted by one, an element dropped — even though GT builds
the whole thing from constants. When that is the case the correct value is recoverable
*deterministically*: simulate GT's instruction stream as a constant-only stack machine,
recover the exact container it builds, and splice ``repr(value)`` back over the mangled
display. No LLM, no guessing — the value is read verbatim from GT bytecode and re-emitted
in GT's construction order.

Conservative and unique-or-defer:
  * GT must build EXACTLY ONE fully-constant, non-empty container of the target kind
    (dict, then set, then list — the first kind that qualifies wins).
  * The derived fragment must contain EXACTLY ONE *top-level* display of that kind; a
    nested display of the same kind (e.g. the mangled ``{...: {...}}``) is subsumed by
    replacing the outermost one, so it does not count as a second display.
  * The recovered value must round-trip through ``repr()`` -> ``ast.literal_eval``.
Anything ambiguous or non-constant defers (returns None). Insertion-time recompilation
plus the oracle distance gate are the final backstop; this operator never emits a value
it did not read from GT, and it never raises.
"""
from __future__ import annotations

import ast
import textwrap

from utils.semantic_operators.leaf import _instructions

# Sentinel for a stack value the constant-only machine cannot prove constant. Any
# container that consumes it is TAINTED (never recorded as fully constant).
_NONCONST = object()

_STORE_OPCODES = {"STORE_NAME", "STORE_FAST", "STORE_GLOBAL", "STORE_DEREF"}
_FINALIZE_OPCODES = _STORE_OPCODES | {"POP_TOP", "RETURN_VALUE", "YIELD_VALUE"}
_KINDS = ("dict", "set", "list")

# ast display class per collectable kind.
_DISPLAY_CLS = {"dict": ast.Dict, "set": ast.Set, "list": ast.List}


class _Container:
    """A mutable display (list/set/dict) under construction on the simulated stack."""
    __slots__ = ("kind", "value", "tainted")

    def __init__(self, kind, value, tainted=False):
        self.kind = kind          # "list" | "set" | "dict"
        self.value = value        # the actual Python list/set/dict built so far
        self.tainted = tainted    # True once a non-constant element has flowed in


def _argint(ins):
    """The integer operand of an instruction (BUILD_* count, *_ADD/*_UPDATE depth), or
    0 when the instruction carries no usable int operand."""
    a = getattr(ins, "arg", None)
    if isinstance(a, int):
        return a
    v = getattr(ins, "argval", None)
    return v if isinstance(v, int) else 0


def _as_element(entry):
    """Resolve a popped stack entry into ``(value, tainted)`` for use as a container
    element. ``value`` is meaningful only when ``tainted`` is False."""
    if entry is _NONCONST:
        return None, True
    if isinstance(entry, _Container):
        return entry.value, entry.tainted
    return entry, False


def _recover_constant_containers(insts: list) -> list[tuple[str, object]]:
    """Simulate GT's instruction stream as a constant-only stack machine and return the
    list of ``(kind, value)`` for every fully-constant, non-empty container of kind
    dict/set/list that GT finalizes (stores / returns / drops as an expression statement).

    A container consumed as an *element* of, or *merged into*, another container is folded
    into its parent rather than reported separately — so a flat dict assembled from a
    BUILD_MAP plus a merged BUILD_CONST_KEY_MAP is reported as ONE dict. Never raises."""
    stack: list = []
    results: list[tuple[str, object]] = []

    def pop():
        return stack.pop() if stack else _NONCONST

    def target_at(depth):
        return stack[-depth] if (depth >= 1 and len(stack) >= depth) else None

    def finalize(entry):
        if isinstance(entry, _Container) and not entry.tainted:
            if entry.kind in _KINDS and len(entry.value) > 0:
                results.append((entry.kind, entry.value))

    for ins in insts:
        op = getattr(ins, "opname", None)
        if op is None or op == "CACHE":
            continue
        n = _argint(ins)

        # --- control flow: any branch crossing a live container's construction poisons
        # it (a conditional element is not a constant), mirroring placeholder.py's guard.
        if getattr(ins, "is_jump", False) or "JUMP" in op or op in ("FOR_ITER", "SEND"):
            for e in stack:
                if isinstance(e, _Container):
                    e.tainted = True
            if op.startswith("POP_JUMP"):
                pop()
            continue

        if op == "LOAD_CONST":
            # a const load that is a branch-merge target is a conditional result, not a
            # plain literal -> refuse to treat it as constant.
            stack.append(_NONCONST if getattr(ins, "is_jump_target", False) else getattr(ins, "argval", None))
            continue

        if op in _FINALIZE_OPCODES:
            finalize(pop())
            continue
        if op == "RETURN_CONST":
            continue  # const returned directly; nothing on the stack to finalize

        # --- container builders --------------------------------------------------------
        if op == "BUILD_TUPLE":
            elems = [pop() for _ in range(n)][::-1]
            vals, tainted = [], False
            for e in elems:
                v, t = _as_element(e)
                tainted = tainted or t
                vals.append(v)
            stack.append(_NONCONST if tainted else tuple(vals))
            continue
        if op in ("BUILD_LIST", "BUILD_SET", "BUILD_MAP", "BUILD_CONST_KEY_MAP"):
            stack.append(_build_map_like(op, n, pop))
            continue
        if op in ("LIST_EXTEND", "SET_UPDATE", "DICT_UPDATE", "DICT_MERGE"):
            src = pop()
            _merge_into(op, target_at(n), src)
            continue
        if op == "MAP_ADD":
            value = pop()
            key = pop()
            _map_add(target_at(n), key, value)
            continue
        if op in ("LIST_APPEND", "SET_ADD"):
            value = pop()
            _seq_add(op, target_at(n), value)
            continue

        # --- calls: consume the arguments (so a constant argument cannot be mistaken for
        # a following container element) and yield a non-constant result.
        if op in ("CALL", "CALL_KW", "CALL_METHOD"):
            for _ in range(min(len(stack), n + 2)):
                stack.pop()
            stack.append(_NONCONST)
            continue
        if op in ("CALL_FUNCTION", "CALL_FUNCTION_KW"):
            for _ in range(min(len(stack), n + 1)):
                stack.pop()
            stack.append(_NONCONST)
            continue
        if op.startswith("CALL"):
            for _ in range(min(len(stack), 2)):
                stack.pop()
            stack.append(_NONCONST)
            continue
        if op == "IMPORT_NAME":
            pop(); pop()
            stack.append(_NONCONST)
            continue
        if op == "BUILD_STRING":
            for _ in range(min(len(stack), n)):
                stack.pop()
            stack.append(_NONCONST)
            continue
        if op == "BUILD_SLICE":
            for _ in range(min(len(stack), n or 2)):
                stack.pop()
            stack.append(_NONCONST)
            continue

        # --- default: value-producing loads push a non-constant; anything else is left
        # alone (any residue is harmless — container ops act on the top of stack only).
        if op.startswith("LOAD_") or op in ("PUSH_NULL", "IMPORT_FROM", "COPY"):
            stack.append(_NONCONST)

    return results


def _build_map_like(op: str, n: int, pop) -> object:
    """Build a list/set/dict (or dict-from-const-keys) from ``n`` popped entries; return a
    ``_Container`` (tainted if any element is non-constant / unhashable)."""
    if op == "BUILD_CONST_KEY_MAP":
        keys_entry = pop()
        keys, ktaint = _as_element(keys_entry)
        vals = [pop() for _ in range(n)][::-1]
        value, tainted = {}, ktaint or not isinstance(keys, tuple) or len(keys) != n
        if not tainted:
            for k, v_entry in zip(keys, vals):
                v, t = _as_element(v_entry)
                tainted = tainted or t
                try:
                    value[k] = v
                except Exception:  # noqa: BLE001 - unhashable key etc.
                    tainted = True
        return _Container("dict", value, tainted)

    raw = [pop() for _ in range(2 * n if op == "BUILD_MAP" else n)][::-1]
    if op == "BUILD_LIST":
        vals, tainted = _resolve_all(raw)
        return _Container("list", list(vals), tainted)
    if op == "BUILD_SET":
        vals, tainted = _resolve_all(raw)
        value = set()
        for v in vals:
            try:
                value.add(v)
            except Exception:  # noqa: BLE001 - unhashable element
                tainted = True
        return _Container("set", value, tainted)
    # BUILD_MAP: raw is [k1, v1, k2, v2, ...]
    value, tainted = {}, False
    for i in range(0, len(raw) - 1, 2):
        k, kt = _as_element(raw[i])
        v, vt = _as_element(raw[i + 1])
        tainted = tainted or kt or vt
        try:
            value[k] = v
        except Exception:  # noqa: BLE001 - unhashable key
            tainted = True
    return _Container("dict", value, tainted)


def _resolve_all(entries) -> tuple[list, bool]:
    vals, tainted = [], False
    for e in entries:
        v, t = _as_element(e)
        tainted = tainted or t
        vals.append(v)
    return vals, tainted


def _merge_into(op: str, target, src) -> None:
    """Apply LIST_EXTEND / SET_UPDATE / DICT_UPDATE / DICT_MERGE of ``src`` into
    ``target`` (a ``_Container``), tainting the target if ``src`` is not constant."""
    if not isinstance(target, _Container):
        return
    value, tainted = _as_element(src)
    if tainted:
        target.tainted = True
        return
    try:
        if op == "LIST_EXTEND":
            target.value.extend(list(value))
        elif op == "SET_UPDATE":
            target.value.update(set(value))
        else:  # DICT_UPDATE / DICT_MERGE
            target.value.update(dict(value))
    except Exception:  # noqa: BLE001
        target.tainted = True


def _map_add(target, key, value) -> None:
    if not isinstance(target, _Container):
        return
    k, kt = _as_element(key)
    v, vt = _as_element(value)
    if kt or vt:
        target.tainted = True
        return
    try:
        target.value[k] = v
    except Exception:  # noqa: BLE001 - unhashable key
        target.tainted = True


def _seq_add(op: str, target, value) -> None:
    if not isinstance(target, _Container):
        return
    v, t = _as_element(value)
    if t:
        target.tainted = True
        return
    try:
        if op == "LIST_APPEND":
            target.value.append(v)
        else:  # SET_ADD
            target.value.add(v)
    except Exception:  # noqa: BLE001
        target.tainted = True


def _round_trips(value) -> bool:
    """True iff ``ast.literal_eval(repr(value)) == value`` — safe to re-emit as a literal."""
    try:
        return ast.literal_eval(repr(value)) == value
    except Exception:  # noqa: BLE001
        return False


def _top_level_displays(tree: ast.AST, node_cls) -> list:
    """The display nodes of ``node_cls`` that are NOT nested inside another display of the
    same class (the outermost ones). Replacing an outermost display subsumes any same-kind
    display nested within it, so nested displays are not counted as separate matches."""
    same = [n for n in ast.walk(tree) if isinstance(n, node_cls)]
    nested_ids: set[int] = set()
    for m in same:
        for d in ast.walk(m):
            if d is not m and isinstance(d, node_cls):
                nested_ids.add(id(d))
    return [n for n in same if id(n) not in nested_ids]


def _replace_span(fragment: str, dedented: str, node: ast.AST, new_text: str) -> str | None:
    """Replace a (possibly multi-line) node's source span with ``new_text``, mapping the
    node's dedented positions back into the original indented ``fragment``."""
    if None in (node.lineno, node.col_offset, node.end_lineno, node.end_col_offset):
        return None
    orig = fragment.splitlines(keepends=True)
    ded = dedented.splitlines(keepends=True)
    li0, li1 = node.lineno - 1, node.end_lineno - 1
    if li0 < 0 or li1 >= len(orig) or li0 >= len(ded) or li1 >= len(ded) or li0 > li1:
        return None
    # col_offset is a utf-8 byte offset; char slicing is only safe on ASCII boundary lines.
    if not orig[li0].isascii() or not orig[li1].isascii():
        return None
    start = node.col_offset + (len(orig[li0]) - len(ded[li0]))
    end = node.end_col_offset + (len(orig[li1]) - len(ded[li1]))
    prefix = orig[li0][:start]
    suffix = orig[li1][end:]
    return "".join(orig[:li0]) + prefix + new_text + suffix + "".join(orig[li1 + 1:])


def container_construction_candidate(gt_code_object, derived_code_object, fragment: str,
                                     repair_context: dict | None) -> dict | None:
    """Regenerate an all-constant dict/set/list literal from GT bytecode over the mangled
    derived display. Returns ``{"text", "operator", "opname", "kind", "confidence"}`` or
    None to defer. Never raises."""
    try:
        if not repair_context:
            return None
        failed = repair_context.get("pylingual_failed_result") or {}
        if failed.get("message") != "Different bytecode":
            return None

        gt = _instructions(gt_code_object)
        if not gt:
            return None
        recovered = _recover_constant_containers(gt)
        if not recovered:
            return None

        try:
            dedented = textwrap.dedent(fragment)
            tree = ast.parse(dedented)
        except SyntaxError:
            return None

        # dict, then set, then list: the first kind for which GT builds exactly one
        # significant constant container AND the fragment has exactly one top-level display.
        for kind in _KINDS:
            gt_containers = [v for (k, v) in recovered if k == kind]
            if len(gt_containers) != 1:
                continue
            container = gt_containers[0]
            displays = _top_level_displays(tree, _DISPLAY_CLS[kind])
            if len(displays) != 1:
                continue
            if not _round_trips(container):
                continue

            new_text = repr(container)
            # sets: repr renders `{...}`; guard the (already-excluded) empty-set case.
            if kind == "set" and not container:
                new_text = "set()"
            new_fragment = _replace_span(fragment, dedented, displays[0], new_text)
            if new_fragment is None or new_fragment == fragment:
                return None
            try:
                ast.parse(textwrap.dedent(new_fragment))
            except SyntaxError:
                return None
            return {
                "text": new_fragment,
                "operator": f"regenerate {kind} literal ({len(container)} items) from GT",
                "opname": "BUILD_CONST_KEY_MAP",
                "kind": "container_construction",
                "confidence": "unique",
            }
        return None
    except Exception:  # noqa: BLE001 - a bad candidate is rejected by the oracle gate; a
        # raised exception would crash the repair loop, so swallow everything.
        return None
