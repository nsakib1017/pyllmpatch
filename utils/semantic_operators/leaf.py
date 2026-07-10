"""Leaf-value operators: deterministic, no-LLM source fixes for the "Different
bytecode" regime, read straight from ground-truth bytecode.

The decompiler often gets the *structure* right but emits a wrong **operand** at
one instruction: a binary/comparison operator, or a literal constant. These
surface as a candidate instruction whose OPCODE matches GT but whose operand
differs — GT's ``argrepr``/``argval`` gives the correct value directly. We
localize the operand in the *source* by AST (not ``co_positions()``, which does
not align 1:1 with pylingual's instruction stream), find the unique — or the
``ordinal``-th, by execution order — matching node, and swap just that token.

Covered opcode families (same-opcode operand swap):
  * ``BINARY_OP``            -> ast.BinOp    (arithmetic/bitwise operator)
  * ``COMPARE_OP``           -> ast.Compare  (comparison operator)
  * ``IS_OP``/``CONTAINS_OP``      -> ast.Compare  (is/is not, in/not in)
  * ``LOAD_CONST``/``RETURN_CONST`` -> ast.Constant (literal value)
  * ``LOAD_FAST``/``LOAD_GLOBAL``/``LOAD_NAME``/``LOAD_DEREF`` -> ast.Name (unique load)
  * ``LOAD_ATTR``/``LOAD_METHOD``   -> ast.Attribute (unique attribute read)
  * ``FORMAT_VALUE``         -> ast.FormattedValue (f-string !r/!s/!a conversion)

A DIFFERENT opcode at the divergence (e.g. ``LOAD_NAME`` vs ``LOAD_CONST``, or a
jump-sense flip) is not an operand swap and is deferred to the structural/LLM
stages. Every candidate still passes the loop's compile + oracle accept gate, so
a wrong guess cannot land.
"""
from __future__ import annotations

import ast
import re
import textwrap
from keyword import iskeyword

# ---- operator symbol <-> AST node class -----------------------------------

_AST_BINOP_SYMBOL: dict[type, str] = {
    ast.Add: "+", ast.Sub: "-", ast.Mult: "*", ast.Div: "/",
    ast.FloorDiv: "//", ast.Mod: "%", ast.Pow: "**",
    ast.LShift: "<<", ast.RShift: ">>", ast.BitOr: "|",
    ast.BitAnd: "&", ast.BitXor: "^", ast.MatMult: "@",
}
_SYMBOL_TO_AST_BINOP: dict[str, type] = {v: k for k, v in _AST_BINOP_SYMBOL.items()}

# Pre-3.11 CPython encodes the binary operator in the OPCODE NAME (BINARY_ADD,
# BINARY_SUBTRACT, ...) rather than in argrepr like 3.11+'s unified BINARY_OP. Map each
# 3.10 binary opcode to its source symbol so the operand-swap operators work on 3.10 too
# (a 3.10 operator swap surfaces as a *different opname*, not a same-opname argrepr diff).
# INPLACE_* (augmented assignment `x += y`) is a different AST shape (ast.AugAssign) and is
# intentionally NOT mapped here — the BinOp-based swap can't apply it, so it would only defer.
_OPNAME_TO_BINOP_SYMBOL: dict[str, str] = {
    "BINARY_ADD": "+", "BINARY_SUBTRACT": "-", "BINARY_MULTIPLY": "*",
    "BINARY_TRUE_DIVIDE": "/", "BINARY_FLOOR_DIVIDE": "//", "BINARY_MODULO": "%",
    "BINARY_POWER": "**", "BINARY_LSHIFT": "<<", "BINARY_RSHIFT": ">>",
    "BINARY_AND": "&", "BINARY_OR": "|", "BINARY_XOR": "^",
    "BINARY_MATRIX_MULTIPLY": "@",
}

_SYMBOL_TO_AST_CMP: dict[str, type] = {
    "<": ast.Lt, ">": ast.Gt, "==": ast.Eq, "!=": ast.NotEq,
    "<=": ast.LtE, ">=": ast.GtE,
}

# IS_OP / CONTAINS_OP identity + membership operators. argrepr gives the exact
# surface form ("is", "is not", "in", "not in"); swapping is a same-family word-op
# token edit in the operand gap, reusing the COMPARE_OP machinery.
_SYMBOL_TO_AST_ISCONTAINS: dict[str, type] = {
    "is": ast.Is, "is not": ast.IsNot, "in": ast.In, "not in": ast.NotIn,
}
# conversion int on ast.FormattedValue <-> FORMAT_VALUE argrepr surface form.
_FORMAT_CONV_FROM_ARGREPR: dict[str, int] = {"": -1, "!s": 115, "!r": 114, "!a": 97}

_OPERATOR_OPCODES = {"BINARY_OP"}
_CONST_OPCODES = {"LOAD_CONST", "RETURN_CONST"}
_NAME_OPCODES = {"LOAD_FAST", "LOAD_GLOBAL", "LOAD_NAME", "LOAD_DEREF"}  # value loads (Load ctx)
# only LOAD_FAST is safe for a whole-scope consistent rename: FAST locals are private
# to one code object. LOAD_DEREF (closure) is shared with nested code objects, and
# LOAD_GLOBAL/LOAD_NAME reference module-level bindings defined elsewhere — renaming
# any of those from a single fragment would desync the other object(s).
_RENAMABLE_LOCAL_OPCODES = {"LOAD_FAST"}
_ATTR_OPCODES = {"LOAD_ATTR", "LOAD_METHOD"}
_SIMPLE_LITERAL_TYPES = (type(None), bool, int, float, str, bytes)
_MISSING = object()  # sentinel so a genuine `argval is None` is distinct from a missing attr


def _instructions(bytecode) -> list:
    try:
        return list(bytecode)
    except Exception:  # noqa: BLE001
        return []


# =========================================================================
# BINARY_OP (kept as standalone functions for existing callers/tests)
# =========================================================================

def extract_operator_divergence(gt_code_object, derived_code_object, failed_offset: int) -> dict | None:
    """Same-opcode BINARY_OP mismatch -> ``{opname, wrong, right, ordinal}`` or None."""
    der = _instructions(derived_code_object)
    gt = _instructions(gt_code_object)
    idx = next((i for i, ins in enumerate(der) if getattr(ins, "offset", None) == failed_offset), None)
    if idx is None or idx >= len(gt):
        return None
    cand, gt_inst = der[idx], gt[idx]
    cop = getattr(cand, "opname", None)
    gop = getattr(gt_inst, "opname", None)
    # Pre-3.11 binary operators live in the opcode NAME (BINARY_ADD vs BINARY_SUBTRACT);
    # a 3.10 swap surfaces as different opnames — read the symbol from the opname map.
    csym = _OPNAME_TO_BINOP_SYMBOL.get(cop)
    gsym = _OPNAME_TO_BINOP_SYMBOL.get(gop)
    if csym and gsym and csym != gsym:
        ordinal = sum(1 for ins in der
                      if _OPNAME_TO_BINOP_SYMBOL.get(getattr(ins, "opname", "")) == csym
                      and getattr(ins, "offset", -1) < failed_offset)
        return {"opname": cop, "wrong": csym, "right": gsym, "ordinal": ordinal}
    if cop != gop:
        return None
    if cop not in _OPERATOR_OPCODES:
        return None
    wrong = (getattr(cand, "argrepr", "") or "").strip()
    right = (getattr(gt_inst, "argrepr", "") or "").strip()
    if not wrong or not right or wrong == right:
        return None
    if wrong not in _SYMBOL_TO_AST_BINOP or right not in _SYMBOL_TO_AST_BINOP:
        return None
    ordinal = sum(
        1 for ins in der
        if ins.opname in _OPERATOR_OPCODES
        and (getattr(ins, "argrepr", "") or "").strip() == wrong
        and getattr(ins, "offset", -1) < failed_offset
    )
    return {"opname": cand.opname, "wrong": wrong, "right": right, "ordinal": ordinal}


def _eval_order_binops(node: ast.AST, wrong_cls: type, out: list) -> None:
    if isinstance(node, ast.BinOp):
        _eval_order_binops(node.left, wrong_cls, out)
        _eval_order_binops(node.right, wrong_cls, out)
        if isinstance(node.op, wrong_cls):
            out.append(node)
        return
    for child in ast.iter_child_nodes(node):
        _eval_order_binops(child, wrong_cls, out)


def _swap_token_in_gap(fragment: str, dedented: str, left_end_lineno: int, left_end_col: int,
                       right_lineno: int, right_col: int, wrong: str, right: str) -> str | None:
    """Replace the ``wrong`` token located between two operand nodes with ``right``,
    mapping dedented AST columns back onto the original (indented) fragment."""
    if None in (left_end_lineno, left_end_col, right_lineno, right_col):
        return None
    if left_end_lineno != right_lineno:
        return None
    orig_lines = fragment.splitlines(keepends=True)
    ded_lines = dedented.splitlines(keepends=True)
    li = left_end_lineno - 1
    if li >= len(orig_lines) or li >= len(ded_lines):
        return None
    line = orig_lines[li]
    if not line.isascii():
        return None
    delta = len(orig_lines[li]) - len(ded_lines[li])
    gap_start = left_end_col + delta
    gap_end = right_col + delta
    gap = line[gap_start:gap_end]
    pos = gap.find(wrong)
    if pos < 0:
        return None
    abs_pos = gap_start + pos
    orig_lines[li] = line[:abs_pos] + right + line[abs_pos + len(wrong):]
    return "".join(orig_lines)


def _apply_operator_swap(fragment: str, dedented: str, node: ast.BinOp, wrong: str, right: str) -> str | None:
    return _swap_token_in_gap(fragment, dedented, node.left.end_lineno, node.left.end_col_offset,
                              node.right.lineno, node.right.col_offset, wrong, right)


def _pick(nodes: list, ordinal: int):
    """Choose the unique node, or the ordinal-th, returning (node, confidence)."""
    if not nodes:
        return None, None
    if len(nodes) == 1:
        return nodes[0], "unique"
    if 0 <= ordinal < len(nodes):
        return nodes[ordinal], "ordinal"
    return None, None


def swap_binary_operator(fragment: str, wrong: str, right: str, ordinal: int = 0) -> tuple[str, str] | None:
    wrong_cls = _SYMBOL_TO_AST_BINOP.get(wrong)
    if wrong_cls is None or right not in _SYMBOL_TO_AST_BINOP:
        return None
    dedented = textwrap.dedent(fragment)
    try:
        tree = ast.parse(dedented)
    except SyntaxError:
        return None
    nodes: list = []
    _eval_order_binops(tree, wrong_cls, nodes)
    node, confidence = _pick(nodes, ordinal)
    if node is None:
        return None
    if confidence == "ordinal" and _has_comprehension(tree):
        return None  # comprehension bytecode order != AST order -> ordinal unreliable
    edited = _apply_operator_swap(fragment, dedented, node, wrong, right)
    if edited is None or edited == fragment:
        return None
    return edited, confidence


def swap_unique_binary_operator(fragment: str, wrong: str, right: str) -> str | None:
    result = swap_binary_operator(fragment, wrong, right, ordinal=0)
    if result is None:
        return None
    edited, confidence = result
    return edited if confidence == "unique" else None


# =========================================================================
# COMPARE_OP (comparison operator swap)
# =========================================================================

def _eval_order_compare_ops(node: ast.AST, wrong_cls: type, out: list) -> None:
    """Collect (Compare node, op-index) pairs whose ops[i] is wrong_cls in true
    BYTECODE evaluation order. A Compare emits ops[i] only after its left operand and
    comparators[0..i] have been evaluated, and any nested comparison inside a
    comparator executes before the enclosing op. A naive post-order walk places ALL of
    a node's own ops after every nested op, so a chained comparison with a nested
    same-family compare in a non-first operand (e.g. ``a is b is (c is d)``) would
    misalign the bytecode-derived ``ordinal`` against these hits and edit the wrong
    (but still compiling) operator. Interleaving each op after its comparator recursion
    reproduces the real execution order and keeps the ordinal mapping correct."""
    if isinstance(node, ast.Compare):
        _eval_order_compare_ops(node.left, wrong_cls, out)
        for i, comparator in enumerate(node.comparators):
            _eval_order_compare_ops(comparator, wrong_cls, out)
            if isinstance(node.ops[i], wrong_cls):
                out.append((node, i))
        return
    for child in ast.iter_child_nodes(node):
        _eval_order_compare_ops(child, wrong_cls, out)


def swap_compare_operator(fragment: str, wrong: str, right: str, ordinal: int = 0) -> tuple[str, str] | None:
    wrong_cls = _SYMBOL_TO_AST_CMP.get(wrong)
    if wrong_cls is None or right not in _SYMBOL_TO_AST_CMP:
        return None
    dedented = textwrap.dedent(fragment)
    try:
        tree = ast.parse(dedented)
    except SyntaxError:
        return None
    hits: list = []
    _eval_order_compare_ops(tree, wrong_cls, hits)
    hit, confidence = _pick(hits, ordinal)
    if hit is None:
        return None
    if confidence == "ordinal" and _has_comprehension(tree):
        return None  # comprehension bytecode order != AST order -> ordinal unreliable
    node, opidx = hit
    left_node = node.left if opidx == 0 else node.comparators[opidx - 1]
    right_node = node.comparators[opidx]
    edited = _swap_token_in_gap(fragment, dedented, left_node.end_lineno, left_node.end_col_offset,
                                right_node.lineno, right_node.col_offset, wrong, right)
    if edited is None or edited == fragment:
        return None
    return edited, confidence


# =========================================================================
# IS_OP / CONTAINS_OP (identity / membership operator swap)
# =========================================================================

def swap_is_contains(fragment: str, wrong: str, right: str, ordinal: int = 0) -> tuple[str, str] | None:
    """Swap ``is``<->``is not`` / ``in``<->``not in`` in the operand gap. Reuses the
    COMPARE_OP node-collection + gap-token machinery; ``wrong`` and ``right`` are the
    surface forms from IS_OP/CONTAINS_OP argrepr, and must be same-family."""
    wrong_cls = _SYMBOL_TO_AST_ISCONTAINS.get(wrong)
    right_cls = _SYMBOL_TO_AST_ISCONTAINS.get(right)
    if wrong_cls is None or right_cls is None:
        return None
    # same family only: identity (Is/IsNot) or membership (In/NotIn); never cross
    if {wrong_cls, right_cls} not in ({ast.Is, ast.IsNot}, {ast.In, ast.NotIn}):
        return None
    dedented = textwrap.dedent(fragment)
    try:
        tree = ast.parse(dedented)
    except SyntaxError:
        return None
    hits: list = []
    _eval_order_compare_ops(tree, wrong_cls, hits)
    hit, confidence = _pick(hits, ordinal)
    if hit is None:
        return None
    if confidence == "ordinal" and _has_comprehension(tree):
        return None
    node, opidx = hit
    left_node = node.left if opidx == 0 else node.comparators[opidx - 1]
    right_node = node.comparators[opidx]
    edited = _swap_token_in_gap(fragment, dedented, left_node.end_lineno, left_node.end_col_offset,
                                right_node.lineno, right_node.col_offset, wrong, right)
    if edited is None or edited == fragment:
        return None
    return edited, confidence


# =========================================================================
# FORMAT_VALUE (f-string conversion flag: {x} <-> {x!r}/{x!s}/{x!a})
# =========================================================================

def _looks_like_fstring(span: str) -> bool:
    m = re.match(r'^\s*([A-Za-z]{0,3})(["\'])', span)
    return bool(m) and "f" in m.group(1).lower()


def swap_format_value(fragment: str, wrong_conv: int, right_conv: int) -> tuple[str, str] | None:
    """Set the conversion on the single f-string field whose current conversion is
    ``wrong_conv``, to ``right_conv``. Fires only when exactly one field matches (no
    fragile f-string ordinal), the enclosing JoinedStr is single-line with a verifiable
    f-string span, and the re-rendered result re-parses. Correctness is still gated by
    the loop's compile + oracle accept."""
    dedented = textwrap.dedent(fragment)
    try:
        tree = ast.parse(dedented)
    except SyntaxError:
        return None
    targets = []
    for js in ast.walk(tree):
        if isinstance(js, ast.JoinedStr):
            for v in js.values:
                if isinstance(v, ast.FormattedValue) and v.conversion == wrong_conv:
                    targets.append((js, v))
    if len(targets) != 1:
        return None
    js, fv = targets[0]
    if None in (js.lineno, js.col_offset, js.end_lineno, js.end_col_offset):
        return None
    if js.lineno != js.end_lineno:  # multi-line f-string: span editing too fragile
        return None
    orig_lines = fragment.splitlines(keepends=True)
    ded_lines = dedented.splitlines(keepends=True)
    li = js.lineno - 1
    if li >= len(orig_lines) or li >= len(ded_lines):
        return None
    line = orig_lines[li]
    if not line.isascii():
        return None
    delta = len(orig_lines[li]) - len(ded_lines[li])
    start = js.col_offset + delta
    end = js.end_col_offset + delta
    if start < 0 or end > len(line) or not _looks_like_fstring(line[start:end]):
        return None
    fv.conversion = right_conv
    try:
        rendered = ast.unparse(js)
    except Exception:  # noqa: BLE001
        return None
    if not _looks_like_fstring(rendered):
        return None
    new_line = line[:start] + rendered + line[end:]
    new_fragment = "".join(orig_lines[:li] + [new_line] + orig_lines[li + 1:])
    if new_fragment == fragment:
        return None
    try:
        ast.parse(textwrap.dedent(new_fragment))
    except SyntaxError:
        return None
    return new_fragment, "unique"


# =========================================================================
# LOAD_CONST / RETURN_CONST (literal value swap)
# =========================================================================

def _const_match(value, target) -> bool:
    """Value-based match that keeps bool distinct from int and handles None."""
    if target is None:
        return value is None
    if isinstance(target, bool):
        return isinstance(value, bool) and value == target
    if isinstance(value, bool):
        return False
    if type(value) is not type(target):
        return False
    try:
        return value == target
    except Exception:  # noqa: BLE001
        return False


def _replace_node_span(fragment: str, dedented: str, node: ast.AST, new_text: str) -> str | None:
    """Replace a single-line node's full source span with new_text."""
    if None in (node.lineno, node.col_offset, node.end_lineno, node.end_col_offset):
        return None
    if node.lineno != node.end_lineno:
        return None
    orig_lines = fragment.splitlines(keepends=True)
    ded_lines = dedented.splitlines(keepends=True)
    li = node.lineno - 1
    if li >= len(orig_lines) or li >= len(ded_lines):
        return None
    line = orig_lines[li]
    if not line.isascii():
        return None
    delta = len(orig_lines[li]) - len(ded_lines[li])
    start = node.col_offset + delta
    end = node.end_col_offset + delta
    orig_lines[li] = line[:start] + new_text + line[end:]
    return "".join(orig_lines)


def _has_comprehension(tree: ast.AST) -> bool:
    return any(isinstance(n, (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp))
               for n in ast.walk(tree))


def _postorder_collect(node: ast.AST, predicate, out: list) -> None:
    """Collect nodes satisfying ``predicate`` in post-order (children before node),
    which approximates CPython evaluation order for value loads."""
    for child in ast.iter_child_nodes(node):
        _postorder_collect(child, predicate, out)
    if predicate(node):
        out.append(node)


def _collect_literal_nodes(node: ast.AST, out: list) -> None:
    """Yield (replace_node, effective_value) for ATOMIC literals in post-order,
    treating a folded unary-signed constant (``-1`` = UnaryOp(USub, Constant(1)))
    as one node with the signed value — so this enumeration matches the bytecode's
    one-LOAD_CONST-per-folded-literal stream (fixes the negative-literal ordinal skew)."""
    if (isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd))
            and isinstance(node.operand, ast.Constant)
            and isinstance(node.operand.value, (int, float, complex))
            and not isinstance(node.operand.value, bool)):
        value = -node.operand.value if isinstance(node.op, ast.USub) else node.operand.value
        out.append((node, value))
        return  # do not descend into the folded inner Constant
    # An all-constant tuple display folds to a SINGLE LOAD_CONST (constant tuple) in the
    # bytecode, so treat it as one tuple-valued literal node and do NOT descend into its
    # element Constants (they have no separate LOAD_CONST). Mirrors the unary-fold above.
    if (isinstance(node, ast.Tuple) and node.elts
            and all(isinstance(e, ast.Constant) for e in node.elts)):
        out.append((node, tuple(e.value for e in node.elts)))
        return
    for child in ast.iter_child_nodes(node):
        _collect_literal_nodes(child, out)
    if isinstance(node, ast.Constant):
        out.append((node, node.value))


def swap_literal(fragment: str, wrong_value, right_value, ordinal: int = 0) -> tuple[str, str] | None:
    if not _round_trips(right_value):
        return None  # inf/nan/unrepresentable repr() would compile to a NAME, not a literal
    dedented = textwrap.dedent(fragment)
    try:
        tree = ast.parse(dedented)
    except SyntaxError:
        return None
    lits: list = []
    _collect_literal_nodes(tree, lits)
    nodes = [n for n, value in lits if _const_match(value, wrong_value)]
    node, confidence = _pick(nodes, ordinal)
    if node is None:
        return None
    if confidence == "ordinal" and _has_comprehension(tree):
        return None  # comprehension bytecode order != AST order -> ordinal unreliable
    edited = _replace_node_span(fragment, dedented, node, repr(right_value))
    if edited is None or edited == fragment:
        return None
    return edited, confidence


# =========================================================================
# NAME (LOAD_FAST/GLOBAL/NAME/DEREF id) and ATTRIBUTE (LOAD_ATTR/METHOD) swaps
# =========================================================================

def swap_name(fragment: str, wrong: str, right: str, ordinal: int = 0) -> tuple[str, str] | None:
    """Swap a Load-context Name ``wrong`` -> ``right`` ONLY when it occurs exactly
    once as a Load anywhere in the fragment (ast.walk includes nested scopes, so a
    name also used in a nested lambda/def makes it non-unique and we defer). The
    caller additionally requires the code object to load it exactly once, so the
    single AST node maps unambiguously to the single bytecode load. ``ordinal`` is
    ignored (kept for signature stability) — bytecode-ordinal -> AST-node mapping is
    unreliable for names (nested scopes, implicit augassign loads, eval-order)."""
    dedented = textwrap.dedent(fragment)
    try:
        tree = ast.parse(dedented)
    except SyntaxError:
        return None
    nodes = [n for n in ast.walk(tree)
             if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load) and n.id == wrong]
    if len(nodes) != 1:
        return None
    edited = _replace_node_span(fragment, dedented, nodes[0], right)
    if edited is None or edited == fragment:
        return None
    return edited, "unique"


_NESTED_SCOPE_NODES = (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda, ast.ClassDef,
                       ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)


def _rename_targets(tree: ast.AST, name: str):
    """Collect ``ast.Name``/``ast.arg`` occurrences of ``name`` in the code object's OWN
    scope (a single top-level function's params + body), separated from occurrences that
    fall inside a nested scope (which is a different code object / variable binding).
    Returns (in_scope, nested) or (None, None) if the fragment is not a single function."""
    body = tree.body if isinstance(tree, ast.Module) else []
    if len(body) != 1 or not isinstance(body[0], (ast.FunctionDef, ast.AsyncFunctionDef)):
        return None, None
    root = body[0]
    in_scope: list = []
    nested: list = []

    def collect(node: ast.AST, in_nested: bool) -> None:
        # classify the node ITSELF first, then descend — so a nested scope that is a
        # DIRECT child of the body (e.g. a top-level `class C:` / `def g():`) is
        # detected before its subtree is walked and doesn't leak into in_scope.
        if (isinstance(node, ast.Name) and node.id == name) or \
           (isinstance(node, ast.arg) and node.arg == name):
            (nested if in_nested else in_scope).append(node)
        child_nested = in_nested or isinstance(node, _NESTED_SCOPE_NODES)
        for child in ast.iter_child_nodes(node):
            collect(child, child_nested)

    # parameter NAMES only — NOT their annotations or default-value expressions, which
    # are evaluated in the ENCLOSING scope at def-execution time (as are decorator_list
    # and the return annotation), so renaming them would corrupt an outer-scope binding.
    a = root.args
    for param in [*a.posonlyargs, *a.args, *a.kwonlyargs, a.vararg, a.kwarg]:
        if param is not None and param.arg == name:
            in_scope.append(param)
    # body statements — collect() moves matches to `nested` on entering any nested scope.
    for stmt in root.body:
        collect(stmt, False)
    return in_scope, nested


def _apply_rename(fragment: str, dedented: str, nodes: list, wrong: str, right: str) -> str | None:
    """Replace every single-line identifier span in ``nodes`` (== ``wrong``) with ``right``,
    mapping dedented AST columns back onto the original indented fragment."""
    orig_lines = fragment.splitlines(keepends=True)
    ded_lines = dedented.splitlines(keepends=True)
    by_line: dict[int, list[tuple[int, int]]] = {}
    for n in nodes:
        if None in (n.lineno, n.end_lineno, n.col_offset, n.end_col_offset) or n.lineno != n.end_lineno:
            return None
        li = n.lineno - 1
        if li >= len(orig_lines) or li >= len(ded_lines) or not orig_lines[li].isascii():
            return None
        delta = len(orig_lines[li]) - len(ded_lines[li])
        start, end = n.col_offset + delta, n.end_col_offset + delta
        if orig_lines[li][start:end] != wrong:
            return None
        by_line.setdefault(li, []).append((start, end))
    for li, spans in by_line.items():
        line = orig_lines[li]
        for start, end in sorted(spans, reverse=True):  # right-to-left keeps offsets valid
            line = line[:start] + right + line[end:]
        orig_lines[li] = line
    return "".join(orig_lines)


def swap_local_rename(fragment: str, wrong: str, right: str) -> tuple[str, str] | None:
    """Consistently rename a FAST local ``wrong`` -> ``right`` across ALL its occurrences
    (params + loads + stores) within a single function's own scope. Safe because FAST
    locals are private to one code object. Declines when: not a single-function fragment;
    ``wrong`` also appears in a nested scope (ambiguous binding); ``right`` is already used
    anywhere in the fragment (collision would merge two variables). A wrong-single-reference
    (not a true rename) renames too much and is caught by the loop's oracle accept gate."""
    if not wrong.isidentifier() or not right.isidentifier() or iskeyword(right) or wrong == right:
        return None
    dedented = textwrap.dedent(fragment)
    try:
        tree = ast.parse(dedented)
    except SyntaxError:
        return None
    in_scope, nested = _rename_targets(tree, wrong)
    if not in_scope or nested:
        return None
    if any((isinstance(n, ast.Name) and n.id == right) or (isinstance(n, ast.arg) and n.arg == right)
           for n in ast.walk(tree)):
        return None  # collision: `right` already denotes something in this fragment
    edited = _apply_rename(fragment, dedented, in_scope, wrong, right)
    if edited is None or edited == fragment:
        return None
    try:
        ast.parse(textwrap.dedent(edited))
    except SyntaxError:
        return None
    return edited, "rename"


def _replace_attr_tail(fragment: str, dedented: str, node: ast.Attribute, wrong: str, right: str) -> str | None:
    """Replace the trailing ``.wrong`` attribute of an Attribute node with ``right``."""
    if node.end_lineno is None or node.end_col_offset is None:
        return None
    orig_lines = fragment.splitlines(keepends=True)
    ded_lines = dedented.splitlines(keepends=True)
    li = node.end_lineno - 1
    if li >= len(orig_lines) or li >= len(ded_lines):
        return None
    line = orig_lines[li]
    if not line.isascii():
        return None
    delta = len(orig_lines[li]) - len(ded_lines[li])
    end = node.end_col_offset + delta
    start = end - len(wrong)
    if start < 0 or line[start:end] != wrong:
        return None
    orig_lines[li] = line[:start] + right + line[end:]
    return "".join(orig_lines)


def swap_attr(fragment: str, wrong: str, right: str, ordinal: int = 0) -> tuple[str, str] | None:
    """Swap a Load-context attribute access ``.wrong`` -> ``.right`` ONLY when it
    occurs exactly once in the fragment. The Load-context filter excludes STORE_ATTR/
    DEL targets (which have no LOAD_ATTR/LOAD_METHOD bytecode and would shift a
    bytecode-ordinal mapping); uniqueness over the whole fragment (ast.walk includes
    nested scopes) removes the reorder/nested-scope ambiguity. ``ordinal`` is ignored."""
    dedented = textwrap.dedent(fragment)
    try:
        tree = ast.parse(dedented)
    except SyntaxError:
        return None
    nodes = [n for n in ast.walk(tree)
             if isinstance(n, ast.Attribute) and isinstance(n.ctx, ast.Load) and n.attr == wrong]
    if len(nodes) != 1:
        return None
    edited = _replace_attr_tail(fragment, dedented, nodes[0], wrong, right)
    if edited is None or edited == fragment:
        return None
    return edited, "unique"


# =========================================================================
# General extraction + dispatch
# =========================================================================

def _clean_argrepr(argrepr: str) -> str:
    name = (argrepr or "").strip()
    if "+" in name:  # 3.11 LOAD_GLOBAL "NULL + x"
        name = name.split("+")[-1].strip()
    return name


def extract_operand_divergence(gt_code_object, derived_code_object, failed_offset: int) -> dict | None:
    """Same-opcode operand mismatch across the BINARY_OP / COMPARE_OP / literal
    families -> ``{opname, kind, wrong, right, ordinal}`` or None."""
    der = _instructions(derived_code_object)
    gt = _instructions(gt_code_object)
    idx = next((i for i, ins in enumerate(der) if getattr(ins, "offset", None) == failed_offset), None)
    if idx is None or idx >= len(gt):
        return None
    cand, gt_inst = der[idx], gt[idx]
    op = getattr(cand, "opname", None)
    gop = getattr(gt_inst, "opname", None)

    # Pre-3.11: the binary operator is encoded in the OPCODE NAME (BINARY_ADD vs
    # BINARY_SUBTRACT), not argrepr, so a 3.10 operator swap surfaces as DIFFERENT opnames.
    # Both sides must be mapped binary opcodes with distinct symbols; the resulting div is
    # identical in shape to the 3.11+ BINARY_OP case, so _dispatch_operand_swap handles it.
    csym = _OPNAME_TO_BINOP_SYMBOL.get(op)
    gsym = _OPNAME_TO_BINOP_SYMBOL.get(gop)
    if csym and gsym and csym != gsym:
        ordinal = sum(1 for ins in der
                      if _OPNAME_TO_BINOP_SYMBOL.get(getattr(ins, "opname", "")) == csym
                      and getattr(ins, "offset", -1) < failed_offset)
        return {"opname": op, "kind": "binary_op", "wrong": csym, "right": gsym, "ordinal": ordinal}

    if op != gop:
        return None

    if op == "BINARY_OP":
        # argrepr IS the operator symbol here — do NOT run _clean_argrepr (it would
        # mangle '+' etc., which contain the '+' used for the NULL-prefix split).
        w = (getattr(cand, "argrepr", "") or "").strip()
        r = (getattr(gt_inst, "argrepr", "") or "").strip()
        if not w or not r or w == r or w not in _SYMBOL_TO_AST_BINOP or r not in _SYMBOL_TO_AST_BINOP:
            return None
        ordinal = sum(1 for ins in der if ins.opname == "BINARY_OP"
                      and (getattr(ins, "argrepr", "") or "").strip() == w
                      and getattr(ins, "offset", -1) < failed_offset)
        return {"opname": op, "kind": "binary_op", "wrong": w, "right": r, "ordinal": ordinal}

    if op == "COMPARE_OP":
        w = (getattr(cand, "argrepr", "") or "").strip()
        r = (getattr(gt_inst, "argrepr", "") or "").strip()
        if w not in _SYMBOL_TO_AST_CMP or r not in _SYMBOL_TO_AST_CMP or w == r:
            return None
        ordinal = sum(1 for ins in der if ins.opname == "COMPARE_OP"
                      and (getattr(ins, "argrepr", "") or "").strip() == w
                      and getattr(ins, "offset", -1) < failed_offset)
        return {"opname": op, "kind": "compare_op", "wrong": w, "right": r, "ordinal": ordinal}

    if op in _CONST_OPCODES:
        w = getattr(cand, "argval", None)
        r = getattr(gt_inst, "argval", None)
        if not _is_swappable_literal(w) or not _is_swappable_literal(r):
            return None
        if _const_match(w, r):  # same literal -> not the divergence we can fix
            return None
        ordinal = sum(1 for ins in der if ins.opname in _CONST_OPCODES
                      and _is_swappable_literal(getattr(ins, "argval", None))
                      and _const_match(getattr(ins, "argval", None), w)
                      and getattr(ins, "offset", -1) < failed_offset)
        return {"opname": op, "kind": "literal", "wrong": w, "right": r, "ordinal": ordinal}

    if op in ("IS_OP", "CONTAINS_OP"):
        w = (getattr(cand, "argrepr", "") or "").strip()
        r = (getattr(gt_inst, "argrepr", "") or "").strip()
        valid = _SYMBOL_TO_AST_ISCONTAINS
        if w not in valid or r not in valid or w == r:
            return None
        ordinal = sum(1 for ins in der if ins.opname == op
                      and (getattr(ins, "argrepr", "") or "").strip() == w
                      and getattr(ins, "offset", -1) < failed_offset)
        return {"opname": op, "kind": "is_contains", "wrong": w, "right": r, "ordinal": ordinal}

    if op == "FORMAT_VALUE":
        w = _FORMAT_CONV_FROM_ARGREPR.get((getattr(cand, "argrepr", "") or "").strip())
        r = _FORMAT_CONV_FROM_ARGREPR.get((getattr(gt_inst, "argrepr", "") or "").strip())
        if w is None or r is None or w == r:
            return None
        total = sum(1 for ins in der if ins.opname == "FORMAT_VALUE"
                    and _FORMAT_CONV_FROM_ARGREPR.get((getattr(ins, "argrepr", "") or "").strip()) == w)
        return {"opname": op, "kind": "format_value", "wrong": w, "right": r, "total": total}

    if op in _NAME_OPCODES:
        w = _clean_argrepr(getattr(cand, "argrepr", "")) or str(getattr(cand, "argval", "") or "")
        r = _clean_argrepr(getattr(gt_inst, "argrepr", "")) or str(getattr(gt_inst, "argval", "") or "")
        if not w or not r or w == r or not w.isidentifier() or not r.isidentifier():
            return None
        # total occurrences in the WHOLE code object (not just before the offset):
        # names/attrs fire only when unique, so the caller needs the global count.
        total = sum(1 for ins in der if ins.opname == op
                    and (_clean_argrepr(getattr(ins, "argrepr", "")) or str(getattr(ins, "argval", "") or "")) == w)
        return {"opname": op, "kind": "name", "wrong": w, "right": r, "total": total}

    if op in _ATTR_OPCODES:
        w = (getattr(cand, "argrepr", "") or "").strip()
        r = (getattr(gt_inst, "argrepr", "") or "").strip()
        if not w or not r or w == r or not w.isidentifier() or not r.isidentifier():
            return None
        total = sum(1 for ins in der if ins.opname == op
                    and (getattr(ins, "argrepr", "") or "").strip() == w)
        return {"opname": op, "kind": "attr", "wrong": w, "right": r, "total": total}

    return None


def _round_trips(value) -> bool:
    """True iff repr(value) parses back to an equal literal (safe to inline)."""
    if not isinstance(value, (str, bytes, int, float, bool, type(None), tuple, frozenset)):
        return False
    try:
        return ast.literal_eval(repr(value)) == value
    except Exception:  # noqa: BLE001
        return False


def _is_swappable_literal(value) -> bool:
    """A scalar literal, or a constant TUPLE of scalar literals (folds to one LOAD_CONST).
    Tuples must repr()->literal round-trip so the replacement compiles back to a literal."""
    if isinstance(value, _SIMPLE_LITERAL_TYPES):
        return True
    return (isinstance(value, tuple)
            and all(isinstance(e, _SIMPLE_LITERAL_TYPES) for e in value)
            and _round_trips(value))


def const_inline_candidate(gt_code_object, derived_code_object, fragment: str,
                           repair_context: dict | None) -> dict | None:
    """Inline a ground-truth constant the decompiler couldn't render.

    Pattern: at the divergence GT has ``LOAD_CONST`` while the candidate has a
    ``LOAD_NAME``/``LOAD_GLOBAL`` of an undefined name — classically the
    self-reference placeholder ``NAME = NAME`` PyLingual emits for a literal it
    failed to decompile. We read GT's constant and materialize it as the RHS
    literal. Gated by a repr()->literal_eval round-trip so we never inline an
    unrepresentable value; the loop's recompile + oracle gate is the backstop.
    """
    if not repair_context:
        return None
    failed = repair_context.get("pylingual_failed_result") or {}
    if failed.get("message") != "Different bytecode":
        return None
    failed_offset = repair_context.get("failed_offset")
    if failed_offset is None:
        return None
    der = _instructions(derived_code_object)
    gt = _instructions(gt_code_object)
    idx = next((i for i, ins in enumerate(der) if getattr(ins, "offset", None) == failed_offset), None)
    if idx is None or idx >= len(gt):
        return None
    cand, gt_inst = der[idx], gt[idx]
    if cand.opname not in ("LOAD_NAME", "LOAD_GLOBAL", "LOAD_FAST", "LOAD_DEREF"):
        return None
    if gt_inst.opname not in _CONST_OPCODES:
        return None
    name = _clean_argrepr(getattr(cand, "argrepr", ""))
    value = getattr(gt_inst, "argval", None)
    if not name or not _round_trips(value):
        return None

    dedented = textwrap.dedent(fragment)
    try:
        tree = ast.parse(dedented)
    except SyntaxError:
        return None
    # Require the UNAMBIGUOUS self-reference placeholder `NAME = NAME` (a PyLingual
    # artifact for a literal it couldn't render). A plain `X = NAME` could be a
    # legitimate assignment of a real global and must never be rewritten — the
    # divergence offset isn't tied to a specific assignment site here.
    self_refs = [n for n in ast.walk(tree)
                 if isinstance(n, ast.Assign) and isinstance(n.value, ast.Name) and n.value.id == name
                 and any(isinstance(t, ast.Name) and t.id == name for t in n.targets)]
    if len(self_refs) != 1:
        return None
    node = self_refs[0].value
    edited = _replace_node_span(fragment, dedented, node, repr(value))
    if edited is None or edited == fragment:
        return None
    preview = repr(value)
    if len(preview) > 40:
        preview = preview[:37] + "..."
    return {
        "text": edited,
        "operator": f"inline {name} = {preview}",
        "opname": "LOAD_CONST",
        "kind": "const_inline",
        "confidence": "unique",
    }


def const_to_name_candidate(gt_code_object, derived_code_object, fragment: str,
                            repair_context: dict | None) -> dict | None:
    """Rewrite a placeholder constant the decompiler emitted where GT loads a name.

    Mirror image of :func:`const_inline_candidate`: at the divergence the candidate
    has ``LOAD_CONST`` (typically ``None``, PyLingual's stand-in for a value it
    couldn't resolve) while GT has a plain value load (``LOAD_FAST``/``LOAD_GLOBAL``/
    ``LOAD_NAME``/``LOAD_DEREF``) of a real variable. We read that variable's name
    verbatim from GT's bytecode and replace the placeholder ``ast.Constant`` in the
    source with an ``ast.Name`` load of it.

    Correctness: the name is read only from GT (never invented); it must be a valid,
    non-keyword identifier (rejects the tuple / non-identifier operand case). The
    placeholder constant is localized by bytecode ordinal, and we only fire when the
    source's count of that constant equals DER's count of that ``LOAD_CONST`` — so the
    ordinal maps to exactly one source Constant (else defer). The tail ``return None``
    -> ``return name`` recovery is left to :func:`tail_return_candidate`, so we decline
    when the placeholder constant is the immediate feeder of a ``RETURN_VALUE`` (or is a
    ``RETURN_CONST``). Any ambiguity -> None; the loop's recompile + oracle gate is the
    backstop.
    """
    if not repair_context:
        return None
    failed = repair_context.get("pylingual_failed_result") or {}
    if failed.get("message") != "Different bytecode":
        return None
    failed_offset = repair_context.get("failed_offset")
    if failed_offset is None:
        return None
    failed_offset = int(failed_offset)
    der = _instructions(derived_code_object)
    gt = _instructions(gt_code_object)
    idx = next((i for i, ins in enumerate(der) if getattr(ins, "offset", None) == failed_offset), None)
    if idx is None or idx >= len(gt):
        return None
    cand, gt_inst = der[idx], gt[idx]
    # mirror of const_inline: DER is a constant, GT is a value-name load.
    if cand.opname not in _CONST_OPCODES:
        return None
    if gt_inst.opname not in _NAME_OPCODES:
        return None
    # Leave tail `return None` -> `return name` to tail_return_candidate: a RETURN_CONST,
    # or a LOAD_CONST that immediately feeds RETURN_VALUE, is the return operand.
    if cand.opname == "RETURN_CONST":
        return None
    nxt = der[idx + 1] if idx + 1 < len(der) else None
    if nxt is not None and getattr(nxt, "opname", None) == "RETURN_VALUE":
        return None

    name = _clean_argrepr(getattr(gt_inst, "argrepr", "")) or str(getattr(gt_inst, "argval", "") or "")
    if not name or not name.isidentifier() or iskeyword(name):
        return None
    value = getattr(cand, "argval", None)
    if not isinstance(value, _SIMPLE_LITERAL_TYPES):
        return None  # only a simple scalar placeholder maps to one source Constant

    def _matches(ins) -> bool:
        v = getattr(ins, "argval", _MISSING)
        return (getattr(ins, "opname", None) in _CONST_OPCODES
                and isinstance(v, _SIMPLE_LITERAL_TYPES) and _const_match(v, value))

    ordinal = sum(1 for ins in der if _matches(ins) and getattr(ins, "offset", -1) < failed_offset)
    der_total = sum(1 for ins in der if _matches(ins))

    dedented = textwrap.dedent(fragment)
    try:
        tree = ast.parse(dedented)
    except SyntaxError:
        return None
    lits: list = []
    _collect_literal_nodes(tree, lits)
    nodes = [n for n, v in lits if _const_match(v, value)]
    # ordinal is only resolvable when the source Constant count matches the bytecode's.
    if not nodes or len(nodes) != der_total:
        return None
    node, confidence = _pick(nodes, ordinal)
    if node is None:
        return None
    if confidence == "ordinal" and _has_comprehension(tree):
        return None  # comprehension bytecode order != AST order -> ordinal unreliable
    edited = _replace_node_span(fragment, dedented, node, name)
    if edited is None or edited == fragment:
        return None
    return {
        "text": edited,
        "operator": f"const {value!r} -> name {name}",
        "opname": "LOAD_CONST",
        "kind": "const_to_name",
        "confidence": "unique",
    }


# LOAD opcodes whose operand is a value name (Load ctx) that a tail `return name`
# would compile to. Reuses the value-load family already vetted for name swaps.
_TAIL_NAME_LOAD_OPCODES = _NAME_OPCODES  # {LOAD_FAST, LOAD_GLOBAL, LOAD_NAME, LOAD_DEREF}


def _returns_none_at_tail(instrs: list) -> bool:
    """True iff the code object's final fall-through is an implicit/explicit ``return None``:
    ``RETURN_CONST None`` (3.12+) or ``LOAD_CONST None; RETURN_VALUE`` (<=3.11)."""
    if not instrs:
        return False
    last = instrs[-1]
    if getattr(last, "opname", None) == "RETURN_CONST" and getattr(last, "argval", _MISSING) is None:
        return True
    if getattr(last, "opname", None) == "RETURN_VALUE" and len(instrs) >= 2:
        prev = instrs[-2]
        if getattr(prev, "opname", None) == "LOAD_CONST" and getattr(prev, "argval", _MISSING) is None:
            return True
    return False


def _gt_tail_return_name(instrs: list) -> str | None:
    """If GT's tail is ``LOAD_<name>; RETURN_VALUE`` (an explicit ``return name`` at the
    body indent), return that identifier; otherwise None. Read straight from the stream
    END so a shiftable jump-offset earlier in the stream can't mislocate it."""
    if len(instrs) < 2:
        return None
    last, load = instrs[-1], instrs[-2]
    if getattr(last, "opname", None) != "RETURN_VALUE":
        return None
    if getattr(load, "opname", None) not in _TAIL_NAME_LOAD_OPCODES:
        return None
    name = _clean_argrepr(getattr(load, "argrepr", "")) or str(getattr(load, "argval", "") or "")
    if not name or not name.isidentifier() or iskeyword(name):
        return None
    return name


def tail_return_candidate(gt_code_object, derived_code_object, fragment: str,
                          repair_context: dict | None) -> dict | None:
    """Recover a lost tail ``return <name>`` the decompiler mis-indented into dead code.

    Pattern: GT's function falls through to ``LOAD_<name>; RETURN_VALUE`` (an explicit
    ``return name`` at the function-body indent), but the derived source placed that
    ``return name`` at a deeper indent — classically right after a ``break``/``return``,
    so it is unreachable — leaving the function to fall through to an implicit
    ``return None`` (``RETURN_CONST None`` on 3.12+, or ``LOAD_CONST None; RETURN_VALUE``
    on <=3.11). We read the wanted name from GT's tail load, then dedent the first
    *viable* over-indented ``return name`` in the function's OWN scope back to the body
    indent (pure text splice, preserving the rest of the body verbatim).

    Deterministic and version-robust: the tail signature is read from the bytecode ENDS
    (not a shiftable ``failed_offset``), we only touch ``return name`` statements in this
    code object's own scope, and every candidate must re-parse AND promote the return to
    a real body-level statement before it reaches the loop's compile + oracle gate. Any
    ambiguity (no such return, non-whitespace cut, broken parse) -> None (defer).
    """
    if not repair_context:
        return None
    failed = repair_context.get("pylingual_failed_result") or {}
    if failed.get("message") != "Different bytecode":
        return None
    der = _instructions(derived_code_object)
    gt = _instructions(gt_code_object)
    if not _returns_none_at_tail(der):
        return None
    name = _gt_tail_return_name(gt)
    if name is None:
        return None

    dedented = textwrap.dedent(fragment)
    try:
        tree = ast.parse(dedented)
    except SyntaxError:
        return None
    body = tree.body if isinstance(tree, ast.Module) else []
    if len(body) != 1 or not isinstance(body[0], (ast.FunctionDef, ast.AsyncFunctionDef)):
        return None
    root = body[0]
    if not root.body:
        return None
    body_indent = root.body[0].col_offset
    if body_indent is None:
        return None

    # Over-indented `return name` statements in THIS function's own scope only. A nested
    # def/lambda/comprehension is a separate code object, so its returns are off-limits.
    over_indented: list[ast.Return] = []

    def collect(node: ast.AST, in_nested: bool) -> None:
        if (not in_nested and isinstance(node, ast.Return)
                and isinstance(node.value, ast.Name) and node.value.id == name):
            col = getattr(node, "col_offset", None)
            if (col is not None and col > body_indent
                    and node.lineno == node.end_lineno):
                over_indented.append(node)
        child_nested = in_nested or isinstance(node, _NESTED_SCOPE_NODES)
        for child in ast.iter_child_nodes(node):
            collect(child, child_nested)

    for stmt in root.body:
        collect(stmt, False)
    if not over_indented:
        return None
    over_indented.sort(key=lambda n: (n.lineno, n.col_offset))

    orig_lines = fragment.splitlines(keepends=True)
    ded_lines = dedented.splitlines(keepends=True)
    for node in over_indented:
        li = node.lineno - 1
        if li >= len(orig_lines) or li >= len(ded_lines):
            continue
        strip = node.col_offset - body_indent  # dedent amount (constant common-prefix cancels)
        if strip <= 0:
            continue
        line = orig_lines[li]
        if len(line) < strip or not line[:strip].isspace():
            continue  # would cut into non-whitespace -> refuse
        edited_lines = list(orig_lines)
        edited_lines[li] = line[strip:]
        edited = "".join(edited_lines)
        if edited == fragment:
            continue
        # Must re-parse AND actually land the return as a body-level (reachable tail)
        # statement of the same single function — otherwise this was the wrong return.
        try:
            new_tree = ast.parse(textwrap.dedent(edited))
        except SyntaxError:
            continue
        nb = new_tree.body
        if len(nb) != 1 or not isinstance(nb[0], (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        # The recovered `return name` must be the LAST statement of the function body —
        # otherwise trailing statements after it become dead code (diverges from GT while
        # still compiling, and could be wrongly accepted). Review fix.
        last = nb[0].body[-1] if nb[0].body else None
        if not (isinstance(last, ast.Return) and isinstance(last.value, ast.Name)
                and last.value.id == name):
            continue
        return {
            "text": edited,
            "operator": f"dedent tail `return {name}`",
            "opname": "RETURN_VALUE",
            "kind": "tail_return",
            "confidence": "unique",
        }
    return None


def _dispatch_operand_swap(div: dict | None, fragment: str) -> dict | None:
    """Apply a same-opcode operand divergence (from ``extract_operand_divergence``) to the
    source fragment. Shared by the offset-based and offset-inferred leaf paths."""
    if div is None:
        return None
    kind = div["kind"]
    ordinal = div.get("ordinal", 0)
    if kind == "binary_op":
        swapped = swap_binary_operator(fragment, div["wrong"], div["right"], ordinal)
        label = f"{div['wrong']} -> {div['right']}"
    elif kind == "compare_op":
        swapped = swap_compare_operator(fragment, div["wrong"], div["right"], ordinal)
        label = f"{div['wrong']} -> {div['right']}"
    elif kind == "literal":
        swapped = swap_literal(fragment, div["wrong"], div["right"], ordinal)
        label = f"{div['wrong']!r} -> {div['right']!r}"
    elif kind == "is_contains":
        swapped = swap_is_contains(fragment, div["wrong"], div["right"], ordinal)
        label = f"{div['wrong']} -> {div['right']}"
    elif kind == "format_value":
        swapped = swap_format_value(fragment, div["wrong"], div["right"]) if div.get("total") == 1 else None
        label = f"f-string conv {div['wrong']} -> {div['right']}"
    elif kind == "name":
        # FAST locals are private to this code object: prefer a consistent
        # whole-scope rename (fixes multi-use renames the single-swap can't, and
        # correctly renames the def+uses together). Fall back to the unique
        # single-swap for other name opcodes (global/closure/module) or when the
        # rename declines. total==1 guards augassign/implicit loads for the swap.
        swapped = None
        if div["opname"] in _RENAMABLE_LOCAL_OPCODES:
            swapped = swap_local_rename(fragment, div["wrong"], div["right"])
        if swapped is None and div.get("total") == 1:
            swapped = swap_name(fragment, div["wrong"], div["right"])
        label = f"name {div['wrong']} -> {div['right']}"
    elif kind == "attr":
        swapped = swap_attr(fragment, div["wrong"], div["right"]) if div.get("total") == 1 else None
        label = f".{div['wrong']} -> .{div['right']}"
    else:
        swapped = None
        label = ""
    if swapped is None:
        return None
    new_fragment, confidence = swapped
    return {
        "text": new_fragment,
        "operator": label,
        "opname": div["opname"],
        "kind": kind,
        "confidence": confidence,
    }


def _operand_repr(ins) -> str:
    """Stable operand surface for divergence detection: argrepr, else str(argval)."""
    ar = (getattr(ins, "argrepr", "") or "").strip()
    if ar:
        return ar
    av = getattr(ins, "argval", _MISSING)
    return "" if av is _MISSING else str(av)


def _infer_single_operand_offset(gt_code_object, derived_code_object) -> int | None:
    """Locate the offset of a SINGLE same-opcode operand divergence WITHOUT a PyLingual
    ``failed_offset`` (the "Different bytecode" + no-offset regime, ~46 of the residual).
    Requires the GT and derived instruction streams to have identical length and pairwise
    identical opcodes (same structure), with EXACTLY ONE position whose operand differs.
    Any structural difference (opcode mismatch / length mismatch) or more than one operand
    divergence -> None (defer: not a clean single-operand swap). Returns the DERIVED offset
    at the divergence, which ``extract_operand_divergence`` then validates per-opcode."""
    der = _instructions(derived_code_object)
    gt = _instructions(gt_code_object)
    if not der or len(der) != len(gt):
        return None
    diff_idx = None
    for i, (dins, gins) in enumerate(zip(der, gt)):
        if getattr(dins, "opname", None) != getattr(gins, "opname", None):
            return None  # structural divergence, not a pure operand swap
        if _operand_repr(dins) != _operand_repr(gins):
            if diff_idx is not None:
                return None  # more than one operand differs -> compound, defer
            diff_idx = i
    if diff_idx is None:
        return None
    return getattr(der[diff_idx], "offset", None)


def leaf_value_no_offset_candidate(gt_code_object, derived_code_object, fragment: str,
                                   repair_context: dict | None) -> dict | None:
    """Offset-independent leaf swap for the "Different bytecode" cases PyLingual does NOT
    localize to a single ``failed_offset`` (they otherwise miss the leaf operator entirely
    and fall to the ``is_diffbc`` branch, which has no plain operand swap). Infers the lone
    diverging instruction from the GT/derived stream diff, then reuses the vetted per-opcode
    swap dispatch. Never raises; defers on any ambiguity; the loop's recompile + oracle gate
    remains the correctness backstop."""
    if not repair_context:
        return None
    failed = repair_context.get("pylingual_failed_result") or {}
    if failed.get("message") != "Different bytecode":
        return None
    # Only the no-offset regime — the offset path is handled by leaf_value_candidate.
    if repair_context.get("failed_offset") is not None:
        return None
    offset = _infer_single_operand_offset(gt_code_object, derived_code_object)
    if offset is None:
        return None
    div = extract_operand_divergence(gt_code_object, derived_code_object, int(offset))
    return _dispatch_operand_swap(div, fragment)


def leaf_value_candidate(gt_code_object, derived_code_object, fragment: str,
                         repair_context: dict | None) -> dict | None:
    """Top-level leaf operator. Returns ``{"text", "operator", "opname", "kind",
    "confidence"}`` for an accepted source edit, or None to defer."""
    if not repair_context:
        return None
    failed = repair_context.get("pylingual_failed_result") or {}
    if failed.get("message") != "Different bytecode":
        return None
    failed_offset = repair_context.get("failed_offset")
    if failed_offset is None:
        return None
    div = extract_operand_divergence(gt_code_object, derived_code_object, int(failed_offset))
    dispatched = _dispatch_operand_swap(div, fragment)
    if dispatched is not None:
        return dispatched
    # Not a same-opcode operand swap — try different-opcode constant inlining
    # (LOAD_NAME placeholder -> GT LOAD_CONST), then tail-return recovery
    # (GT `LOAD_<name>; RETURN_VALUE` vs a derived implicit `return None`).
    inlined = const_inline_candidate(gt_code_object, derived_code_object, fragment, repair_context)
    if inlined is not None:
        return inlined
    named = const_to_name_candidate(gt_code_object, derived_code_object, fragment, repair_context)
    if named is not None:
        return named
    return tail_return_candidate(gt_code_object, derived_code_object, fragment, repair_context)
