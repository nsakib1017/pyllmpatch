"""Structural operators (M4): deterministic, no-LLM reconstruction of control
flow from ground-truth bytecode, for the "Different control flow" regime.

Flagship: TRY/EXCEPT reconstruction from the GT exception table. When the GT
protects a call statement with a try/except that the candidate emitted unwrapped
(often alongside a dead empty `try/except` placeholder the decompiler misplaced),
we:
  1. read the protected call's dotted name from the GT protected instruction range,
  2. find that call as an unwrapped statement in the candidate AST,
  3. reconstruct the handler skeleton from the GT handler instructions,
  4. wrap the statement in try/except and drop the dead placeholder / empty else,
  5. regenerate the source with ast.unparse.

Deterministic SKELETON reconstruction only: it fires when the handler is trivially
recoverable (bare `except: pass` or `except SomeError: pass`) and the protected
body is a single call already present in the candidate. Anything richer returns
None and defers to the LLM. Every candidate still passes the loop's compile +
oracle accept gate, so a wrong reconstruction cannot land.
"""
from __future__ import annotations

import ast
import textwrap


def _instructions(bc) -> list:
    try:
        return list(bc)
    except Exception:  # noqa: BLE001
        return []


def _real_try_blocks(bc) -> list[tuple]:
    """Exception-table entries that introduce a handler (lasti False) — i.e. real
    try bodies, not the auto-generated cleanup/reraise entries."""
    table = getattr(bc, "named_exception_table", None) or []
    out = []
    for e in table:
        if getattr(e, "lasti", False):
            continue
        out.append((getattr(e, "start", None), getattr(e, "end", None),
                    getattr(e, "target", None), getattr(e, "depth", None)))
    return out


def _clean_name(argrepr: str) -> str:
    """LOAD_GLOBAL argrepr in 3.11+ can be 'NULL + os'; take the real name."""
    name = (argrepr or "").strip()
    if "+" in name:
        name = name.split("+")[-1].strip()
    return name


def _protected_call_parts(gt_insts: list, start: int, end: int) -> list[str] | None:
    """Dotted call parts for the single call protected by [start, end) in GT, e.g.
    ['os', 'remove']. None if the range is not exactly one simple call."""
    window = [i for i in gt_insts if start is not None and end is not None
              and start <= getattr(i, "offset", -1) < end]
    if not window:
        return None
    calls = [i for i in window if i.opname in ("CALL", "CALL_FUNCTION", "CALL_METHOD")]
    if len(calls) != 1:
        return None
    parts: list[str] = []
    for i in window:
        op = i.opname
        if not parts:
            if op in ("LOAD_GLOBAL", "LOAD_NAME", "LOAD_FAST", "LOAD_DEREF"):
                name = _clean_name(getattr(i, "argrepr", ""))
                if name:
                    parts = [name]
            # else keep scanning for the base load
        elif op in ("LOAD_METHOD", "LOAD_ATTR"):
            # only attribute loads that CONTINUE the callee chain; the first
            # non-attr op (an argument load or the CALL) ends the callee.
            # 3.12 method-call LOAD_ATTR argrepr can be 'NULL|self + remove'; clean it the same
            # way as the base (attribute names never contain '+', so this only strips the prefix).
            attr = _clean_name(getattr(i, "argrepr", ""))
            if not attr:
                break
            parts.append(attr)
        else:
            break
    return parts or None


def _handler_skeleton(gt_insts: list, target: int) -> tuple[bool, str | None]:
    """Inspect the GT handler at `target`. Returns (is_trivial, exc_type_expr).

    Trivial = bare `except: pass` or `except SomeError: pass` (no real work in the
    handler body). exc_type_expr is the (possibly dotted) exception name for the
    typed form, else None.
    """
    # The handler runs from `target` up to and including its exit (the JUMP that
    # leaves the handler, or a RERAISE). A fixed offset window would overrun into
    # the following code (e.g. the function's return), so stop at the exit.
    window: list = []
    started = False
    for i in gt_insts:
        if getattr(i, "offset", -1) == target:
            started = True
        if not started:
            continue
        window.append(i)
        if i.opname in ("JUMP_FORWARD", "JUMP_BACKWARD", "RERAISE"):
            break
    if not window:
        return False, None
    ops = [i.opname for i in window]
    # Only pure stack-cleanup / exception-match opcodes are allowed in a trivial
    # (pass) handler. Anything that does work — RETURN_*, STORE_*, LOAD_CONST,
    # CALL, etc. — means the handler has a real body and must NOT be reduced to pass.
    body_ops = [o for o in ops if o not in (
        "PUSH_EXC_INFO", "POP_TOP", "POP_EXCEPT", "RERAISE", "COPY", "JUMP_FORWARD",
        "JUMP_BACKWARD", "CHECK_EXC_MATCH", "LOAD_GLOBAL", "LOAD_NAME",
        "POP_JUMP_FORWARD_IF_FALSE", "POP_JUMP_IF_FALSE", "NOP",
    )]
    if body_ops:
        return False, None
    exc_type = None
    if "CHECK_EXC_MATCH" in ops:
        for i in window:
            if i.opname in ("LOAD_GLOBAL", "LOAD_NAME"):
                exc_type = _clean_name(getattr(i, "argrepr", "")) or None
                break
        if exc_type is None:
            return False, None
    return True, exc_type


def _call_parts(call: ast.Call) -> list[str]:
    node = call.func
    parts: list[str] = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
    parts.reverse()
    return parts


def _only_pass_or_dead(stmts: list[ast.stmt]) -> bool:
    """A block that does nothing: only Pass, or placeholder Try/If whose bodies
    are themselves dead."""
    if not stmts:
        return True
    for s in stmts:
        if isinstance(s, ast.Pass):
            continue
        if isinstance(s, ast.Try) and _only_pass_or_dead(s.body) and all(
            _only_pass_or_dead(h.body) for h in s.handlers
        ) and _only_pass_or_dead(s.orelse) and _only_pass_or_dead(s.finalbody):
            continue
        return False
    return True


class _WrapCall(ast.NodeTransformer):
    def __init__(self, parts: list[str], exc_type: str | None):
        self.parts = parts
        self.exc_type = exc_type
        self.count = 0

    def visit_Expr(self, node: ast.Expr):
        if isinstance(node.value, ast.Call) and _call_parts(node.value) == self.parts:
            self.count += 1
            handler_type = None
            if self.exc_type:
                handler_type = ast.parse(self.exc_type, mode="eval").body
            handler = ast.ExceptHandler(type=handler_type, name=None, body=[ast.Pass()])
            return ast.Try(body=[node], handlers=[handler], orelse=[], finalbody=[])
        return node


class _DropDead(ast.NodeTransformer):
    def visit_Try(self, node: ast.Try):
        self.generic_visit(node)
        if _only_pass_or_dead(node.body) and all(_only_pass_or_dead(h.body) for h in node.handlers) \
                and not node.finalbody and not node.orelse:
            return None  # remove dead placeholder try
        return node

    def visit_If(self, node: ast.If):
        self.generic_visit(node)
        if node.orelse and _only_pass_or_dead(node.orelse):
            node.orelse = []
        return node


def _match_indent_convention(original: str, unparsed: str) -> str | None:
    """Re-indent ast.unparse output to match the original fragment's convention.

    Extracted fragments are "hanging": the first line sits at the qualname's own
    column stripped to 0 while the body carries its absolute file indentation.
    ast.unparse instead produces first-line col 0 + 4-space body, so on splice the
    body would be under-indented. Restore: first line at the original's first-line
    indent, body shifted so its first level matches the original body indent.
    """
    o_lines = original.splitlines()
    u_lines = unparsed.splitlines()
    if not o_lines or not u_lines:
        return None
    fi = len(o_lines[0]) - len(o_lines[0].lstrip())
    # The body begins after the first block header (decorators + `def`/`class` may
    # precede it at the same indent); computing bi over ALL following lines would
    # wrongly pick up a same-indent `def` line under a decorator and over-dedent.
    header_idx = 0
    for idx, line in enumerate(o_lines):
        if line.rstrip().endswith(":"):
            header_idx = idx
            break
    body = [len(l) - len(l.lstrip()) for l in o_lines[header_idx + 1:] if l.strip()]
    bi = min(body) if body else fi + 4
    shift = bi - 4  # ast.unparse indents the first body level by 4 spaces
    out = [(" " * fi + u_lines[0]) if fi else u_lines[0]]
    for line in u_lines[1:]:
        if not line.strip():
            out.append(line)
        elif shift > 0:
            out.append(" " * shift + line)
        elif shift < 0:
            cur = len(line) - len(line.lstrip())
            out.append(line[min(-shift, cur):])
        else:
            out.append(line)
    result = "\n".join(out)
    if original.endswith("\n") and not result.endswith("\n"):
        result += "\n"
    return result


def try_except_candidate(gt_code_object, derived_code_object, fragment: str,
                         repair_context: dict | None) -> dict | None:
    """Deterministic try/except reconstruction. Returns {"text", "detail"} or None."""
    if not repair_context:
        return None
    failed = repair_context.get("pylingual_failed_result") or {}
    if failed.get("message") != "Different control flow":
        return None

    gt_insts = _instructions(gt_code_object)
    gt_blocks = _real_try_blocks(gt_code_object)
    der_blocks = _real_try_blocks(derived_code_object)
    if len(gt_blocks) != len(der_blocks) + 1:
        return None  # only the "GT introduces exactly one new try" case

    dedented = textwrap.dedent(fragment)
    try:
        ast.parse(dedented)
    except SyntaxError:
        return None

    for (s, e, tgt, depth) in gt_blocks:
        parts = _protected_call_parts(gt_insts, s, e)
        if not parts:
            continue
        trivial, exc_type = _handler_skeleton(gt_insts, tgt)
        if not trivial:
            continue
        work = ast.parse(dedented)  # fresh tree per attempt
        wrapper = _WrapCall(parts, exc_type)
        work = wrapper.visit(work)
        if wrapper.count != 1:
            continue  # need a unique unwrapped call to wrap
        work = _DropDead().visit(work)
        ast.fix_missing_locations(work)
        try:
            new_dedented = ast.unparse(work)
        except Exception:  # noqa: BLE001
            continue
        reindented = _match_indent_convention(fragment, new_dedented)
        if reindented is None or reindented.strip() == fragment.strip():
            continue
        # Safety net: never emit a fragment that doesn't parse (guards against
        # indentation/empty-body edge cases). Decline instead of propagating bad source.
        try:
            ast.parse(textwrap.dedent(reindented))
        except SyntaxError:
            continue
        header = f"except {exc_type}" if exc_type else "except"
        return {"text": reindented, "detail": f"wrapped {'.'.join(parts)} in try/{header}"}
    return None
