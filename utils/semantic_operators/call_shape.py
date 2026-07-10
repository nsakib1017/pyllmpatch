"""Call-shape operator: deterministically fix a call the decompiler rendered with the
wrong POSITIONAL-vs-KEYWORD form, reading the correct shape straight from ground-truth
bytecode.

3.11+ bytecode makes keyword-argument names EXPLICIT, which is what makes this a
*deterministic* recovery rather than a guess:

  * 3.11 / 3.12 — a ``KW_NAMES`` instruction loads a const tuple of the trailing kwarg
    names immediately before the ``CALL`` (e.g. ``KW_NAMES ('key',)`` then ``CALL 2``).
  * 3.13        — the names ride on ``CALL_KW`` (its immediately-preceding ``LOAD_CONST``
    is the names tuple; the ``CALL_KW`` arg is the total argument count).
  * <= 3.10     — no ``KW_NAMES`` / ``CALL_KW`` in this form, so this operator DEFERS
    (returns None). That is expected and fine.

So for a call in GT we recover, deterministically: ``total`` argument count (the CALL
arg), the exact tuple of trailing keyword names, and therefore the positional split
``n_positional = total - len(names)``. When the decompiler emitted the SAME callee with
the SAME argument VALUES but passed those trailing kwargs POSITIONALLY (``g(1, 2)`` where
GT has ``g(1, key=2)``), only the positional/keyword FORM differs — the value expressions
are byte-for-byte identical between the two streams, so ``difflib.SequenceMatcher`` over
the opname streams surfaces the ``KW_NAMES`` as a lone ``delete`` (present in GT, absent
in derived). That is the signal we key on (mirrors ``literal_form.py`` /
``tools/classify_semantic_divergences.py``).

We then locate the UNIQUE ``ast.Call`` in the derived fragment that currently passes those
arguments positionally (matched by argument count + fully-positional shape) and rewrite
its trailing ``n_kw`` positional arguments into ``name=<expr>`` keyword arguments, KEEPING
the derived VALUE expressions verbatim (only the pos/kw form changes) and preserving order.
The names are read ONLY from GT — never invented.

Unique-or-defer everywhere; the loop's recompile + oracle accept gate is the backstop; the
operator never emits a name it did not read from GT, and it NEVER raises (top-level
try/except -> None).
"""
from __future__ import annotations

import ast
import textwrap
from difflib import SequenceMatcher

from utils.semantic_operators.leaf import _instructions

# CALL-family terminators. 3.11/3.12 emit CALL (optionally preceded by PRECALL); 3.13 emits
# CALL_KW for a keyword call. CALL_FUNCTION* are the <=3.10 forms (kept for completeness; the
# KW_NAMES/CALL_KW keyword signal they lack means those simply never produce a divergence here).
_CALL_OPCODES = {"CALL", "CALL_KW", "CALL_FUNCTION", "CALL_FUNCTION_KW"}


def _real_instructions(bytecode) -> list:
    """The instruction stream with CACHE pseudo-instructions removed, so a 3.11+ opcode that
    trails CACHE entries aligns 1:1 against a cache-less stream in the opname alignment."""
    return [i for i in _instructions(bytecode) if getattr(i, "opname", None) != "CACHE"]


def _argint(ins):
    """The integer operand of an instruction (a CALL's total arg count), or None."""
    a = getattr(ins, "arg", None)
    if isinstance(a, int):
        return a
    v = getattr(ins, "argval", None)
    return v if isinstance(v, int) else None


def _valid_names(names) -> bool:
    """True iff ``names`` is a non-empty tuple of distinct, valid keyword identifiers."""
    if not isinstance(names, tuple) or not names:
        return False
    if not all(isinstance(n, str) and n.isidentifier() for n in names):
        return False
    return len(set(names)) == len(names)


def _call_total_after(insts: list, i: int) -> int | None:
    """From a ``KW_NAMES`` at index ``i``, the total arg count of the CALL it feeds (the next
    CALL-family op, tolerating a 3.11 ``PRECALL`` filler in between). None if none follows."""
    for j in range(i + 1, len(insts)):
        op = getattr(insts[j], "opname", None)
        if op in _CALL_OPCODES:
            return _argint(insts[j])
        if op in ("PRECALL", "CACHE"):
            continue
        break  # any other opcode between KW_NAMES and its CALL -> not the clean signal
    return None


def _callkw_names(insts: list, i: int):
    """3.13: the names tuple of a ``CALL_KW`` at index ``i`` is its immediately-preceding
    ``LOAD_CONST`` tuple. Returns that tuple, or None."""
    if i - 1 < 0:
        return None
    prev = insts[i - 1]
    if getattr(prev, "opname", None) != "LOAD_CONST":
        return None
    v = getattr(prev, "argval", None)
    return v if isinstance(v, tuple) else None


def _find_call_shape_divergences(gt: list, der: list) -> list[dict]:
    """Align GT vs derived opname streams and return every call whose keyword names GT
    makes explicit but derived does NOT — i.e. GT keyworded arguments the decompiler passed
    positionally. Each entry is ``{"names": tuple, "total": int, "n_positional": int}``.

    Detection: within a non-``equal`` alignment chunk (a divergence between the streams),
    a GT ``KW_NAMES`` (3.11/3.12) or ``CALL_KW`` (3.13) with a valid names tuple whose CALL
    total >= len(names) marks such a call. A same-shape keyword call (present on BOTH sides)
    aligns into an ``equal`` chunk and is correctly NOT reported."""
    sm = SequenceMatcher(a=[getattr(i, "opname", None) for i in gt],
                         b=[getattr(i, "opname", None) for i in der], autojunk=False)
    divs: list[dict] = []
    for tag, i1, i2, _j1, _j2 in sm.get_opcodes():
        if tag == "equal":
            continue
        for i in range(i1, i2):
            op = getattr(gt[i], "opname", None)
            if op == "KW_NAMES":
                names = getattr(gt[i], "argval", None)
                total = _call_total_after(gt, i)
            elif op == "CALL_KW":
                names = _callkw_names(gt, i)
                total = _argint(gt[i])
            else:
                continue
            if not _valid_names(names) or total is None:
                continue
            n_positional = total - len(names)
            if n_positional < 0:
                continue
            divs.append({"names": tuple(names), "total": total, "n_positional": n_positional})
    return divs


def _dotted_name(node: ast.AST) -> str | None:
    """The dotted callee name of a call target (``g`` / ``obj.meth`` / ``a.b.c``), or None
    when the callee is not a plain name/attribute chain (subscript, call result, ...)."""
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        base = _dotted_name(node.value)
        return f"{base}.{node.attr}" if base else None
    return None


def _positional_call_candidates(tree: ast.AST, total: int) -> list[ast.Call]:
    """Every ``ast.Call`` in ``tree`` that currently passes exactly ``total`` arguments, ALL
    positionally (no keywords, no ``*args``/``**kwargs``), with a plain dotted callee. These
    are the calls whose trailing positional args could be the mis-rendered keyword args."""
    out: list[ast.Call] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if node.keywords:  # already has keyword args -> not the fully-positional case we own
            continue
        if any(isinstance(a, ast.Starred) for a in node.args):
            continue  # *args: positional count does not map to the recovered split
        if len(node.args) != total:
            continue
        if _dotted_name(node.func) is None:
            continue
        out.append(node)
    return out


def _replace_call_span(fragment: str, dedented: str, node: ast.AST, new_text: str) -> str | None:
    """Replace a (possibly multi-line) node's source span with ``new_text``, mapping the
    node's dedented positions back into the original indented ``fragment``. ASCII-guarded on
    the boundary lines (``col_offset`` is a utf-8 byte offset)."""
    if None in (node.lineno, node.col_offset, node.end_lineno, node.end_col_offset):
        return None
    orig = fragment.splitlines(keepends=True)
    ded = dedented.splitlines(keepends=True)
    li0, li1 = node.lineno - 1, node.end_lineno - 1
    if li0 < 0 or li1 >= len(orig) or li0 >= len(ded) or li1 >= len(ded) or li0 > li1:
        return None
    if not orig[li0].isascii() or not orig[li1].isascii():
        return None
    start = node.col_offset + (len(orig[li0]) - len(ded[li0]))
    end = node.end_col_offset + (len(orig[li1]) - len(ded[li1]))
    prefix = orig[li0][:start]
    suffix = orig[li1][end:]
    return "".join(orig[:li0]) + prefix + new_text + suffix + "".join(orig[li1 + 1:])


def call_shape_candidate(gt_code_object, derived_code_object, fragment: str,
                         repair_context: dict | None) -> dict | None:
    """Convert a positionally-rendered call back to GT's keyword shape.

    Returns ``{"text", "operator", "opname", "kind", "confidence"}`` for an accepted source
    edit, or None to defer. Deferral cases: not the "Different bytecode" regime; no (or >1)
    call-shape divergence in the bytecode; not exactly one matching positional call in the
    fragment (argument-count mismatch, ambiguity, or the value split does not line up). Never
    raises — a raised exception would crash the repair loop, so everything is swallowed and
    the oracle accept gate is the correctness backstop."""
    try:
        if not repair_context:
            return None
        failed = repair_context.get("pylingual_failed_result") or {}
        if failed.get("message") != "Different bytecode":
            return None

        gt = _real_instructions(gt_code_object)
        der = _real_instructions(derived_code_object)
        if not gt or not der:
            return None

        # Require EXACTLY ONE call-shape divergence: 0 = nothing to do / same shape, >1 =
        # ambiguous which call to touch (the oracle gate cannot localize for us) -> defer.
        divs = _find_call_shape_divergences(gt, der)
        if len(divs) != 1:
            return None
        div = divs[0]
        names = div["names"]
        total = div["total"]
        n_positional = div["n_positional"]

        try:
            dedented = textwrap.dedent(fragment)
            tree = ast.parse(dedented)
        except SyntaxError:
            return None

        # Locate the UNIQUE derived call that passes ``total`` args positionally; >1 (or 0)
        # means we cannot attribute the divergence to a single call site -> defer.
        candidates = _positional_call_candidates(tree, total)
        if len(candidates) != 1:
            return None
        node = candidates[0]

        kept_positional = node.args[:n_positional]
        to_convert = node.args[n_positional:]
        if len(to_convert) != len(names):
            return None  # split does not line up -> defer

        callee = _dotted_name(node.func)
        new_keywords = [ast.keyword(arg=nm, value=val) for nm, val in zip(names, to_convert)]
        new_call = ast.Call(func=node.func, args=list(kept_positional), keywords=new_keywords)
        try:
            new_text = ast.unparse(new_call)
        except Exception:  # noqa: BLE001
            return None

        edited = _replace_call_span(fragment, dedented, node, new_text)
        if edited is None or edited == fragment:
            return None
        try:
            ast.parse(textwrap.dedent(edited))
        except SyntaxError:
            return None

        opname = "CALL_KW" if any(getattr(i, "opname", None) == "CALL_KW" for i in gt) else "KW_NAMES"
        return {
            "text": edited,
            "operator": f"positional->keyword {callee}({', '.join(names)})",
            "opname": opname,
            "kind": "call_shape",
            "confidence": "unique",
        }
    except Exception:  # noqa: BLE001 - a bad candidate is rejected by the oracle gate; a
        # raised exception would crash the repair loop, so swallow everything.
        return None
