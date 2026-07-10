"""Literal-form operator: deterministically fix a SINGLE leaf whose OPCODE (not just
operand) differs between ground-truth and derived bytecode — the "quoting" bug the
decompiler produces when it drops or adds the quotes around a string.

``leaf.py`` deliberately DEFERS the cross-opcode case (its module docstring: "A DIFFERENT
opcode at the divergence ... is not an operand swap and is deferred"). This operator owns
exactly that case, but only for the tightest, self-verifying signature: a string constant
whose *spelling equals* the identifier the other side loaded as a bare name.

  * quote   — GT ``LOAD_CONST 'TIMESTAMP'`` where the candidate emitted a bare name load
              (``LOAD_NAME``/``LOAD_FAST``/``LOAD_GLOBAL``/``LOAD_DEREF``) of ``TIMESTAMP``.
              The decompiler dropped the quotes; we quote it back (``TIMESTAMP`` -> ``'TIMESTAMP'``).
  * unquote — the reverse: GT loads the name ``TIMESTAMP`` while the candidate emitted
              ``LOAD_CONST 'TIMESTAMP'`` (over-quoted); we unquote it (``'TIMESTAMP'`` -> ``TIMESTAMP``).

The two sides differ ONLY in opcode: this is not the same-opcode ``LOAD_CONST`` value swap
(``'a'`` -> ``'b'``) that ``leaf.swap_literal`` already owns — that case has matching opnames
and never surfaces here (it lands in a SequenceMatcher ``equal`` chunk, not a ``replace``), so
it is left to leaf. We align the GT vs derived opname streams with ``difflib.SequenceMatcher``
(as ``tools/classify_semantic_divergences.py`` does), require EXACTLY ONE such quoting
divergence, then locate the UNIQUE matching source node (a ``Name`` for quote, a str
``Constant`` for unquote) and splice. Unique-or-defer everywhere; the loop's recompile +
oracle accept gate is the backstop; this never emits a value it did not read from GT, and it
never raises.
"""
from __future__ import annotations

import ast
import textwrap
from difflib import SequenceMatcher
from keyword import iskeyword

from utils.semantic_operators.leaf import _instructions, _replace_node_span, _round_trips

# The two opcode families whose XOR is the quoting signature. A const *load* on one side,
# a bare value-name load on the other. RETURN_CONST / RETURN_VALUE (tail returns) are
# intentionally excluded — those are owned by leaf's tail_return / const_to_name operators.
_CONST_LOAD_OPCODES = {"LOAD_CONST"}
_NAME_LOAD_OPCODES = {"LOAD_NAME", "LOAD_FAST", "LOAD_GLOBAL", "LOAD_DEREF"}


def _real_instructions(bytecode) -> list:
    """The code object's instruction stream with CACHE pseudo-instructions removed, so a
    3.11+ ``LOAD_GLOBAL`` (which trails CACHE entries) aligns 1:1 against a cache-less
    ``LOAD_CONST`` in the opname stream."""
    return [i for i in _instructions(bytecode) if getattr(i, "opname", None) != "CACHE"]


def _name_of(ins) -> str | None:
    """The identifier a value-name load references, read from ``argval`` (a str for
    LOAD_NAME/FAST/GLOBAL/DEREF), falling back to a cleaned ``argrepr`` that strips the
    3.11 ``"NULL + name"`` LOAD_GLOBAL prefix. None when it is not a plain identifier."""
    v = getattr(ins, "argval", None)
    if isinstance(v, str) and v.isidentifier():
        return v
    r = (getattr(ins, "argrepr", "") or "").strip()
    if "+" in r:  # 3.11 LOAD_GLOBAL "NULL + x"
        r = r.split("+")[-1].strip()
    return r if r.isidentifier() else None


def _find_quoting_divergences(gt: list, der: list) -> list[dict]:
    """Align GT vs derived opname streams and return every 1-for-1 opcode swap that is a
    quoting bug: a const-load on one side and a same-text bare-name load on the other, where
    the constant is a str whose value EQUALS the identifier. Only a clean 1-for-1 ``replace``
    chunk qualifies (a wider chunk is a compound divergence we cannot attribute to quoting)."""
    sm = SequenceMatcher(a=[i.opname for i in gt], b=[i.opname for i in der], autojunk=False)
    divs: list[dict] = []
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag != "replace" or i2 - i1 != 1 or j2 - j1 != 1:
            continue
        g, d = gt[i1], der[j1]
        gop, dop = getattr(g, "opname", None), getattr(d, "opname", None)
        if gop in _CONST_LOAD_OPCODES and dop in _NAME_LOAD_OPCODES:
            cval = getattr(g, "argval", None)
            ident = _name_of(d)
            if isinstance(cval, str) and ident is not None and cval == ident:
                divs.append({"direction": "quote", "identifier": ident, "const_value": cval})
        elif gop in _NAME_LOAD_OPCODES and dop in _CONST_LOAD_OPCODES:
            cval = getattr(d, "argval", None)
            ident = _name_of(g)
            if isinstance(cval, str) and ident is not None and cval == ident:
                divs.append({"direction": "unquote", "identifier": ident, "const_value": cval})
    return divs


def literal_form_candidate(gt_code_object, derived_code_object, fragment: str,
                           repair_context: dict | None) -> dict | None:
    """Fix a single dropped/added quote pair by reading the correct leaf form from GT
    bytecode. Returns ``{"text", "operator", "opname", "kind", "confidence"}`` or None to
    defer. Never raises (top-level try/except -> None)."""
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

        # Require EXACTLY ONE quoting divergence — defer on 0 (nothing to do / not our case)
        # or >1 (ambiguous which leaf to touch; the oracle gate can't localize for us).
        divs = _find_quoting_divergences(gt, der)
        if len(divs) != 1:
            return None
        div = divs[0]
        ident = div["identifier"]
        const_value = div["const_value"]

        try:
            dedented = textwrap.dedent(fragment)
            tree = ast.parse(dedented)
        except SyntaxError:
            return None

        if div["direction"] == "quote":
            # bare name -> string literal. repr() must round-trip so we never inline a
            # value that would re-parse to something else.
            if not _round_trips(const_value):
                return None
            nodes = [n for n in ast.walk(tree)
                     if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load) and n.id == ident]
            if len(nodes) != 1:
                return None
            new_text = repr(const_value)
            opname = "LOAD_CONST"
            label = f"quote name {ident} -> {const_value!r}"
        else:  # unquote: string literal -> bare name
            if not ident.isidentifier() or iskeyword(ident):
                return None
            nodes = [n for n in ast.walk(tree)
                     if isinstance(n, ast.Constant) and isinstance(n.value, str) and n.value == const_value]
            if len(nodes) != 1:
                return None
            new_text = ident
            opname = "LOAD_NAME"
            label = f"unquote {const_value!r} -> name {ident}"

        edited = _replace_node_span(fragment, dedented, nodes[0], new_text)
        if edited is None or edited == fragment:
            return None
        try:
            ast.parse(textwrap.dedent(edited))
        except SyntaxError:
            return None
        return {
            "text": edited,
            "operator": label,
            "opname": opname,
            "kind": "literal_form",
            "confidence": "unique",
        }
    except Exception:  # noqa: BLE001 - a bad candidate is rejected by the oracle gate; a
        # raised exception would crash the repair loop, so swallow everything.
        return None
