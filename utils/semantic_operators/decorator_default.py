"""Decorator / default-argument operator: deterministically restore a function
**decorator** or **positional default value** the decompiler dropped, reading both straight
from the ground-truth ``MAKE_FUNCTION`` bytecode.

A ``def`` and its wrapper produce two *separate* code objects:

  * the function **body** (its own code object) — this is byte-identical whether or not the
    source carried ``@decorator`` / ``param=default``; defaults and decorators are set up in
    the **enclosing** scope, not inside the body;
  * the **enclosing** scope (module / class body / outer function) — this is where
    ``MAKE_FUNCTION`` (defaults tuple + flag) and the decorator ``LOAD ... CALL ... STORE``
    live.

Because the body is identical, the *only* code object whose bytecode diverges when a default
or decorator is dropped is the **parent**. That is exactly the code object the repair loop
targets and passes to this operator, so no parent-climbing is needed: we scan the *given*
code object for ``MAKE_FUNCTION`` definition sites. If we are ever handed a bare function
**body** (which contains no ``MAKE_FUNCTION`` of its own) we simply find no sites and defer.

Two sub-cases, dispatched by what GT has that the derived object lacks:

  * **DEFAULTS** — GT's ``MAKE_FUNCTION`` carries positional defaults (flag ``0x01``: a single
    ``LOAD_CONST <const tuple>`` just before the code load) that the derived ``def`` lacks.
    The default values are read verbatim from GT and attached (``param=<repr(value)>``) to the
    trailing positional parameters they align to via ``co_argcount``.
  * **DECORATOR** — GT wraps the ``def`` in exactly one simple ``@name`` / ``@a.b`` decorator
    (a ``LOAD name`` [``LOAD_ATTR`` ...] before the code load and a ``CALL`` after
    ``MAKE_FUNCTION``, before ``STORE``) that the derived ``def`` dropped. The decorator
    expression is recovered from bytecode and prepended as ``@<expr>``.

Conservative and unique-or-defer, mirroring the other deterministic operators:

  * exactly ONE function name may diverge in the given regime, and that name must resolve to a
    UNIQUE ``FunctionDef``/``AsyncFunctionDef`` in the fragment (and a unique GT/derived site);
  * default values must round-trip through ``repr()`` -> ``ast.literal_eval`` (reuses
    ``leaf._round_trips``); non-constant defaults (``b=[]``, ``b=NAME``), kwonly-default /
    annotation / closure entanglement, stacked or call-form decorators (``@deco(x)``), and any
    parameter-name mismatch all **defer** (return ``None``);
  * the emitted fragment is re-parsed before returning.

The recompile + oracle distance gate is the final backstop; this operator never emits a value
it did not read verbatim from GT bytecode, and it **never raises**.
"""
from __future__ import annotations

import ast
import textwrap

from utils.semantic_operators.leaf import _instructions, _round_trips

# MAKE_FUNCTION oparg flag bits (CPython 3.10-3.12).
_FLAG_DEFAULTS = 0x01      # a tuple of positional defaults on the stack
_FLAG_KWDEFAULTS = 0x02    # a dict of keyword-only defaults
_FLAG_ANNOTATIONS = 0x04   # annotations
_FLAG_CLOSURE = 0x08       # a tuple of cells (closure)

_STORE_OPCODES = {"STORE_NAME", "STORE_FAST", "STORE_GLOBAL", "STORE_DEREF"}
_CALL_OPCODES = {"CALL", "CALL_FUNCTION", "CALL_METHOD", "CALL_FUNCTION_EX"}
_NAME_LOAD_OPCODES = {"LOAD_NAME", "LOAD_GLOBAL", "LOAD_DEREF", "LOAD_FAST", "LOAD_CLASSDEREF"}
_ATTR_LOAD_OPCODES = {"LOAD_ATTR", "LOAD_METHOD"}


def _argint(ins):
    """Integer oparg of an instruction (MAKE_FUNCTION flags, CALL argc), or 0."""
    a = getattr(ins, "arg", None)
    if isinstance(a, int):
        return a
    v = getattr(ins, "argval", None)
    return v if isinstance(v, int) else 0


def _clean_ident(text) -> str | None:
    """A bare identifier from an argval/argrepr, stripping a ``NULL + `` call-target prefix."""
    if text is None:
        return None
    s = str(text).strip()
    if s.startswith("NULL + "):
        s = s[len("NULL + "):].strip()
    if s.startswith("NULL|"):
        s = s.split("|", 1)[1].strip()
    return s if s.isidentifier() else None


def _name_of(ins) -> str | None:
    v = getattr(ins, "argval", None)
    if isinstance(v, str):
        c = _clean_ident(v)
        if c is not None:
            return c
    return _clean_ident(getattr(ins, "argrepr", None))


def _is_code_object(value) -> bool:
    return hasattr(value, "co_argcount") and hasattr(value, "co_varnames")


class _DefSite:
    """One ``MAKE_FUNCTION`` definition site in a parent code object's instruction stream."""
    __slots__ = ("name", "flags", "codeobj", "defaults", "defaults_ok",
                 "decorator", "decorator_count", "clean")

    def __init__(self, name, flags, codeobj):
        self.name = name
        self.flags = flags
        self.codeobj = codeobj
        self.defaults = None          # recovered const tuple of positional defaults (or None)
        self.defaults_ok = False      # True iff a clean const-tuple defaults was recovered
        self.decorator = None         # recovered decorator expression string (or None)
        self.decorator_count = 0      # number of decorator-application CALLs after MAKE_FUNCTION
        self.clean = True             # False if the region was too tangled to interpret


def _decorator_expr_before(seq: list, code_idx: int) -> str | None:
    """Recover a single simple decorator expression from the ``LOAD`` run immediately before
    the code-object load at ``code_idx``: a bare name (``LOAD_NAME/GLOBAL/DEREF/FAST``) or a
    dotted attribute chain (``base.attr...``). Returns the expression string or None."""
    k = code_idx - 1
    attrs: list[str] = []
    while k >= 0 and getattr(seq[k], "opname", None) in _ATTR_LOAD_OPCODES:
        a = _name_of(seq[k])
        if a is None:
            return None
        attrs.append(a)
        k -= 1
    if k < 0 or getattr(seq[k], "opname", None) not in _NAME_LOAD_OPCODES:
        return None
    base = _name_of(seq[k])
    if base is None:
        return None
    parts = [base] + list(reversed(attrs))
    expr = ".".join(parts)
    # Validate the expression is a clean Name / dotted-Attribute (never a call/subscript).
    try:
        node = ast.parse(expr, mode="eval").body
    except SyntaxError:
        return None
    cur = node
    while isinstance(cur, ast.Attribute):
        cur = cur.value
    if not isinstance(cur, ast.Name):
        return None
    return expr


def _def_sites(insts: list) -> dict[str, _DefSite]:
    """Map each stored function name -> its (unique) ``_DefSite``. A name that appears at more
    than one site is marked ``clean = False`` (ambiguous). Never raises."""
    seq = [i for i in insts if getattr(i, "opname", None) not in (None, "CACHE")]
    sites: dict[str, _DefSite] = {}
    for idx, ins in enumerate(seq):
        if getattr(ins, "opname", None) != "MAKE_FUNCTION":
            continue
        if idx == 0:
            continue
        code_ins = seq[idx - 1]
        if getattr(code_ins, "opname", None) != "LOAD_CONST":
            continue
        codeobj = getattr(code_ins, "argval", None)
        if not _is_code_object(codeobj):
            continue
        flags = _argint(ins)

        # Walk MAKE_FUNCTION -> STORE, counting decorator-application CALLs. Anything else in
        # between (a nested expr, a def used as an argument) makes this site un-interpretable.
        j = idx + 1
        store_name = None
        decorator_count = 0
        clean = True
        while j < len(seq):
            op = getattr(seq[j], "opname", None)
            if op in _CALL_OPCODES:
                decorator_count += 1
                j += 1
                continue
            if op in _STORE_OPCODES:
                store_name = _name_of(seq[j])
                break
            clean = False
            break
        if store_name is None:
            continue

        site = _DefSite(store_name, flags, codeobj)
        site.decorator_count = decorator_count
        site.clean = clean

        # DEFAULTS recovery: only the simple, dominant case where positional defaults are the
        # *only* MAKE_FUNCTION attribute (flags == 0x01) so the const tuple sits immediately
        # before the code load. Any other flag combination is left unrecovered (defers later).
        if flags == _FLAG_DEFAULTS and idx >= 2:
            defaults_ins = seq[idx - 2]
            if (getattr(defaults_ins, "opname", None) == "LOAD_CONST"
                    and isinstance(getattr(defaults_ins, "argval", None), tuple)):
                site.defaults = defaults_ins.argval
                site.defaults_ok = True

        # DECORATOR recovery: a single decorator, no other MAKE_FUNCTION attributes (flags == 0)
        # so the pre-code region is purely the decorator load(s).
        if flags == 0 and decorator_count == 1 and clean:
            site.decorator = _decorator_expr_before(seq, idx - 1)

        if store_name in sites:
            sites[store_name].clean = False  # duplicate name -> ambiguous
        else:
            sites[store_name] = site
    return sites


def _positional_params(fn: ast.AST) -> list[ast.arg]:
    args = fn.args
    return list(getattr(args, "posonlyargs", [])) + list(args.args)


def _unique_funcdef(tree: ast.AST, name: str) -> ast.AST | None:
    matches = [
        n for n in ast.walk(tree)
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == name
    ]
    return matches[0] if len(matches) == 1 else None


def _line_maps(fragment: str):
    orig = fragment.splitlines(keepends=True)
    dedented = textwrap.dedent(fragment)
    ded = dedented.splitlines(keepends=True)
    return orig, dedented, ded


def _orig_col(orig: list, ded: list, lineno: int, ded_col: int) -> int | None:
    """Map a dedented (lineno, col) to a char column in the original line. Returns None if the
    line is out of range or non-ASCII (col_offset is a utf-8 byte offset)."""
    li = lineno - 1
    if li < 0 or li >= len(orig) or li >= len(ded):
        return None
    if not orig[li].isascii() or not ded[li].isascii():
        return None
    return ded_col + (len(orig[li]) - len(ded[li]))


def _apply_default_edits(fragment: str, fn: ast.AST, targets: list[tuple[ast.arg, str]]) -> str | None:
    """Insert ``=<repr>`` after each ``(arg, repr_text)`` target, editing the original
    fragment from the last position backward so earlier offsets stay valid."""
    orig, _dedented, ded = _line_maps(fragment)
    edits: list[tuple[int, int, str]] = []
    for arg, rhs in targets:
        li = (arg.end_lineno or 0) - 1
        col = _orig_col(orig, ded, arg.end_lineno, arg.end_col_offset)
        if col is None or li < 0 or li >= len(orig):
            return None
        edits.append((li, col, f"={rhs}"))
    lines = list(orig)
    for li, col, text in sorted(edits, key=lambda e: (e[0], e[1]), reverse=True):
        line = lines[li]
        if col > len(line):
            return None
        lines[li] = line[:col] + text + line[col:]
    return "".join(lines)


def _apply_decorator_edit(fragment: str, fn: ast.AST, expr: str) -> str | None:
    orig, _dedented, ded = _line_maps(fragment)
    li = (fn.lineno or 0) - 1
    if li < 0 or li >= len(orig):
        return None
    def_line = orig[li]
    indent = def_line[:len(def_line) - len(def_line.lstrip())]
    new_line = f"{indent}@{expr}\n"
    lines = list(orig)
    lines.insert(li, new_line)
    return "".join(lines)


def _default_candidate(gt_site: _DefSite, der_site: _DefSite, tree: ast.AST,
                       fragment: str) -> dict | None:
    """Build a DEFAULTS candidate for one diverging name, or None."""
    if not gt_site.defaults_ok or gt_site.defaults is None:
        return None
    if der_site.flags & _FLAG_DEFAULTS:
        return None  # derived already has positional defaults; not a clean "dropped" case
    # The sole difference must be the positional-defaults bit.
    if gt_site.flags != (der_site.flags | _FLAG_DEFAULTS):
        return None

    defaults = gt_site.defaults
    if not defaults:
        return None
    # Every default value must round-trip so we re-emit it verbatim.
    rendered: list[str] = []
    for value in defaults:
        if not _round_trips(value):
            return None
        rhs = repr(value)
        try:
            if ast.literal_eval(rhs) != value:
                return None
        except Exception:  # noqa: BLE001
            return None
        rendered.append(rhs)

    fn = _unique_funcdef(tree, gt_site.name)
    if fn is None:
        return None
    # No positional param may already carry a default in the derived def.
    if fn.args.defaults:
        return None
    positional = _positional_params(fn)
    argcount = getattr(gt_site.codeobj, "co_argcount", None)
    if argcount is None or len(positional) != argcount:
        return None
    # Parameter names must match GT's positional varnames (strong alignment).
    gt_names = list(getattr(gt_site.codeobj, "co_varnames", ()))[:argcount]
    if [p.arg for p in positional] != gt_names:
        return None
    k = len(defaults)
    if k > len(positional):
        return None
    target_params = positional[len(positional) - k:]
    targets = list(zip(target_params, rendered))

    new_fragment = _apply_default_edits(fragment, fn, targets)
    if new_fragment is None or new_fragment == fragment:
        return None
    try:
        ast.parse(textwrap.dedent(new_fragment))
    except SyntaxError:
        return None

    preview = ", ".join(f"{p.arg}={r}" for p, r in targets)
    if len(preview) > 60:
        preview = preview[:57] + "..."
    return {
        "text": new_fragment,
        "operator": f"restore default(s) {gt_site.name}({preview})",
        "opname": "MAKE_FUNCTION",
        "kind": "decorator_default",
        "confidence": "unique",
    }


def _decorator_candidate(gt_site: _DefSite, der_site: _DefSite, tree: ast.AST,
                         fragment: str) -> dict | None:
    """Build a DECORATOR candidate for one diverging name, or None."""
    if gt_site.decorator is None or gt_site.decorator_count != 1:
        return None
    if der_site.decorator_count != 0:
        return None
    # Pure decorator difference: no default/annotation/closure entanglement on either side.
    if gt_site.flags != 0 or der_site.flags != 0:
        return None

    fn = _unique_funcdef(tree, gt_site.name)
    if fn is None:
        return None
    if getattr(fn, "decorator_list", None):
        return None  # derived def already decorated; not a clean "dropped" case

    new_fragment = _apply_decorator_edit(fragment, fn, gt_site.decorator)
    if new_fragment is None or new_fragment == fragment:
        return None
    try:
        ast.parse(textwrap.dedent(new_fragment))
    except SyntaxError:
        return None

    return {
        "text": new_fragment,
        "operator": f"restore decorator @{gt_site.decorator} on {gt_site.name}",
        "opname": "MAKE_FUNCTION",
        "kind": "decorator_default",
        "confidence": "unique",
    }


def decorator_default_candidate(gt_code_object, derived_code_object, fragment: str,
                                repair_context: dict | None) -> dict | None:
    """Restore a dropped positional default value or ``@decorator`` from GT ``MAKE_FUNCTION``
    bytecode. Returns ``{"text", "operator", "opname", "kind", "confidence"}`` or None to
    defer. Never raises."""
    try:
        if not repair_context:
            return None
        failed = repair_context.get("pylingual_failed_result") or {}
        if failed.get("message") != "Different bytecode":
            return None

        gt = _instructions(gt_code_object)
        der = _instructions(derived_code_object)
        if not gt:
            return None

        gt_sites = _def_sites(gt)
        der_sites = _def_sites(der)
        if not gt_sites:
            return None

        try:
            tree = ast.parse(textwrap.dedent(fragment))
        except SyntaxError:
            return None

        # Collect every name that diverges in a repairable way. Unique-or-defer: exactly one
        # candidate across BOTH regimes may apply.
        candidates: list[dict] = []
        for name, gt_site in gt_sites.items():
            if not gt_site.clean:
                continue
            der_site = der_sites.get(name)
            if der_site is None or not der_site.clean:
                continue
            cand = _default_candidate(gt_site, der_site, tree, fragment)
            if cand is not None:
                candidates.append(cand)
            cand = _decorator_candidate(gt_site, der_site, tree, fragment)
            if cand is not None:
                candidates.append(cand)

        if len(candidates) != 1:
            return None
        return candidates[0]
    except Exception:  # noqa: BLE001 - a bad candidate is rejected by the oracle gate; a
        # raised exception would crash the repair loop, so swallow everything.
        return None
