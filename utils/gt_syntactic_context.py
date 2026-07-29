"""Ground-truth structure briefs for the SYNTACTIC repair prompt.

When the syntactic-repair LLM faces a broken decompiled code object it cannot reconstruct
blind -- leaked PEP-695 generics (`def <generic parameters of cc_zip>(.defaults):`,
`class .type_params(...)`), stray `@overload` stacks, `# ***<qualname>: Failure` markers --
this module gives it the object's TRUE structure (signature/placement, plus a compact
control-flow/nesting skeleton for objects that have one), read directly from the
ground-truth `.pyc` alongside the broken source.

The broken source is, by definition, not valid Python (that is why syntactic repair is
running at all), so it has no AST. `_enclosing_qualname` therefore scans lines as text,
walking upward from the error line for the nearest `def`/`class` header at each new
(shallower) indent level, to recover the dotted qualname of the object enclosing the error.
That qualname is then matched against the GT `.pyc`'s own code-object tree (built once via
the project's crash-safe pyc loader) and rendered through the semantic-side GT brief
generator.

Pure best-effort utility: `build_gt_object_context` NEVER raises. Any failure (no gt path,
missing file, load error, module-scope error with no enclosing object, no name match) yields
"".
"""
from __future__ import annotations

import os
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

_HDR = re.compile(r'^(\s*)(?:async\s+def|def|class)\s+([A-Za-z_]\w*)')

# Bounds for the CONTROL-FLOW/STRUCTURE enrichment appended below the signature brief --
# keeps the addition compact enough to never bloat the syntactic-repair prompt.
_MAX_CONTROL_FLOW_ITEMS = 20
_MAX_CONTROL_FLOW_CHARS = 700


def _enclosing_qualname(lines: list[str], err_line: int) -> str:
    """Scans TEXT (no AST -- the broken source may not parse) upward from err_line for the
    nearest def/class header at each new shallower indent, building a dotted qualname of the
    object enclosing the error. Mirrors how a human would read nesting from indentation."""
    parts: list[str] = []
    cur = None
    for i in range(min(err_line, len(lines)) - 1, -1, -1):
        m = _HDR.match(lines[i])
        if not m:
            continue
        ind = len(m.group(1))
        if cur is None or ind < cur:
            parts.append(m.group(2))
            cur = ind
            if ind == 0:
                break
    return ".".join(reversed(parts))


def _iter_gt_nodes(root: Any):
    """Depth-first traversal of the GT EditableBytecode tree: root then each
    child_bytecodes subtree, recursively."""
    yield root
    for child in getattr(root, "child_bytecodes", None) or ():
        yield from _iter_gt_nodes(child)


def _find_gt_node(root: Any, qual: str) -> Any:
    """Matches `qual` against GT node dotted names. Prefers an exact match or a node whose
    name ends with `.` + qual; falls back to a node whose LEAF name equals qual. Takes the
    first match in traversal order when there are duplicates (e.g. repeated @overload
    stubs sharing a leaf name)."""
    nodes = list(_iter_gt_nodes(root))

    for node in nodes:
        name = getattr(node, "name", None)
        if name is None:
            continue
        if name == qual or name.endswith("." + qual):
            return node

    for node in nodes:
        name = getattr(node, "name", None)
        if name is None:
            continue
        if name.split(".")[-1] == qual:
            return node

    return None


def _capped_control_flow_brief(code_object: Any) -> str:
    """Best-effort CONTROL-FLOW/STRUCTURE section for the enrichment appended below the
    signature brief. Reuses the semantic-side GT control-flow generator (`_gt_control_flow_brief`
    -- bytecode-only, version-drift-tolerant) so the syntactic and semantic pipelines describe
    control flow the same way. Item-capped at the source (`max_items`) and additionally
    char-capped here so a handful of long try/call lines can't blow the budget either.

    Pure best-effort: returns "" for straight-line code (nothing to add) AND on ANY failure
    (missing/incompatible generator, bad code object, exception of any kind). Callers must be
    able to treat "" as ordinary graceful degradation -- this function NEVER raises, so a
    failure here can never cost the caller the signature-only brief it already has."""
    try:
        from utils.reattach_source_code_object import _gt_control_flow_brief

        text = _gt_control_flow_brief(code_object, max_items=_MAX_CONTROL_FLOW_ITEMS)
    except Exception:
        return ""
    if not text:
        return ""
    if len(text) > _MAX_CONTROL_FLOW_CHARS:
        text = text[:_MAX_CONTROL_FLOW_CHARS].rstrip() + "\n- ... (truncated)"
    return text


def build_gt_object_context(broken_source: str, error_line: int, gt_pyc_path: str) -> str:
    """Returns a short human-readable brief of the TRUE structure (signature/placement) of
    the code object enclosing `error_line` in `broken_source`, drawn from the ground-truth
    `.pyc` at `gt_pyc_path`. Returns "" on ANY failure -- never raises."""
    if not gt_pyc_path:
        return ""

    try:
        lines = broken_source.splitlines()
        qual = _enclosing_qualname(lines, error_line)
        if not qual:
            return ""

        gt_path = Path(gt_pyc_path)
        if not gt_path.is_file():
            return ""

        # Deferred import: pyc loading is heavy (xdis + the vendored pylingual package on
        # sys.path) and a broken/missing gt path should never pay for it.
        from utils.pyc_code_object_distance import load_editable_bytecode_from_pyc

        root = load_editable_bytecode_from_pyc(gt_path)

        node = _find_gt_node(root, qual)
        if node is None:
            return ""

        from utils.reattach_source_code_object import _gt_signature_and_placement_brief

        code_object = getattr(node, "codeobj", None)
        if code_object is None:
            code_object = node

        brief = _gt_signature_and_placement_brief(code_object, qual)
        if not brief:
            return ""

        header = (
            f"GROUND-TRUTH structure of the enclosing object `{qual}` "
            "(reconstruct to match this; the broken lines are decompiler artifacts):\n"
        )
        result = header + brief

        # Additive enrichment: a decompiler-flattened body needs the object's control-flow
        # SHAPE, not just its bare signature, to be re-nested correctly. Failure here (no
        # control flow, generator unavailable, exception) is fully absorbed by
        # `_capped_control_flow_brief` -- it can only ever add to `result`, never subtract
        # from it, so the signature-only brief always survives unchanged.
        control_flow = _capped_control_flow_brief(code_object)
        if control_flow:
            result += "\nControl flow (nesting to reconstruct):\n" + control_flow

        # Lever 1 (SYNTACTIC_CFR_DIRECTIVE, default OFF): a SOURCE-LEVEL control-flow directive
        # recovered from the GT bytecode's construct tree (try/if/for/while/with with recovered
        # headers, via structural_directive.gt_structure_directive) -- the readable nesting the
        # offset-based brief above cannot convey. Targets the ~47% of malware decompilations
        # whose failure is decompiler control-flow-reconstruction damage (orphaned statements,
        # dangling else/except). Purely additive; absorbs its own failures.
        if os.getenv("SYNTACTIC_CFR_DIRECTIVE", "0") == "1":
            try:
                from utils.semantic_operators.structural_directive import gt_structure_directive

                directive = gt_structure_directive(node)
            except Exception:
                directive = ""
            if directive:
                result += "\n" + directive

        return result
    except Exception:
        return ""


# ---------------------------------------------------------------------------
# Constant-sequence harvest (GT co_consts splice)
# ---------------------------------------------------------------------------
# PyLingual truncates a constant sequence after 51 elements, leaving a dangling opening quote
# inside an unclosed bracket. `balance_delimiters` closes it, so the file compiles and the repair
# reports zero deleted lines -- while every element past the 51st is gone. The complete sequence
# is still in the GT `.pyc` the pipeline already has on disk, so the faithful repair is mechanical.
# These helpers supply `syntactic_prepass.recover_truncated_literal` with its candidate pool.

_MIN_SEQUENCE_LENGTH = 2
_MAX_SEQUENCES = 5000
_MAX_WALK_NODES = 20000


def constant_sequences_from_code(code_object, max_sequences=_MAX_SEQUENCES,
                                 min_length=_MIN_SEQUENCE_LENGTH):
    """Every tuple/frozenset constant in `code_object`'s tree, as a de-duplicated list.

    Walks nested code objects, because the truncated literal is usually inside a function rather
    than at module scope. Only sequences of str/int/bytes are kept -- those are what a source-level
    literal can be rebuilt from, and admitting anything else would offer `recover_truncated_literal`
    candidates it must reject anyway.

    Both the node walk and the result set are bounded: packed malware routinely carries thousands
    of constants, and an unbounded harvest would turn a per-file prepass into a memory problem.
    Returns [] for anything that is not a code object; never raises.
    """
    try:
        if code_object is None or not hasattr(code_object, "co_consts"):
            return []
        found, seen = [], set()
        pending, visited = [code_object], 0
        while pending and len(found) < max_sequences and visited < _MAX_WALK_NODES:
            current = pending.pop()
            visited += 1
            for const in getattr(current, "co_consts", ()) or ():
                if hasattr(const, "co_consts"):
                    pending.append(const)
                    continue
                if not isinstance(const, (tuple, frozenset)):
                    continue
                items = tuple(const)
                if len(items) < min_length:
                    continue
                if not all(isinstance(x, (str, int, bytes)) for x in items):
                    continue
                key = (isinstance(const, frozenset), items)
                if key in seen:
                    continue
                seen.add(key)
                found.append(const)
                if len(found) >= max_sequences:
                    break
        return found
    except Exception:
        return []


def _harvest_uncached(gt_pyc_path, max_sequences):
    """Load the `.pyc` and harvest. Returns a tuple so the cache holds an immutable value."""
    try:
        path = Path(gt_pyc_path)
        if not path.is_file():
            return ()
        # Deferred: pyc loading pulls in xdis and the vendored pylingual package.
        from utils.pyc_code_object_distance import load_editable_bytecode_from_pyc

        root = load_editable_bytecode_from_pyc(path)
        code_object = getattr(root, "codeobj", None) or root
        return tuple(constant_sequences_from_code(code_object, max_sequences=max_sequences))
    except Exception:
        return ()


@lru_cache(maxsize=4)
def _harvest_cached(gt_pyc_path, max_sequences):
    return _harvest_uncached(gt_pyc_path, max_sequences)


def harvest_constant_sequences(gt_pyc_path, max_sequences=_MAX_SEQUENCES):
    """`constant_sequences_from_code` over the code-object tree of the `.pyc` at `gt_pyc_path`.

    A single file is probed by the prepass once before the LLM and again after each LLM candidate,
    so the same ground truth is asked for repeatedly; loading it through xdis every time would
    multiply a seconds-long load by the retry budget. The cache is deliberately tiny (4 entries):
    the access pattern is one file at a time, and packed-malware constant pools are large enough
    that retaining more would cost real memory for no hit-rate.

    Returns [] on ANY failure (no path, missing file, unreadable/foreign-version bytecode) -- a
    missing ground truth must degrade the repair to the blind close, never abort the file.
    """
    if not gt_pyc_path:
        return []
    return list(_harvest_cached(str(gt_pyc_path), max_sequences))


harvest_constant_sequences.cache_clear = _harvest_cached.cache_clear
