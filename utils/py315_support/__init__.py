"""Runtime support for parsing/scoring CPython 3.15 bytecode with xdis 6.3.0.

xdis 6.3.0 ships opcode modules through 3.14 and knows 3.15 *alpha* magics, but not the
3.15.0 magic (3666) and has no 3.15 opcode module — so the repair oracle fails to parse a 3.15
``.pyc`` with ``Unknown magic number 3666``. This module builds an ``opcode_315`` table at
runtime using xdis's OWN builders, driven by authoritative opcode metadata extracted from the
3.15 interpreter (``opcode_315_data.json``, regeneratable via tools/regen_315_opcodes.py), and
registers it + the magic. It touches only in-memory xdis state (nothing in site-packages).

3.15 is a beta target: its magic/opcodes can still move before GA. Because everything is derived
from ``opcode_315_data.json``, re-supporting the final release is one regenerate + reimport.
"""
from __future__ import annotations

import json
import types
from pathlib import Path

_DATA_PATH = Path(__file__).with_name("opcode_315_data.json")
_installed = False


def _build_opcode_315_module():
    import xdis.opcodes.opcode_3x.opcode_314 as op314
    from xdis.opcodes import base as B

    data = json.loads(_DATA_PATH.read_text())
    opmap: dict = data["opmap"]
    hasjrel = set(data["hasjrel"])
    hasjabs = set(data["hasjabs"])
    hasconst = set(data["hasconst"])
    hasname = set(data["hasname"])
    haslocal = set(data["haslocal"])
    hasfree = set(data["hasfree"])
    jumps: dict = data["jumps"]  # name -> conditional?
    have_argument = int(data["HAVE_ARGUMENT"])
    num2name = {v: k for k, v in opmap.items()}

    # pop/push classification borrowed from xdis's 3.14 module, keyed by opcode NAME (shared
    # opcodes keep their stack behaviour across 3.14->3.15; only opcode NUMBERS were renumbered).
    ref_pop, ref_push = {}, {}
    for i, nm in enumerate(op314.opname):
        if nm and not nm.startswith("<"):
            ref_pop[nm] = op314.oppop[i] if i < len(op314.oppop) else 0
            ref_push[nm] = op314.oppush[i] if i < len(op314.oppush) else 0

    mod = types.ModuleType("xdis.opcodes.opcode_3x.opcode_315")
    loc = mod.__dict__
    B.init_opdata(loc, None, (3, 15))

    max_op = max(opmap.values())
    if max_op >= len(loc["opname"]):
        loc["opname"].extend([f"<{i}>" for i in range(len(loc["opname"]), max_op + 1)])
        loc["oppop"].extend([0] * (max_op + 1 - len(loc["oppop"])))
        loc["oppush"].extend([0] * (max_op + 1 - len(loc["oppush"])))

    for num in sorted(num2name):
        name = num2name[num]
        pop = ref_pop.get(name, 0)
        push = ref_push.get(name, 0)
        if num in hasjabs:
            B.jabs_op(loc, name, num, pop, push, conditional=bool(jumps.get(name)))
        elif num in hasjrel:
            B.jrel_op(loc, name, num, pop, push, conditional=bool(jumps.get(name)))
        elif num in hasconst:
            B.const_op(loc, name, num, pop, push)
        elif num in hasname:
            B.name_op(loc, name, num, pop, push)
        elif num in haslocal:
            B.local_op(loc, name, num, pop, push)
        elif num in hasfree:
            B.free_op(loc, name, num, pop, push)
        else:
            B.def_op(loc, name, num, pop, push)

    # CRITICAL: 3.15's HAVE_ARGUMENT is 41 (3.14 uses 44). init_opdata leaves the base default
    # (90), which would make opcodes 41..89 (incl. BINARY_OP) decode without their arg -> the
    # +/-/* distinction vanishes and divergent files wrongly score distance 0.
    loc["HAVE_ARGUMENT"] = have_argument
    loc["hasarg"] = sorted(n for n in opmap.values() if n >= have_argument)
    loc["hasjabs"] = sorted(hasjabs)
    loc["hasjump"] = loc["hasjrel"]
    loc["hasexc"] = sorted(
        opmap[n] for n in ("SETUP_CLEANUP", "SETUP_FINALLY", "SETUP_WITH") if n in opmap
    )
    # reuse 3.14's arg-formatting (BINARY_OP nb_ops table etc.) — same operand semantics — but
    # wrap each in a guard: argrepr is display-only, so a bad/None arg must never abort the whole
    # disassembly (which would desync the instruction stream and corrupt the distance).
    def _safe(fn):
        def _g(arg):
            try:
                return fn(arg)
            except Exception:
                return str(arg)
        return _g

    loc["opcode_arg_fmt"] = {k: _safe(v) for k, v in getattr(op314, "opcode_arg_fmt314", {}).items()}
    loc["opcode_extended_fmt"] = dict(getattr(op314, "opcode_extended_fmt314", {}))
    if hasattr(op314, "findlinestarts"):
        loc["findlinestarts"] = op314.findlinestarts
    loc["version_tuple"] = (3, 15)
    loc["python_implementation"] = getattr(op314, "python_implementation", "CPython")

    B.update_pj3(loc, loc)
    B.finalize_opcodes(loc)
    return mod


def install() -> bool:
    """Register 3.15 with xdis (idempotent). Returns True if 3.15 is now parseable."""
    global _installed
    if _installed:
        return True
    import xdis.magics as magics
    import xdis.op_imports as op_imports

    import sys
    import xdis.opcodes

    data = json.loads(_DATA_PATH.read_text())
    ver_str = str(data.get("version", "3.15.0"))
    canonic = ".".join(ver_str.split(".")[:2]) + ".0"  # e.g. "3.15.0"

    mod = _build_opcode_315_module()
    # pylingual's PYCFile does getattr(xdis.opcodes, "opcode_315"), so expose it as an attribute
    # of the package and in sys.modules, in addition to the op_imports dict xdis's own loader uses.
    setattr(xdis.opcodes, "opcode_315", mod)
    sys.modules["xdis.opcodes.opcode_315"] = mod
    sys.modules["xdis.opcodes.opcode_3x.opcode_315"] = mod
    for key in ("3.15", "3.15.0", ver_str, canonic, 3.15):
        op_imports.op_imports[key] = mod
    # magic is READ FROM THE DATA FILE (regenerated per release), NOT hardcoded — a GA/new beta
    # is one `tools/regen_315_opcodes.py` run away from being supported.
    magic_int = int(data.get("magic")) if data.get("magic") is not None else None
    if magic_int is not None:
        magics.add_magic_from_int(magic_int, canonic)
    # any 3.15 magic xdis already knows (alphas) should also resolve to our module
    for ver in set(magics.versions.values()):
        if str(ver).startswith("3.15"):
            op_imports.op_imports.setdefault(ver, mod)
    _installed = True
    return True
