"""Regenerate utils/py315_support/opcode_315_data.json from a target CPython 3.15 interpreter.

3.15 is a moving target until GA — its magic number and opcode table can change between
pre-releases. This tool re-extracts the authoritative metadata (magic, opmap, HAVE_ARGUMENT,
has* sets, jump classification) from whatever 3.15 the host resolves via uv, so re-supporting a
new beta/RC/GA is a single command:

    python tools/regen_315_opcodes.py            # uses uv --python 3.15
    python tools/regen_315_opcodes.py 3.15.0     # pin an exact version

The shim (utils/py315_support) reads magic + opcodes from the JSON, so nothing else changes.
"""
import json
import os
import subprocess
import sys
from pathlib import Path

OUT = Path(__file__).resolve().parent.parent / "utils" / "py315_support" / "opcode_315_data.json"

_PROBE = r"""
import json, opcode, sys, importlib.util
inv = {v: k for k, v in opcode.opmap.items()}
jset = set(opcode.hasjrel) | set(opcode.hasjabs)
def _conditional(name):
    return ("IF" in name or "FOR_ITER" in name or name.startswith("SEND")
            or "END_ASYNC_FOR" in name)
out = {
    "version": ".".join(map(str, sys.version_info[:3])),
    "magic": int.from_bytes(importlib.util.MAGIC_NUMBER[:2], "little"),
    "HAVE_ARGUMENT": opcode.HAVE_ARGUMENT,
    "opmap": dict(opcode.opmap),
    "hasjrel": sorted(opcode.hasjrel),
    "hasjabs": sorted(opcode.hasjabs),
    "hasconst": sorted(opcode.hasconst),
    "hasname": sorted(opcode.hasname),
    "haslocal": sorted(opcode.haslocal),
    "hasfree": sorted(opcode.hasfree),
    "hascompare": sorted(opcode.hascompare),
    "jumps": {inv[o]: _conditional(inv[o]) for o in sorted(jset)},
}
print(json.dumps(out, indent=1))
"""


def main() -> int:
    version = sys.argv[1] if len(sys.argv) > 1 else "3.15"
    env = {**os.environ, "UV_CACHE_DIR": os.environ.get("UV_CACHE_DIR", "/tmp/uv-cache")}
    proc = subprocess.run(
        ["uvx", "--no-config", "--python", version, "python", "-c", _PROBE],
        env=env, capture_output=True, text=True, timeout=300,
    )
    if proc.returncode != 0 or not proc.stdout.strip():
        sys.stderr.write(f"extraction failed for {version}:\n{proc.stderr[-2000:]}\n")
        return 2
    data = json.loads(proc.stdout)
    OUT.write_text(json.dumps(data, indent=1))
    print(f"wrote {OUT}")
    print(f"  version={data['version']} magic={data['magic']} "
          f"HAVE_ARGUMENT={data['HAVE_ARGUMENT']} opcodes={len(data['opmap'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
