"""3.14 works out of the box via xdis; 3.15 needs our regeneratable shim (magic 3666 + opcode
table). These tests pin: (a) the shim installs and registers a correct opcode_315 (pure, fast),
and (b) the repair oracle parses real 3.14 and 3.15 .pyc files and scores them so that divergent
code objects get distance > 0 while identical ones get 0 (the property the whole repair loop
relies on). The compile-based checks skip gracefully if a uv-managed interpreter isn't available."""
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "pylingual"))

import utils.py315_support as py315
from utils.reattach_source_code_object import run_comparison


def _uv_compile(version: str, src: Path, out: Path) -> bool:
    env = {**os.environ, "UV_CACHE_DIR": os.environ.get("UV_CACHE_DIR", "/tmp/uv-cache")}
    try:
        r = subprocess.run(
            ["uvx", "--no-config", "--python", version, "python", "-c",
             f"import py_compile;py_compile.compile({str(src)!r},cfile={str(out)!r})"],
            env=env, capture_output=True, timeout=240,
        )
    except Exception:
        return False
    return out.exists()


class Py315ShimTest(unittest.TestCase):
    def test_install_registers_opcode_315(self):
        self.assertTrue(py315.install())
        import xdis.opcodes
        mod = xdis.opcodes.opcode_315
        # 3.15's HAVE_ARGUMENT is 41 — the value that makes BINARY_OP decode its operand
        self.assertEqual(mod.HAVE_ARGUMENT, 41)
        # BINARY_OP present and above HAVE_ARGUMENT so its arg (+/-/*) is decoded
        self.assertIn("BINARY_OP", mod.opmap)
        self.assertGreaterEqual(mod.opmap["BINARY_OP"], mod.HAVE_ARGUMENT)
        # magic 3666 resolves
        import xdis.magics as magics
        self.assertIn("3.15", str(magics.versions.get(magics.int2magic(3666))))


class OracleAcrossVersionsTest(unittest.TestCase):
    def _distance(self, version, a_src, b_src):
        py315.install()
        d = Path(tempfile.mkdtemp())
        (d / "a.py").write_text(a_src)
        (d / "b.py").write_text(b_src)
        if not (_uv_compile(version, d / "a.py", d / "a.pyc") and _uv_compile(version, d / "b.py", d / "b.pyc")):
            self.skipTest(f"uv-managed Python {version} unavailable")
        same = run_comparison(d / "a.pyc", d / "a.pyc").get("combined_distance")
        diff = run_comparison(d / "a.pyc", d / "b.pyc").get("combined_distance")
        return same, diff

    def test_314_oracle_operator_diff(self):
        same, diff = self._distance("3.14", "def f(x):\n    return x + 1\n", "def f(x):\n    return x - 1\n")
        self.assertEqual(same, 0)
        self.assertIsInstance(diff, int)
        self.assertGreater(diff, 0)

    def test_315_oracle_operator_diff(self):
        same, diff = self._distance("3.15", "def f(x):\n    return x + 1\n", "def f(x):\n    return x - 1\n")
        self.assertEqual(same, 0)
        self.assertIsInstance(diff, int)
        self.assertGreater(diff, 0)

    def test_315_oracle_controlflow_diff(self):
        same, diff = self._distance(
            "3.15",
            "def f(x):\n    if x > 0:\n        return 1\n    return 2\n",
            "def f(x):\n    if x < 0:\n        return 1\n    return 2\n",
        )
        self.assertEqual(same, 0)
        self.assertGreater(diff, 0)


if __name__ == "__main__":
    unittest.main()
