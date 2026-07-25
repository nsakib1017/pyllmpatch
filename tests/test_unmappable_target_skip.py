"""An oracle-selected MATCHED repair target whose qualname has no source mapping (e.g. a PEP-695
synthetic type-param/alias object) must not abort the whole-file repair loop. `_find_target_row`
raising `ReattachError("No mapped source code object found for qualname: ...")` for one target
should be skipped (logged, `continue`) so the loop still repairs every OTHER target instead of
propagating out of `repair_mismatching_code_objects` entirely."""
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "pylingual"))

import utils.reattach_source_code_object as R
from utils.generate_bytecode import compile_version

HOST = f"{sys.version_info[0]}.{sys.version_info[1]}"


def _scenario(gt_src: str, der_src: str) -> Path:
    d = Path(tempfile.mkdtemp())
    (d / "gt.py").write_text(gt_src)
    (d / "der.py").write_text(der_src)
    compile_version(str(d / "gt.py"), str(d / "gt.pyc"), HOST)
    compile_version(str(d / "der.py"), str(d / "der.pyc"), HOST)
    return d


GT = "def a(x):\n    return x + 1\n\ndef b(x):\n    return x + 2\n"
DER = "def a(x):\n    return x - 1\n\ndef b(x):\n    return x - 2\n"


class UnmappableMatchedTargetSkipTest(unittest.TestCase):
    """Drives `repair_mismatching_code_objects` end to end (the function that owns the
    MATCHED-target loop and the `_find_target_row` call site) with `_find_target_row`
    monkeypatched to raise `ReattachError` for qualname `<module>.a` only, delegating to the
    real implementation for everything else (`<module>.b`). Both `a` and `b` are simple leaf
    divergences; the pre-LLM deterministic prepass is disabled so both survive to reach the
    MATCHED-target loop (mirrors the existing `SEMANTIC_DETERMINISTIC_PREPASS=False` pattern in
    tests/test_post_llm_deterministic.py) and `fragment_fixer=None` exercises the GT-verbatim
    fallback so `b`'s fix is deterministic without needing a real fixer callable."""

    def setUp(self):
        self._prev_mode = R.ACCEPTANCE_MODE

    def tearDown(self):
        R.ACCEPTANCE_MODE = self._prev_mode

    def test_unmappable_target_is_skipped_and_other_targets_still_repaired(self):
        d = _scenario(GT, DER)
        original_find_target_row = R._find_target_row

        def _patched_find_target_row(source_path, pyc_path, qualname, strict_map=False, target_identity=None):
            if qualname == "<module>.a":
                raise R.ReattachError(f"No mapped source code object found for qualname: {qualname}")
            return original_find_target_row(
                source_path, pyc_path, qualname, strict_map=strict_map, target_identity=target_identity
            )

        with mock.patch.object(R, "SEMANTIC_DETERMINISTIC_PREPASS", False), \
             mock.patch.object(R, "_find_target_row", side_effect=_patched_find_target_row):
            # Must not raise: the unmappable `<module>.a` target must not abort the whole-file loop.
            res = R.repair_mismatching_code_objects(
                d / "gt.pyc", d / "der.pyc", d / "der.py", gt_source=d / "gt.py",
                output_dir=d / "out", fragment_fixer=None, acceptance_mode="oracle",
            )

        final_rows = R.compare_code_object_distances(Path(res["gt_pyc"]), Path(res["final_pyc"]))
        row_a = R._find_distance_row(final_rows, "<module>.a")
        row_b = R._find_distance_row(final_rows, "<module>.b")

        # `b` is mappable and must still get repaired even though `a` was skipped.
        self.assertIsNotNone(row_b)
        self.assertEqual(row_b.get("combined_distance"), 0)
        # `a` is genuinely unmappable and must remain unrepaired (proves it was skipped, not
        # silently fixed through some other path).
        self.assertIsNotNone(row_a)
        self.assertGreater(row_a.get("combined_distance"), 0)

        # The skip must be logged (mirrors the existing graceful deletion/prepass pattern).
        skip_steps = [
            s for s in res["steps"]
            if s.get("qualname") == "<module>.a" and s.get("accepted") is False
            and "could not be mapped" in str(s.get("acceptance_reason") or "")
        ]
        self.assertTrue(skip_steps, f"expected a logged skip step for <module>.a, got steps: {res['steps']}")


if __name__ == "__main__":
    unittest.main()
