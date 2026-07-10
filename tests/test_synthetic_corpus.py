"""CPU-only tests for the synthetic training-data generator.

These tests never touch a GPU: the decompile step is injected as a FAKE that returns
a deliberately-broken source (an inverted branch / a while-loop where GT had an if),
so recompiling it diverges from the GT bytecode and the real oracle repair machinery
manufactures a (brief-ON prompt -> correct-source) training row.

All bytecode work uses the HOST version ("3.12") so compile_version() takes the native
py_compile path -- no uvx, no network, no GPU.
"""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
if str(_REPO_ROOT / "pylingual") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "pylingual"))

from tools.build_synthetic_corpus import build_synthetic_corpus, _module_file_hash

# Host version: native compile, no uvx/network/GPU.
_HOST_VERSION = f"{sys.version_info.major}.{sys.version_info.minor}"

# A correct module with a real if/else branch.
CORRECT = '''def classify(n):
    if n > 0:
        result = "positive"
    else:
        result = "nonpositive"
    return result
'''

# A deliberately-broken "decompiled" version: the if became a while loop, so the
# recompiled bytecode has genuinely different control flow.
BROKEN = '''def classify(n):
    while n > 0:
        result = "positive"
    result = "nonpositive"
    return result
'''


def _fake_decompile_returning(source_text: str):
    """Build a fake decompile_fn that ignores the pyc and always returns source_text.

    Matches the injectable contract decompile_fn(pyc, save_to=..., version=...): it also
    writes save_to (like the real pylingual.decompiler.decompile) so either resolution
    path in the generator works.
    """

    def _fake(pyc, save_to=None, version=None):
        if save_to is not None:
            Path(save_to).write_text(source_text, encoding="utf-8")
        return source_text

    return _fake


def _assert_finetuner_schema(test: unittest.TestCase, row: dict) -> None:
    test.assertIn("prompt", row)
    test.assertIn("replacement_text", row)
    messages = row["prompt"]["messages"]
    test.assertIsInstance(messages, list)
    test.assertGreaterEqual(len(messages), 2)
    for message in messages:
        test.assertTrue(str(message["role"]).strip())
        test.assertTrue(str(message["content"]).strip())
    test.assertIsInstance(row["replacement_text"], str)
    test.assertTrue(row["replacement_text"].strip())


class SyntheticCorpusTest(unittest.TestCase):
    def test_emits_valid_row_from_divergent_fake_decompile(self):
        with tempfile.TemporaryDirectory() as td:
            src_dir = Path(td) / "src"
            src_dir.mkdir()
            (src_dir / "classify.py").write_text(CORRECT, encoding="utf-8")

            rows, stats = build_synthetic_corpus(
                source_dir=src_dir,
                versions=[_HOST_VERSION],
                limit=None,
                holdout_paths=[],
                decompile_fn=_fake_decompile_returning(BROKEN),
            )

            self.assertGreaterEqual(len(rows), 1, f"expected >=1 row; stats={stats}")
            row = rows[0]
            _assert_finetuner_schema(self, row)

            # The user message must carry the GT-bytecode reconstruction brief.
            user = next(m["content"] for m in row["prompt"]["messages"] if m["role"] == "user")
            self.assertTrue(
                ("Ground-truth control-flow shape" in user)
                or ("Reconstruct the missing" in user),
                "user message is missing the GT reconstruction brief text",
            )

            # The label must be the ORIGINAL correct source span, not the broken input.
            self.assertEqual(row["replacement_text"].strip(), CORRECT.strip())
            self.assertNotIn("while", row["replacement_text"])
            self.assertIn("if n > 0:", row["replacement_text"])

            # Integrity: the row's source_file_hash is the manufactured module's hash.
            self.assertEqual(
                row["source_file_hash"], _module_file_hash(src_dir / "classify.py")
            )

    def test_identical_decompile_yields_no_rows(self):
        with tempfile.TemporaryDirectory() as td:
            src_dir = Path(td) / "src"
            src_dir.mkdir()
            (src_dir / "identity.py").write_text(CORRECT, encoding="utf-8")

            # Fake "decompiler" reproduces the source exactly -> no bytecode divergence.
            rows, stats = build_synthetic_corpus(
                source_dir=src_dir,
                versions=[_HOST_VERSION],
                limit=None,
                holdout_paths=[],
                decompile_fn=_fake_decompile_returning(CORRECT),
            )

            self.assertEqual(len(rows), 0, f"identical decompile must yield 0 rows; stats={stats}")

    def test_holdout_hashes_are_excluded(self):
        with tempfile.TemporaryDirectory() as td:
            src_dir = Path(td) / "src"
            src_dir.mkdir()
            module = src_dir / "classify.py"
            module.write_text(CORRECT, encoding="utf-8")

            holdout_csv = Path(td) / "holdout.csv"
            holdout_csv.write_text(
                "file_hash\n{}\n".format(_module_file_hash(module)), encoding="utf-8"
            )

            rows, stats = build_synthetic_corpus(
                source_dir=src_dir,
                versions=[_HOST_VERSION],
                limit=None,
                holdout_paths=[holdout_csv],
                decompile_fn=_fake_decompile_returning(BROKEN),
            )

            self.assertEqual(len(rows), 0, f"holdout hash must be excluded; stats={stats}")
            self.assertEqual(stats.get("modules_excluded_by_holdout"), 1)

    def test_limit_caps_processed_modules(self):
        with tempfile.TemporaryDirectory() as td:
            src_dir = Path(td) / "src"
            src_dir.mkdir()
            for i in range(3):
                (src_dir / f"m{i}.py").write_text(CORRECT, encoding="utf-8")

            rows, stats = build_synthetic_corpus(
                source_dir=src_dir,
                versions=[_HOST_VERSION],
                limit=1,
                holdout_paths=[],
                decompile_fn=_fake_decompile_returning(BROKEN),
            )

            self.assertEqual(stats.get("modules_processed"), 1)
            self.assertGreaterEqual(len(rows), 1)


if __name__ == "__main__":
    unittest.main()
