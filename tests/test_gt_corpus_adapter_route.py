"""TDD spec for the PRODUCTION adapter route of tools/build_gt_reconstruction_corpus.py.

The adapter dataset (`pyllm_adapter/winnable_run.csv`) has `source == "pylingual"`, which
makes `utils.file_helpers.fetch_pyllmpatch_source_path` take its non-PyPi branch and return
`decompiler_output/indented_*.py` -- the DECOMPILED file, byte-identical to derived_source.
Every row therefore fails the `_gt_differs_from_derived` gate and the corpus comes out EMPTY.

The real GT source lives in a separate tree keyed by the sha512 of the source bytes, which is
the adapter `file_hash` with its trailing `_3XX` version tag stripped (`base_hash`).

This suite pins:
- base-hash stripping selects the right GT directory (incl. a nested `**/` layout),
- a resolved GT that DIFFERS from the decompiled source survives the gt != derived gate,
- an unresolved GT is counted in `rows_gt_unresolved`, is NOT emitted, and is NEVER
  labelled with the decompiled text (that would train the model on its own input),
- `--allow-csv` keeps only listed file_hashes,
- emitted rows still satisfy the finetuner contract pinned by
  tests/test_gt_reconstruction_corpus.py:224.

CPU-only: tiny synthetic sources in tmp dirs. The real 9k-dir dataset is never touched, and
the repair loop is stubbed -- this pins the builder's routing/funnel, not the oracle.
"""
from __future__ import annotations

import csv
import hashlib
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from tools.build_gt_reconstruction_corpus import (
    _filter_allow_rows,
    _gt_differs_from_derived,
    _load_allow_hashes,
    _resolve_gt_source_override,
    build_corpus,
)

GT_SOURCE = (
    "def alpha(x):\n"
    "    return x + 1\n"
    "\n"
    "\n"
    "def gamma(z):\n"
    "    if z > 0:\n"
    "        return z\n"
    "    return -z\n"
)
# What the decompiler emitted: gamma dropped. This is the model's INPUT, never a label.
DERIVED_SOURCE = "def alpha(x):\n    return x + 1\n"

BASE_HASH = hashlib.sha512(GT_SOURCE.encode()).hexdigest()
ROW_HASH = f"{BASE_HASH}_313"


def _write_gt_tree(root: Path, base_hash: str, source: str, *, nested: str = "") -> Path:
    directory = root / nested / base_hash if nested else root / base_hash
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "meta.json").write_text("{}", encoding="utf-8")
    py_path = directory / "program.py"
    py_path.write_text(source, encoding="utf-8")
    return py_path


class ResolveGtSourceOverrideTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)

    def test_base_hash_stripping_picks_the_right_dir(self):
        expected = _write_gt_tree(self.root, BASE_HASH, GT_SOURCE)
        # A decoy sibling that must NOT be selected for the `_313` row.
        other = hashlib.sha512(b"unrelated").hexdigest()
        _write_gt_tree(self.root, other, "def other():\n    pass\n")

        for tag in ("_313", "_310", "_314", "_315", "_39"):
            with self.subTest(tag=tag):
                self.assertEqual(
                    _resolve_gt_source_override(self.root, f"{BASE_HASH}{tag}"), expected
                )
        # An untagged hash resolves to the same dir.
        self.assertEqual(_resolve_gt_source_override(self.root, BASE_HASH), expected)

    def test_a_bare_hex_tail_is_not_mistaken_for_a_version_tag(self):
        """base_hash only strips `_3<digit>[digit]`; a hash ending `_3` keeps its identity."""
        odd = f"{BASE_HASH}_3"
        expected = _write_gt_tree(self.root, odd, GT_SOURCE)
        self.assertEqual(_resolve_gt_source_override(self.root, odd), expected)

    def test_nested_layout_is_found(self):
        expected = _write_gt_tree(self.root, BASE_HASH, GT_SOURCE, nested="shard/a")
        self.assertEqual(_resolve_gt_source_override(self.root, ROW_HASH), expected)

    def test_unresolvable_returns_none_never_a_fallback(self):
        # No dir at all.
        self.assertIsNone(_resolve_gt_source_override(self.root, ROW_HASH))
        # Dir exists but holds no .py -> unresolved, NOT a guess.
        (self.root / BASE_HASH).mkdir(parents=True)
        (self.root / BASE_HASH / "meta.json").write_text("{}", encoding="utf-8")
        self.assertIsNone(_resolve_gt_source_override(self.root, ROW_HASH))


class AllowlistTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.dir = Path(self._tmp.name)

    def _write_csv(self, name: str, hashes: list[str]) -> Path:
        path = self.dir / name
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["file_hash", "error_type", "source"])
            for h in hashes:
                writer.writerow([h, "semantic_error", "pylingual"])
        return path

    def test_allow_csv_keeps_only_listed_hashes(self):
        allow_csv = self._write_csv("allow.csv", ["A1", "A2"])
        allow = _load_allow_hashes([allow_csv])
        self.assertEqual(allow, {"A1", "A2"})

        rows = [
            {"file_hash": "A2", "source": "pylingual"},    # listed -> kept
            {"file_hash": "NOPE", "source": "pylingual"},  # unlisted -> dropped
            {"file_hash": "A1", "source": "pylingual"},    # listed -> kept
        ]
        survivors = list(_filter_allow_rows(rows, allow))
        self.assertEqual([r["file_hash"] for r in survivors], ["A2", "A1"])

    def test_empty_allowlist_is_a_no_op_not_a_total_wipe(self):
        rows = [{"file_hash": "A1"}, {"file_hash": "A2"}]
        self.assertEqual(len(list(_filter_allow_rows(rows, set()))), 2)


class _StubLoop:
    """Stands in for CodeObjectRepairLoop: drives the fixer once for the missing target."""

    captured_gt_sources: list[Path] = []

    def __init__(self, fixer):
        self._fixer = fixer

    def run(self, **kwargs):
        self._fixer.generate_candidates(
            qualname="<module>.gamma",
            gt_code_object=None,
            derived_code_object=None,
            derived_source_fragment="",
            repair_context={"target_is_missing": True, "target_kind": "code_object_fragment"},
        )
        return {}


class _StubOracle:
    """Returns the GT span read from whatever gt_source the builder handed the fixer."""

    def __init__(self, gt_pyc, gt_source):
        _StubLoop.captured_gt_sources.append(Path(gt_source))
        self._text = Path(gt_source).read_text(encoding="utf-8")

    def generate_candidate(self, **kwargs):
        # The GT span for `gamma`; absent from the decompiled text by construction, so a
        # builder that wrongly passed the decompiled file as GT would blow up loudly here.
        head, _, tail = self._text.partition("def gamma")
        if not tail:
            raise AssertionError("oracle was handed a source with no GT gamma span")
        return f"def gamma{tail}".strip()


class BuildCorpusAdapterRouteTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.dir = Path(self._tmp.name)
        self.gt_root = self.dir / "gt_root"
        self.gt_root.mkdir()
        _StubLoop.captured_gt_sources = []

        # The decompiled artifacts the adapter route resolves to (gt_source == derived_source).
        self.decompiled = self.dir / "indented_1.py"
        self.decompiled.write_text(DERIVED_SOURCE, encoding="utf-8")
        self.gt_pyc = self.dir / "gt.pyc"
        self.gt_pyc.write_bytes(b"\x00")
        self.der_pyc = self.dir / "der.pyc"
        self.der_pyc.write_bytes(b"\x00")

        self.dataset = self.dir / "rows.csv"

    def _write_dataset(self, hashes: list[str]) -> None:
        with self.dataset.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["file_hash", "error_type", "source"])
            for h in hashes:
                writer.writerow([h, "semantic_error", "pylingual"])

    def _run(self, **overrides):
        defaults = dict(
            dataset_path=self.dataset,
            source="pylingual",
            holdout_paths=[],
            limit=None,
            row_range=None,
        )
        defaults.update(overrides)
        # The adapter route hands back the DECOMPILED file as gt_source -- the bug under test.
        resolved = (self.decompiled, self.gt_pyc, self.der_pyc, self.decompiled)
        with mock.patch(
            "tools.build_gt_reconstruction_corpus._resolve_row", return_value=resolved
        ), mock.patch(
            "tools.build_gt_reconstruction_corpus.CodeObjectRepairLoop", _StubLoop
        ), mock.patch(
            "tools.build_gt_reconstruction_corpus.OracleFragmentFixer", _StubOracle
        ):
            return build_corpus(**defaults)

    def test_without_override_every_row_is_dropped_as_gt_equals_derived(self):
        """Regression pin: this is exactly the zero-row production bug."""
        self._write_dataset([ROW_HASH])
        sink, _by_op, stats = self._run()
        self.assertEqual(sink, [])
        self.assertEqual(stats["rows_gt_equals_derived"], 1)

    def test_resolved_gt_overrides_decompiled_and_survives_the_gate(self):
        _write_gt_tree(self.gt_root, BASE_HASH, GT_SOURCE)
        self._write_dataset([ROW_HASH])

        sink, _by_op, stats = self._run(gt_source_root=self.gt_root)

        self.assertEqual(stats["rows_gt_equals_derived"], 0)
        self.assertEqual(stats["rows_gt_unresolved"], 0)
        self.assertEqual(stats["rows_processed"], 1)
        self.assertEqual(len(sink), 1)
        # The oracle was handed the REAL GT, not the decompiled file.
        self.assertEqual(
            _StubLoop.captured_gt_sources, [self.gt_root / BASE_HASH / "program.py"]
        )
        # The label carries GT-only content and is not the model's own input.
        self.assertIn("gamma", sink[0]["replacement_text"])
        self.assertNotEqual(sink[0]["replacement_text"].strip(), DERIVED_SOURCE.strip())

    def test_unresolved_gt_is_bucketed_and_never_labelled_with_decompiled_text(self):
        self._write_dataset([ROW_HASH])  # no GT tree written

        sink, _by_op, stats = self._run(gt_source_root=self.gt_root)

        self.assertEqual(stats["rows_gt_unresolved"], 1)
        self.assertEqual(stats["rows_processed"], 0)
        self.assertEqual(sink, [])
        # Critically: the builder must NOT have fallen back to the decompiled file as GT.
        self.assertEqual(_StubLoop.captured_gt_sources, [])

    def test_allow_csv_filters_rows_in_build_corpus(self):
        _write_gt_tree(self.gt_root, BASE_HASH, GT_SOURCE)
        other_base = hashlib.sha512(b"other").hexdigest()
        _write_gt_tree(self.gt_root, other_base, GT_SOURCE)
        self._write_dataset([ROW_HASH, f"{other_base}_312"])

        allow_csv = self.dir / "allow.csv"
        with allow_csv.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["file_hash"])
            writer.writerow([ROW_HASH])

        sink, _by_op, stats = self._run(gt_source_root=self.gt_root, allow_paths=[allow_csv])

        self.assertEqual(stats["rows_considered"], 2)
        self.assertEqual(stats["rows_after_allow"], 1)
        self.assertEqual({r["source_file_hash"] for r in sink}, {ROW_HASH})

    def test_emitted_row_matches_trainer_contract(self):
        _write_gt_tree(self.gt_root, BASE_HASH, GT_SOURCE)
        self._write_dataset([ROW_HASH])

        sink, _by_op, _stats = self._run(gt_source_root=self.gt_root)

        self.assertEqual(len(sink), 1)
        row = sink[0]
        self.assertEqual(
            set(row),
            {
                "prompt",
                "replacement_text",
                "source_file_hash",
                "qualname",
                "target_kind",
                "repair_operation",
            },
        )
        messages = row["prompt"]["messages"]
        self.assertIsInstance(messages, list)
        self.assertGreaterEqual(len(messages), 2)
        for message in messages:
            self.assertIsInstance(message, dict)
            self.assertTrue(str(message["role"]).strip())
            self.assertTrue(str(message["content"]).strip())
        self.assertTrue(row["replacement_text"])


class BlankLabelTest(unittest.TestCase):
    """A blank oracle label must never become a corpus row.

    The oracle returns '' for targets it cannot span (notably `<module>`, whose repair is
    intentionally disabled). The trainer's build_chat_samples silently drops rows with a
    falsy replacement_text, so emitting them yields a corpus whose advertised size overstates
    its trainable size -- on the 40-row smoke sample, 17 of 58 rows were blank `<module>`s.
    """

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.dir = Path(self._tmp.name)
        self.gt_py = self.dir / "gt.py"
        self.gt_py.write_text(GT_SOURCE, encoding="utf-8")

    def _fixer(self, label: str, sink: list, seen: set):
        from tools.build_gt_reconstruction_corpus import OracleCaptureFragmentFixer

        class _BlankOracle:
            def __init__(self, *a, **k):
                pass

            def generate_candidate(self, **kwargs):
                return label

        with mock.patch(
            "tools.build_gt_reconstruction_corpus.OracleFragmentFixer", _BlankOracle
        ):
            return OracleCaptureFragmentFixer(
                gt_pyc=self.dir / "gt.pyc",
                gt_source=self.gt_py,
                source_file_hash="deadbeef",
                sink=sink,
                seen=seen,
            )

    def _drive(self, label: str) -> list:
        sink: list = []
        fixer = self._fixer(label, sink, set())
        fixer.generate_candidates(
            qualname="<module>",
            gt_code_object=None,
            derived_code_object=None,
            derived_source_fragment="",
            repair_context={"target_is_missing": True},
        )
        return sink

    def test_blank_label_is_not_emitted(self):
        for label in ("", "\n", "   \n  "):
            with self.subTest(label=repr(label)):
                self.assertEqual(self._drive(label), [])

    def test_real_label_is_still_emitted(self):
        self.assertEqual(len(self._drive("def gamma(z):\n    return z\n")), 1)


class GtDiffersGateTest(unittest.TestCase):
    def test_gate_distinguishes_real_gt_from_decompiled(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            gt = d / "gt.py"
            gt.write_text(GT_SOURCE, encoding="utf-8")
            der = d / "der.py"
            der.write_text(DERIVED_SOURCE, encoding="utf-8")
            same = d / "same.py"
            same.write_text(DERIVED_SOURCE, encoding="utf-8")

            self.assertTrue(_gt_differs_from_derived(gt, der))
            self.assertFalse(_gt_differs_from_derived(same, der))


if __name__ == "__main__":
    unittest.main()
