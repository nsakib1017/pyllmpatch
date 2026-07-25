"""Leak-proof holdout selection for measuring repair models.

Every ceiling estimate this project has produced was measured on hybrid-100, where 72 of 100 hashes
appear in `dataset/flywheel_corpus.jsonl` / `dataset/accepted_hard_ft_refresh.jsonl` -- both of
which store `replacement_text`, i.e. the answer. The resulting "41%" projection missed the clean
production number (~28%) by ~14 points. A holdout is only worth building if it cannot leak.

Two leak paths, both live in this dataset:
  1. Direct: the eval hash was trained on.
  2. Version siblings: `pyllm_adapter` stores one source at several Python versions
     (`001ca04f..._313` AND `001ca04f..._315` both exist on disk). Training on `_313` and
     evaluating on `_315` is training on the eval answer with different bytecode. ~30% of the
     corpus has siblings, so a naive per-file split leaks massively.

Hence: normalize to BASE hash, exclude anything any corpus has seen, and move whole sibling GROUPS
together so a base hash can never straddle the split.
"""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from tools.build_clean_holdout import (
    base_hash,
    load_seen_base_hashes,
    select_holdout,
)


class BaseHashTest(unittest.TestCase):
    def test_strips_version_suffix(self):
        self.assertEqual(base_hash("abc123_313"), "abc123")
        self.assertEqual(base_hash("abc123_315"), "abc123")

    def test_siblings_share_a_base_hash(self):
        # The whole point: these two are the same source and must never straddle the split.
        self.assertEqual(base_hash("deadbeef_311"), base_hash("deadbeef_315"))

    def test_hash_without_suffix_is_unchanged(self):
        self.assertEqual(base_hash("abc123"), "abc123")

    def test_does_not_strip_a_hash_ending_in_digits(self):
        # A sha ending in "_2" or plain hex digits must survive; only a real _3XX tag is a version.
        self.assertEqual(base_hash("abc123def456"), "abc123def456")
        self.assertEqual(base_hash("abc_299"), "abc_299")


class LoadSeenTest(unittest.TestCase):
    def test_collects_base_hashes_from_corpora(self):
        d = Path(tempfile.mkdtemp())
        p = d / "c.jsonl"
        p.write_text(
            json.dumps({"source_file_hash": "aaa_313", "replacement_text": "x"}) + "\n"
            + json.dumps({"source_file_hash": "bbb_315", "replacement_text": "y"}) + "\n"
        )
        seen = load_seen_base_hashes([p])
        self.assertEqual(seen, {"aaa", "bbb"})

    def test_missing_corpus_file_is_not_silently_empty(self):
        # An unreadable corpus must raise. Treating it as "nothing seen" would silently produce a
        # contaminated holdout that LOOKS clean -- the exact failure mode being defended against.
        with self.assertRaises(FileNotFoundError):
            load_seen_base_hashes([Path("/nonexistent/corpus.jsonl")])


def _rows(spec):
    """spec: list of (file_hash, version)."""
    return [{"file_hash": h, "bytecode_version": v} for h, v in spec]


class SelectHoldoutTest(unittest.TestCase):
    def test_excludes_contaminated_base_hashes(self):
        # size counts ELIGIBLE groups: aaa is excluded, so 2 is the most this can yield.
        rows = _rows([("aaa_313", "3.13"), ("bbb_313", "3.13"), ("ccc_313", "3.13")])
        hold, _ = select_holdout(rows, seen={"aaa"}, size=2, seed=1)
        self.assertNotIn("aaa_313", {r["file_hash"] for r in hold})

    def test_excludes_contaminated_via_a_sibling(self):
        # Trained on aaa_313 => aaa_315 is equally contaminated. Version does not launder it.
        rows = _rows([("aaa_315", "3.15"), ("bbb_315", "3.15")])
        hold, _ = select_holdout(rows, seen={"aaa"}, size=1, seed=1)
        self.assertEqual({r["file_hash"] for r in hold}, {"bbb_315"})

    def test_siblings_never_straddle_the_split(self):
        rows = _rows([(f"h{i}_313", "3.13") for i in range(10)]
                     + [(f"h{i}_315", "3.15") for i in range(10)])
        hold, _ = select_holdout(rows, seen=set(), size=10, seed=7)
        held = {base_hash(r["file_hash"]) for r in hold}
        rest = {base_hash(r["file_hash"]) for r in rows} - held
        self.assertEqual(held & rest, set(),
                         "a base hash in the holdout must not also appear in the training remainder")

    def test_is_deterministic_for_a_seed(self):
        rows = _rows([(f"h{i}_313", "3.13") for i in range(50)])
        a, _ = select_holdout(rows, seen=set(), size=10, seed=42)
        b, _ = select_holdout(rows, seen=set(), size=10, seed=42)
        self.assertEqual([r["file_hash"] for r in a], [r["file_hash"] for r in b])

    def test_preserves_the_version_mix(self):
        # Production is 3.11-3.15 with a specific shape; a holdout that drifts off it measures a
        # different population than the one we ship against.
        rows = _rows([(f"a{i}_313", "3.13") for i in range(80)]
                     + [(f"b{i}_315", "3.15") for i in range(20)])
        hold, _ = select_holdout(rows, seen=set(), size=20, seed=3)
        vs = [r["bytecode_version"] for r in hold]
        self.assertAlmostEqual(vs.count("3.13") / len(vs), 0.8, delta=0.15)

    def test_reports_what_it_excluded(self):
        rows = _rows([("aaa_313", "3.13"), ("bbb_313", "3.13")])
        _, stats = select_holdout(rows, seen={"aaa"}, size=1, seed=1)
        self.assertEqual(stats["excluded_contaminated"], 1)
        self.assertEqual(stats["eligible_groups"], 1)

    def test_raises_when_too_few_clean_groups(self):
        # Silently returning a short holdout would understate the eval's noise floor.
        rows = _rows([("aaa_313", "3.13"), ("bbb_313", "3.13")])
        with self.assertRaises(ValueError):
            select_holdout(rows, seen={"aaa", "bbb"}, size=10, seed=1)


if __name__ == "__main__":
    unittest.main()
