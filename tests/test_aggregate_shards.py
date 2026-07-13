"""Speedup step 4: cross-shard aggregation. The parallel run writes one result.json tree per
worker OUTDIR; the final headline (file-perfect N, per-version) must be computed over ALL shards,
deduped by file_hash. Disjoint shards mean no real collisions, but a resumed/rerun row can appear
twice — dedup must prefer the perfect result so a stale non-perfect artifact can't mask a win."""
import unittest

from tools.aggregate_shards import file_hash_of, is_perfect, merge_results, summarize


def R(fh, perfect, ver="3.11"):
    return {
        "gt_pyc": f"/data/{fh}/gt.pyc",
        "pylingual_verification": {"all_equal": perfect},
        "target_python_version": ver,
    }


class HelpersTest(unittest.TestCase):
    def test_file_hash_from_gt_pyc(self):
        self.assertEqual(file_hash_of(R("abc123", True)), "abc123")

    def test_is_perfect_true_false_missing(self):
        self.assertTrue(is_perfect(R("h", True)))
        self.assertFalse(is_perfect(R("h", False)))
        self.assertFalse(is_perfect({}))  # missing verification -> not perfect


class MergeTest(unittest.TestCase):
    def test_dedup_prefers_perfect(self):
        merged = merge_results([("h1", R("h1", False)), ("h1", R("h1", True))])
        self.assertEqual(len(merged), 1)
        self.assertTrue(is_perfect(merged["h1"]))

    def test_perfect_not_overwritten_by_later_nonperfect(self):
        merged = merge_results([("h1", R("h1", True)), ("h1", R("h1", False))])
        self.assertTrue(is_perfect(merged["h1"]))

    def test_distinct_hashes_all_kept(self):
        merged = merge_results([("h1", R("h1", True)), ("h2", R("h2", False))])
        self.assertEqual(set(merged), {"h1", "h2"})


class SummarizeTest(unittest.TestCase):
    def test_headline_and_per_version(self):
        results = {
            "h1": R("h1", True, "3.11"),
            "h2": R("h2", False, "3.11"),
            "h3": R("h3", True, "3.13"),
        }
        s = summarize(results)
        self.assertEqual(s["total"], 3)
        self.assertEqual(s["file_perfect"], 2)
        self.assertEqual(s["per_version"]["3.11"], {"total": 2, "perfect": 1})
        self.assertEqual(s["per_version"]["3.13"], {"total": 1, "perfect": 1})

    def test_empty(self):
        s = summarize({})
        self.assertEqual(s["total"], 0)
        self.assertEqual(s["file_perfect"], 0)


if __name__ == "__main__":
    unittest.main()
