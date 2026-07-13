"""Speedup step 4 (pure core): row-sharding + per-worker env wiring for the parallel scale run.

N worker processes each run the UNCHANGED run_scale_repair.py over a DISJOINT ROW_RANGE shard,
pointed at one shared local vllm-serve endpoint. Correctness hazards these pin (from the
architecture map):
  * shards must be disjoint AND complete (a shared work-queue would double-process under
    SEMANTIC_RESUME's check-then-act; only disjoint shards are safe);
  * each worker needs a DISTINCT OUTDIR (results CSV / run_log / scale_result.json open 'w' and
    would clobber otherwise) and per-worker ACCEPTED_CODE_OBJECT_FILE / _TELEMETRY_FILE (the two
    fixed learning-corpus JSONLs interleave/corrupt under concurrent O_APPEND otherwise);
  * workers must NOT build an in-process engine (VLLM off) — they route to the server via
    SEMANTIC_VLLM_SERVER_URL, so N copies of the 30B model never load (that would OOM/reboot).
"""
import unittest

from tools.scale_shard import compute_shards, parse_range, worker_env


class ComputeShardsTest(unittest.TestCase):
    def test_even_partition(self):
        self.assertEqual(
            compute_shards(3000, 8),
            ["0:375", "375:750", "750:1125", "1125:1500",
             "1500:1875", "1875:2250", "2250:2625", "2625:3000"],
        )

    def test_balanced_remainder(self):
        # 10 rows across 3 workers -> 4,3,3 (earlier shards absorb the remainder)
        shards = compute_shards(10, 3)
        sizes = [parse_range(s)[1] - parse_range(s)[0] for s in shards]
        self.assertEqual(sizes, [4, 3, 3])

    def test_covers_and_disjoint_and_ordered(self):
        shards = compute_shards(100, 7)
        covered = []
        for s in shards:
            a, b = parse_range(s)
            covered.extend(range(a, b))
        self.assertEqual(covered, list(range(100)))  # complete + ordered => disjoint

    def test_more_workers_than_rows_drops_empty(self):
        self.assertEqual(compute_shards(3, 8), ["0:1", "1:2", "2:3"])

    def test_zero_rows(self):
        self.assertEqual(compute_shards(0, 8), [])

    def test_single_worker(self):
        self.assertEqual(compute_shards(500, 1), ["0:500"])


class WorkerEnvTest(unittest.TestCase):
    def _env(self, idx=2):
        base = {"MAX_ITER": "5", "PATH": "/x"}
        return worker_env(
            base, worker_idx=idx, outdir=f"/runs/shard{idx}",
            server_url="http://localhost:8001/v1", row_range="750:1125",
            model_name="repair-model",
        )

    def test_shard_routing_env(self):
        e = self._env(2)
        self.assertEqual(e["ROW_RANGE"], "750:1125")
        self.assertEqual(e["OUTDIR"], "/runs/shard2")
        self.assertEqual(e["RESULT"], "/runs/shard2/scale_result.json")
        self.assertEqual(e["SEMANTIC_VLLM_SERVER_URL"], "http://localhost:8001/v1")
        self.assertEqual(e["SEMANTIC_VLLM_SERVER_MODEL"], "repair-model")

    def test_resume_on_and_in_process_engine_off(self):
        e = self._env()
        self.assertEqual(e["SEMANTIC_RESUME"], "1")
        self.assertEqual(e["VLLM"], "0")  # no in-process 30B load in a worker

    def test_per_worker_corpus_files_are_distinct(self):
        e0 = self._env(0)
        e1 = self._env(1)
        self.assertNotEqual(e0["ACCEPTED_CODE_OBJECT_FILE"], e1["ACCEPTED_CODE_OBJECT_FILE"])
        self.assertNotEqual(
            e0["ACCEPTED_CODE_OBJECT_TELEMETRY_FILE"], e1["ACCEPTED_CODE_OBJECT_TELEMETRY_FILE"]
        )
        self.assertIn("0", e0["ACCEPTED_CODE_OBJECT_FILE"])
        self.assertIn("1", e1["ACCEPTED_CODE_OBJECT_FILE"])

    def test_base_env_preserved(self):
        e = self._env()
        self.assertEqual(e["MAX_ITER"], "5")
        self.assertEqual(e["PATH"], "/x")

    def test_does_not_mutate_base(self):
        base = {"A": "1"}
        worker_env(base, worker_idx=0, outdir="/o", server_url="u", row_range="0:1")
        self.assertEqual(base, {"A": "1"})


if __name__ == "__main__":
    unittest.main()
