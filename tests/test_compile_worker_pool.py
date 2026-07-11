"""The persistent compile worker is wired into compile_version behind SEMANTIC_COMPILE_WORKER,
with a per-version pool and a bulletproof fallback to the legacy uv/pyenv path. These tests
pin the pool lifecycle (cache, respawn-on-death, negative-cache) and the routing decisions
(flag off -> legacy; worker ok -> no legacy; worker unavailable -> legacy) without spawning
uvx, by injecting the launch command and spying on the compile helpers."""
import os
import sys
import unittest
from unittest import mock

import utils.generate_bytecode as gb
from utils.generate_bytecode import WorkerUnavailable

HOST = (sys.version_info[0], sys.version_info[1])


class CompileWorkerPoolTest(unittest.TestCase):
    def setUp(self):
        gb._WORKER_POOL.clear()

    def tearDown(self):
        for v in list(gb._WORKER_POOL):
            gb._drop_worker(v)
        gb._WORKER_POOL.clear()

    def test_pool_caches_and_respawns_on_death(self):
        with mock.patch.object(gb, "_worker_launch_cmd", return_value=[sys.executable]):
            w1 = gb._get_compile_worker(HOST)
            w2 = gb._get_compile_worker(HOST)
            self.assertIs(w1, w2, "pool did not cache the worker")

            w1.close()
            self.assertFalse(w1.alive)
            w3 = gb._get_compile_worker(HOST)
            self.assertIsNot(w3, w1, "pool did not respawn a dead worker")
            self.assertTrue(w3.alive)

    def test_failed_start_is_negative_cached(self):
        launch = mock.Mock(return_value=["/nonexistent/python-xyz-does-not-exist"])
        with mock.patch.object(gb, "_worker_launch_cmd", launch):
            with self.assertRaises(WorkerUnavailable):
                gb._get_compile_worker((3, 11))
            with self.assertRaises(WorkerUnavailable):
                gb._get_compile_worker((3, 11))
        self.assertEqual(launch.call_count, 1, "negative cache did not prevent a respawn attempt")


class CompileVersionRoutingTest(unittest.TestCase):
    def setUp(self):
        gb._WORKER_POOL.clear()

    def _cross_version(self):
        # any UV version that is not the host, so the worker/uv branch is taken
        for cand in (3, 10), (3, 11), (3, 13):
            if cand != HOST:
                return cand
        return (3, 10)

    def test_flag_off_uses_legacy_uv(self):
        v = self._cross_version()
        with mock.patch.dict(os.environ, {"SEMANTIC_COMPILE_WORKER": "false"}), \
             mock.patch.object(gb, "_compile_worker") as worker, \
             mock.patch.object(gb, "_compile_uv") as uv, \
             mock.patch.object(gb, "_compile_pyenv") as pyenv:
            gb.compile_version("s.py", "o.pyc", v)
        worker.assert_not_called()
        uv.assert_called_once()
        pyenv.assert_not_called()

    def test_flag_on_worker_success_skips_legacy(self):
        v = self._cross_version()
        with mock.patch.dict(os.environ, {"SEMANTIC_COMPILE_WORKER": "1"}), \
             mock.patch.object(gb, "_compile_worker") as worker, \
             mock.patch.object(gb, "_compile_uv") as uv, \
             mock.patch.object(gb, "_compile_pyenv") as pyenv:
            gb.compile_version("s.py", "o.pyc", v)
        worker.assert_called_once()
        uv.assert_not_called()
        pyenv.assert_not_called()

    def test_flag_on_worker_unavailable_falls_back_to_legacy(self):
        v = self._cross_version()
        with mock.patch.dict(os.environ, {"SEMANTIC_COMPILE_WORKER": "1"}), \
             mock.patch.object(gb, "_compile_worker", side_effect=WorkerUnavailable("boom")) as worker, \
             mock.patch.object(gb, "_compile_uv") as uv, \
             mock.patch.object(gb, "_compile_pyenv") as pyenv:
            gb.compile_version("s.py", "o.pyc", v)
        worker.assert_called_once()
        uv.assert_called_once()


if __name__ == "__main__":
    unittest.main()
