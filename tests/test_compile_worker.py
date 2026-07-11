"""A persistent per-version compile worker eliminates the per-candidate ``uvx`` spawn in
the semantic-repair hot loop. It MUST produce byte-identical ``.pyc`` output to the
existing subprocess path (no quality drop) while reusing one long-lived interpreter
across many compiles. Tests drive the worker with the native interpreter (``sys.executable``)
so they stay fast and need no uv network access; production injects the ``uvx --python <ver>``
launcher through the same interface."""
import sys
import tempfile
import unittest
from pathlib import Path

from utils.compile_worker import CompileWorker
from utils.generate_bytecode import CompileError, _compile_native


class CompileWorkerTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def _write(self, name: str, text: str) -> Path:
        p = self.tmp / name
        p.write_text(text, encoding="utf-8")
        return p

    def test_reuses_one_process_across_compiles(self):
        worker = CompileWorker(launch_cmd=[sys.executable], version=(sys.version_info[0], sys.version_info[1]))
        worker.start()
        try:
            src1 = self._write("a.py", "x = 1\n")
            out1 = self.tmp / "a.pyc"
            worker.compile(src1, out1)
            self.assertTrue(out1.exists(), "worker did not produce a .pyc")
            pid_after_first = worker.pid

            src2 = self._write("b.py", "def f():\n    return 2\n")
            out2 = self.tmp / "b.pyc"
            worker.compile(src2, out2)
            self.assertTrue(out2.exists())
            self.assertEqual(worker.pid, pid_after_first, "worker respawned instead of reusing the process")
        finally:
            worker.close()

    def test_bad_source_raises_and_worker_survives(self):
        worker = CompileWorker(launch_cmd=[sys.executable], version=(sys.version_info[0], sys.version_info[1]))
        worker.start()
        try:
            bad = self._write("bad.py", "def (:\n")
            with self.assertRaises(CompileError):
                worker.compile(bad, self.tmp / "bad.pyc")
            # worker must remain usable for the next candidate
            good = self._write("good.py", "y = 42\n")
            out = self.tmp / "good.pyc"
            worker.compile(good, out)
            self.assertTrue(out.exists())
        finally:
            worker.close()

    def test_output_is_byte_identical_to_native_compile(self):
        worker = CompileWorker(launch_cmd=[sys.executable], version=(sys.version_info[0], sys.version_info[1]))
        worker.start()
        try:
            src = self._write("m.py", "import os\n\ndef g(a, b=3):\n    return a + b + len(os.sep)\n")
            worker_out = self.tmp / "worker.pyc"
            native_out = self.tmp / "native.pyc"
            worker.compile(src, worker_out)
            _compile_native(py_file=str(src), out_file=str(native_out))
            self.assertEqual(
                worker_out.read_bytes(),
                native_out.read_bytes(),
                "worker .pyc differs from native py_compile output — quality would drift",
            )
        finally:
            worker.close()


if __name__ == "__main__":
    unittest.main()
