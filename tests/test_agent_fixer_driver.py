"""TDD for the single-agent fixer driver (campaign_ops/agent_fixer_driver.py).

Pure state machine over a FAKE handoff dir: no LLM, no GPU, no pipeline subprocess.
"""
from __future__ import annotations

import importlib.util
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

DRIVER = Path("/home/mxs220189/pylingual_collaboration/pylingual_download/campaign_ops/agent_fixer_driver.py")


def load_driver():
    spec = importlib.util.spec_from_file_location("agent_fixer_driver", DRIVER)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["agent_fixer_driver"] = mod
    spec.loader.exec_module(mod)
    return mod


def write_csv(path: Path, rows):
    hdr = "file_hash,python_version,gt_pyc,gt_source,derived_pyc,derived_source,prev_final_distance\n"
    body = "".join(
        f"{h},{v},/gt/{h}.pyc,,/d/{h}.pyc,/d/{h}.py,{d}\n" for h, v, d in rows
    )
    path.write_text(hdr + body, encoding="utf-8")


class InitTest(unittest.TestCase):
    def test_init_sorts_by_ascending_residual_and_records_pending(self):
        mod = load_driver()
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            csvp = root / "t.csv"
            write_csv(csvp, [("aaa", "3.12", 500), ("bbb", "3.11", 12), ("ccc", "3.13", 90)])
            work = root / "work"
            mod.init(str(csvp), str(work))
            state = json.loads((work / "campaign.json").read_text())
            self.assertEqual([f["file_hash"] for f in state["files"]], ["bbb", "ccc", "aaa"])
            self.assertTrue(all(f["status"] == "pending" for f in state["files"]))

    def test_init_is_resume_safe_preserving_done_files(self):
        mod = load_driver()
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            csvp = root / "t.csv"
            write_csv(csvp, [("aaa", "3.12", 5), ("bbb", "3.11", 9)])
            work = root / "work"
            mod.init(str(csvp), str(work))
            state = json.loads((work / "campaign.json").read_text())
            state["files"][0]["status"] = "done"
            state["files"][0]["perfect"] = True
            (work / "campaign.json").write_text(json.dumps(state))
            mod.init(str(csvp), str(work))  # re-init must NOT reset progress
            state2 = json.loads((work / "campaign.json").read_text())
            done = [f for f in state2["files"] if f["status"] == "done"]
            self.assertEqual(len(done), 1)
            self.assertTrue(done[0]["perfect"])


class PumpTest(unittest.TestCase):
    def _campaign(self, root: Path, rows=None):
        mod = load_driver()
        csvp = root / "t.csv"
        write_csv(csvp, rows or [("aaa", "3.12", 5)])
        work = root / "work"
        mod.init(str(csvp), str(work), results_root=str(root / "res"))
        return mod, work

    def test_pump_surfaces_pending_request_as_prompt(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            mod, work = self._campaign(root)
            handoff = mod._case_dir(str(work), "aaa") / "handoff"
            handoff.mkdir(parents=True)
            (handoff / "req_7.json").write_text(json.dumps(
                {"id": "7", "system": "SYS", "user": "USR"}))
            with patch.object(mod, "_launch_current", return_value=None), \
                 patch.object(mod, "_pipeline_alive", return_value=True):
                out = mod.pump(str(work))
            self.assertEqual(out["action"], "prompt")
            self.assertEqual(out["id"], "7")
            self.assertEqual(out["system"], "SYS")
            self.assertEqual(out["user"], "USR")
            self.assertEqual(out["file_hash"], "aaa")

    def test_pump_returns_idle_when_alive_with_no_request(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            mod, work = self._campaign(root)
            (mod._case_dir(str(work), "aaa") / "handoff").mkdir(parents=True)
            with patch.object(mod, "_launch_current", return_value=None), \
                 patch.object(mod, "_pipeline_alive", return_value=True):
                out = mod.pump(str(work))
            self.assertEqual(out["action"], "idle")

    def test_reply_writes_response_and_request_not_resurfaced(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            mod, work = self._campaign(root)
            handoff = mod._case_dir(str(work), "aaa") / "handoff"
            handoff.mkdir(parents=True)
            (handoff / "req_7.json").write_text(json.dumps({"id": "7", "system": "S", "user": "U"}))
            frag = root / "fix.txt"
            frag.write_text("def f():\n    return 1\n", encoding="utf-8")
            mod.reply(str(work), "7", str(frag))
            resp = handoff / "resp_7.json"
            self.assertTrue(resp.exists())
            self.assertEqual(json.loads(resp.read_text())["content"], "def f():\n    return 1\n")
            # the pipeline consumes req_ files; simulate that then confirm idle, not a re-prompt
            (handoff / "req_7.json").unlink()
            with patch.object(mod, "_launch_current", return_value=None), \
                 patch.object(mod, "_pipeline_alive", return_value=True):
                out = mod.pump(str(work))
            self.assertEqual(out["action"], "idle")

    def test_pump_advances_to_next_file_when_pipeline_done(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            mod, work = self._campaign(root, rows=[("aaa", "3.12", 5), ("bbb", "3.11", 9)])
            (mod._case_dir(str(work), "aaa") / "handoff").mkdir(parents=True)
            with patch.object(mod, "_launch_current", return_value=None), \
                 patch.object(mod, "_pipeline_alive", return_value=False), \
                 patch.object(mod, "_harvest_current", return_value={"added": 3, "perfect": True}):
                out = mod.pump(str(work))
            self.assertEqual(out["action"], "advance")
            self.assertEqual(out["file_hash"], "aaa")
            state = json.loads((work / "campaign.json").read_text())
            by = {f["file_hash"]: f for f in state["files"]}
            self.assertEqual(by["aaa"]["status"], "done")
            self.assertTrue(by["aaa"]["perfect"])

    def test_pump_reports_alldone_with_stats(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            mod, work = self._campaign(root)
            state = json.loads((work / "campaign.json").read_text())
            state["files"][0].update(status="done", perfect=True)
            (work / "campaign.json").write_text(json.dumps(state))
            out = mod.pump(str(work))
            self.assertEqual(out["action"], "alldone")
            self.assertEqual(out["stats"]["total"], 1)
            self.assertEqual(out["stats"]["done"], 1)
            self.assertEqual(out["stats"]["perfect"], 1)


class LaunchMaxIterationsTest(unittest.TestCase):
    def test_claude_serve_launch_passes_max_iterations_when_given(self):
        sys.path.insert(0, "/home/mxs220189/pylingual_collaboration/pylingual_download/code")
        import importlib
        cs = importlib.import_module("tools.agent_fixer.claude_serve")
        with tempfile.TemporaryDirectory() as td:
            captured = {}

            class FakeProc:
                pid = 4242

            def fake_popen(cmd, **kw):
                captured["cmd"] = cmd
                return FakeProc()

            with patch.object(cs.subprocess, "Popen", side_effect=fake_popen), \
                 patch("utils.generate_bytecode.compile_version", return_value=None):
                cs.launch("/gt.pyc", "/s.py", "3.12", td, max_iterations=2)
            self.assertIn("--max-iterations", captured["cmd"])
            self.assertEqual(captured["cmd"][captured["cmd"].index("--max-iterations") + 1], "2")


class PersistenceTest(unittest.TestCase):
    def test_case_dir_resolves_under_results_root(self):
        mod = load_driver()
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            csvp = root / "t.csv"
            write_csv(csvp, [("aaa", "3.12", 5)])
            work = root / "work"
            res = root / "results" / "malware_agent_fixer"
            mod.init(str(csvp), str(work), results_root=str(res))
            case = mod._case_dir(str(work), "aaa")
            self.assertTrue(str(case).startswith(str(res)))
            self.assertIn("aaa", str(case))

    def test_launch_does_not_skip_module_body_repair(self):
        mod = load_driver()
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            csvp = root / "t.csv"
            write_csv(csvp, [("aaa", "3.12", 5)])
            work = root / "work"
            mod.init(str(csvp), str(work), results_root=str(root / "res"))
            entry = json.loads((work / "campaign.json").read_text())["files"][0]
            os.environ.pop("SEMANTIC_SKIP_MODULE_BODY_REPAIR", None)
            captured = {}

            def fake_launch(gt, src, ver, wd, max_iterations=1):
                captured["skip"] = os.environ.get("SEMANTIC_SKIP_MODULE_BODY_REPAIR")
                captured["det"] = os.environ.get("SEMANTIC_DETERMINISTIC_OPERATORS")

            import claude_serve
            with patch.object(claude_serve, "launch", side_effect=fake_launch):
                mod._launch_current(str(work), entry)
            # module repairs must be attempted -> the skip flag is never turned on
            self.assertNotIn(captured.get("skip"), ("1", "true", "yes", "on"))
            self.assertEqual(captured.get("det"), "true")

    def test_harvest_backfills_missing_final_pyc_and_writes_valid_index(self):
        mod = load_driver()
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            csvp = root / "t.csv"
            write_csv(csvp, [("aaa", "3.12", 5)])
            work = root / "work"
            res = root / "res"
            mod.init(str(csvp), str(work), results_root=str(res))
            entry = json.loads((work / "campaign.json").read_text())["files"][0]
            case = mod._case_dir(str(work), "aaa")
            case.mkdir(parents=True)
            final_src = case / "prompts" / "step1_cand0.py"
            final_src.parent.mkdir(parents=True)
            final_src.write_text("x = 1\n")
            gone_pyc = case / "prompts" / "__pycache__" / "missing.pyc"  # recorded but absent
            (case / "result.json").write_text(json.dumps({
                "final_source": str(final_src),
                "final_pyc": str(gone_pyc),
                "gt_pyc": "/gt/aaa.pyc",
                "target_python_version": "3.12",
                "final_summary": {"combined_distance": 0, "all_equal": True},
                "pylingual_verification": {"all_equal": True},
                "steps": [{"accepted": True}],
            }))

            def fake_compile(src, out, ver):
                Path(out).write_bytes(b"PYC")

            with patch("utils.generate_bytecode.compile_version", side_effect=fake_compile), \
                 patch.object(__import__("claude_serve"), "harvest", return_value=None):
                out = mod._harvest_current(str(work), entry)

            self.assertTrue(out["perfect"])
            idx = res / "run_index.jsonl"
            self.assertTrue(idx.exists())
            row = json.loads(idx.read_text().splitlines()[-1])
            self.assertEqual(row["file_hash"], "aaa")
            self.assertTrue(row["all_equal"])
            self.assertTrue(Path(row["final_source"]).exists())
            self.assertTrue(Path(row["final_pyc"]).exists())   # backfilled


if __name__ == "__main__":
    unittest.main()
