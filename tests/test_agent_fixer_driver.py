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
        mod.init(str(csvp), str(work))
        return mod, work

    def test_pump_surfaces_pending_request_as_prompt(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            mod, work = self._campaign(root)
            handoff = work / "cases" / "aaa" / "handoff"
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
            (work / "cases" / "aaa" / "handoff").mkdir(parents=True)
            with patch.object(mod, "_launch_current", return_value=None), \
                 patch.object(mod, "_pipeline_alive", return_value=True):
                out = mod.pump(str(work))
            self.assertEqual(out["action"], "idle")

    def test_reply_writes_response_and_request_not_resurfaced(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            mod, work = self._campaign(root)
            handoff = work / "cases" / "aaa" / "handoff"
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
            (work / "cases" / "aaa" / "handoff").mkdir(parents=True)
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


if __name__ == "__main__":
    unittest.main()
