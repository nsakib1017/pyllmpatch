import sys, os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tools.chain_syntactic_to_semantic import build_semantic_case


def test_builds_triple_from_compilable(tmp_path):
    py = tmp_path / "r.py"
    py.write_text("X = 1\n")
    gt = tmp_path / "gt.pyc"
    gt.write_bytes(b"\x00")  # placeholder path; existence only
    case = build_semantic_case(str(py), "abc", "3.12", str(gt), str(tmp_path / "out"))
    assert case["error_type"] == "semantic_error"
    assert case["derived_source"] == str(py)
    assert os.path.exists(case["derived_pyc"]) and case["gt_pyc"] == str(gt)


def test_raises_on_noncompilable(tmp_path):
    py = tmp_path / "bad.py"
    py.write_text("def f(:\n")
    import pytest

    with pytest.raises(ValueError):
        build_semantic_case(str(py), "abc", "3.12", str(tmp_path / "gt.pyc"), str(tmp_path / "out"))
