import sys; from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils.syntactic_prepass import run_syntactic_prepass, host_compile

def test_fixes_single_error():
    r = run_syntactic_prepass("x = f(1, 2\n")
    assert r.compiled and "balance_delimiters" in r.operations

def test_fixes_multi_error_file():
    src = "a = 1\n    b = 2\n"    # stray indent
    r = run_syntactic_prepass(src)
    assert r.compiled

def test_defers_on_out_of_scope():
    # orphaned except -> no focused operator handles it -> not compiled, no crash, no regression
    src = "def f():\n    return 1\n    except X:\n        pass\n"
    r = run_syntactic_prepass(src)
    assert r.compiled is False and r.source  # returns best-effort source, never crashes

def test_never_regresses():
    src = "valid = 1\n"   # already compiles
    r = run_syntactic_prepass(src)
    assert r.compiled and r.source == src and r.operations == []


def test_compile_gate_rejects_non_advancing_candidate(monkeypatch):
    # An operator that produces a real, distinct candidate which still fails
    # to compile at the SAME line (no advance) must be discarded by the
    # driver's compile gate: the result must never regress to that candidate.
    import utils.syntactic_prepass as sp

    src = "def f():\n    return 1\n    except X:\n        pass\n"

    def _non_advancing_operator(source, error):
        # returns a different string each time it can still "improve" it,
        # but the candidate still fails to compile at the same error line.
        marker = "except  X:"
        if marker in source:
            return None
        return source.replace("except X:", marker)

    _non_advancing_operator.__name__ = "_non_advancing_operator"
    monkeypatch.setattr(sp, "_OPERATORS", [_non_advancing_operator] + sp._OPERATORS)

    r = sp.run_syntactic_prepass(src)

    assert r.compiled is False
    assert r.source == src                                  # discarded, not regressed
    assert "_non_advancing_operator" not in r.operations
