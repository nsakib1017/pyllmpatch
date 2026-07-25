import sys; from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils.syntactic_prepass import SyntaxErrorInfo, probe_syntax, host_compile, advanced

def test_probe_returns_none_for_valid(): assert probe_syntax("x = 1\n", host_compile) is None
def test_probe_returns_error_for_invalid():
    e = probe_syntax("def f(:\n    pass\n", host_compile)
    assert e is not None and e.lineno == 1 and "syntax" in e.msg.lower()
def test_advanced_true_when_fixed():
    assert advanced(SyntaxErrorInfo(1,0,"x"), None) is True
def test_advanced_true_when_line_increases():
    assert advanced(SyntaxErrorInfo(1,0,"x"), SyntaxErrorInfo(5,0,"y")) is True
def test_advanced_false_when_same_or_earlier():
    assert advanced(SyntaxErrorInfo(5,0,"x"), SyntaxErrorInfo(5,0,"y")) is False
    assert advanced(SyntaxErrorInfo(5,0,"x"), SyntaxErrorInfo(2,0,"y")) is False
