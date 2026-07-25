import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pipeline.runner import codeobject_window_syntax_context

NESTED = "class A:\n    def m(self):\n        x = f(1, 2\n        y = 3\n"


def test_provider_isolates_enclosing_object(tmp_path):
    p = tmp_path / "s.py"
    p.write_text(NESTED)
    seg = codeobject_window_syntax_context(p, error_line=3, error_description="'(' was never closed", expansion_level=0)
    assert seg is not None and "def m(self):" in seg.text and seg.start_line == 2


def test_provider_falls_back_when_error_line_missing(tmp_path):
    p = tmp_path / "s.py"
    p.write_text(NESTED)
    assert codeobject_window_syntax_context(p, error_line=None, error_description="x", expansion_level=0) is None


def test_module_scope_uses_minimal_window(tmp_path):
    p = tmp_path / "s.py"
    p.write_text("x = f(1, 2\n")
    seg = codeobject_window_syntax_context(p, error_line=1, error_description="'(' was never closed", expansion_level=0)
    assert seg is not None and seg.text.strip() == "x = f(1, 2"


# The measured winner (object-window 75% vs baseline 52% LLM-solo, +23pts,
# nearly doubling 3.15) is now the DEFAULT: unset env -> ON; explicit "0" -> OFF.
def test_codeobject_window_default_on_when_env_unset(monkeypatch):
    from pipeline.runner import syntactic_codeobject_window_enabled
    monkeypatch.delenv("SYNTACTIC_CODEOBJECT_WINDOW", raising=False)
    assert syntactic_codeobject_window_enabled() is True


def test_codeobject_window_explicit_off(monkeypatch):
    from pipeline.runner import syntactic_codeobject_window_enabled
    monkeypatch.setenv("SYNTACTIC_CODEOBJECT_WINDOW", "0")
    assert syntactic_codeobject_window_enabled() is False


def test_codeobject_window_explicit_on(monkeypatch):
    from pipeline.runner import syntactic_codeobject_window_enabled
    monkeypatch.setenv("SYNTACTIC_CODEOBJECT_WINDOW", "1")
    assert syntactic_codeobject_window_enabled() is True
