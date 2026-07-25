# Mechanical Syntactic-Repair — SDD progress ledger
Plan: docs/superpowers/plans/2026-07-22-mechanical-syntactic-repair.md
Rule: NEVER commit (human commits). Task-end = full suite green (baseline 796).
User reqs: focused/high-precision operators; MINIMIZE repair window (Task 5b) so LLM sees smallest snippet + clean reattach.

- [x] Tasks 1-5: utils/syntactic_prepass.py (SyntaxErrorInfo/probe/advanced + balance_delimiters, dedent_stray_block, fix_line_continuation, fix_numeric_literal + driver) + tests
- [x] Task 5b: minimal_repair_window + reattach_window (smallest snippet + exact reattach) + tests
- [x] Task 6: pipeline/runner.py maybe_prepass hook + LLM residual uses minimal window + test
- [x] Task 7: tools/chain_syntactic_to_semantic.py handoff + test
- [x] Task 8: tools/syntactic_prepass_bench.py measurement harness

## Task 1-5b review (2026-07-22)
Suite 819 green, 23 new tests. Review found: IMPORTANT (1) minimal_repair_window indent-heuristic breaks Black-style dedented closers -> fixing with bracket-depth tracking; (2) compile-gate reject path untested -> adding test. MINOR (3) triple-quote in balance_delimiters; (4) tautological test operators.py:52; (5) empty-source guard untested; (6) type nit list[str]. Fix subagent dispatched for all.
Task 1-5b: COMPLETE (review clean after fix; 36 tests, suite 832 green; minimal-window bracket-depth fix verified on Black-style + never-closed cases).

## Task 6 review (2026-07-22)
maybe_prepass hook + minimal-window LLM localization; 7 tests, suite 839. Review CONFIRMED an IMPORTANT production defect: compile_version always raises CompileError (not SyntaxError) even host-version -> probe_syntax loses lineno -> 2/4 operators + multi-round driver DEAD in production. Fix: probe_syntax parses "line N" from message text (~3 lines) + production-shaped tests + flag-off regression test. Fix subagent dispatched. LLM-window path unaffected (repair_engine has own extract_line_number). Minor: reattach-detection hack, _line_roles duplicate, perf (Task 8).
Task 6: COMPLETE (probe_syntax lineno fix verified in production shape; 844 suite green)
Task 7: COMPLETE (build_semantic_case verified: triple built, raises on non-compilable; suite 846)
Task 8: COMPLETE (bench harness; 20-row smoke: 5/20 mislabels, 1/15 genuine mechanically fixed, per-op dedent_stray_block+balance_delimiters fired).
INFRA BUG FOUND: compile_version (utils/generate_bytecode.py:201-204) falls to _compile_pyenv on uv CompileError; pyenv not installed -> pyenv error REPLACES real "line N" for non-host versions -> starves lineno operators + LLM localization. Fix (preserve uv error when pyenv can't help) dispatched TDD. Full mechanical bench (1797 files) running.
INFRA FIX DONE: compile_version now preserves the real uv "line N" diagnostic when pyenv can't help (3 tests, suite 849 green, no regressions). Whole syntactic pipeline (mechanical + LLM localization) now gets correct line numbers on non-host versions.

## Cause-aware window feature (2026-07-22)
Plan: docs/superpowers/plans/2026-07-22-cause-aware-window.md
Goal: LLM window captures the CAUSE (often 5-10 lines above the reported symptom line) + minimal context.
- [ ] Task 1: locate_cause (symptom->cause anchoring rules)
- [ ] Task 2: cause_aware_window (build from cause + preceding header/decorators/statement, capped, round-trip preserved)
- [ ] Task 3: wire cause_aware_window into runner minimal_window_syntax_context + recompute site

## Annotation Reconciliation Stage 1 (2026-07-25)
Plan: docs/superpowers/plans/2026-07-25-annotation-reconciliation-stage1.md
Branch: feat/annotation-reconciliation | Base: 71a7adc
Constraints: oracle never weakened (reuse existing prepass gate); flag SEMANTIC_ANNOTATION_RECONCILE
default off -> byte-identical; defer (None) on ambiguity; Python 3.10+; semantic-core edits authorized.
- [x] Task 1: flag + `_annotate_enclosing_qualname` (`X.__annotate__` -> `X`)
- [ ] Task 2: `_gt_def_source_by_qualname` (extract GT def source by qualname)
- [ ] Task 3: annotation branch in run_deterministic_prepass (oracle backend) + engine fixture test (150->0)
- [ ] Task 4: broaden fixtures + full regression (suite green, flag-off byte-identical)
