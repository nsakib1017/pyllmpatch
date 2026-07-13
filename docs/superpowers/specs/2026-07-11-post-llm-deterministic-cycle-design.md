# Post-LLM Deterministic Reconciliation Cycle — Design

**Date:** 2026-07-11
**Status:** Approved (design), pending implementation
**Flag:** `SEMANTIC_POST_LLM_DETERMINISTIC` (default off)

## Goal

Implement the repair cycle: **pre-LLM deterministic → LLM call → if not perfect → post-LLM deterministic → test perfect → repeat.** After each LLM iteration, re-run the oracle-gated deterministic operators on the residual, because a *structural* LLM edit can flip a "Different control flow" object into a "Different bytecode" object and thereby **expose a leaf divergence the pre-LLM prepass could not reach**. Measure whether this converts additional files to byte-perfect.

## Architecture

Reuse the existing `run_deterministic_prepass()` closure in `utils/reattach_source_code_object.py` (it already applies every deterministic operator to an internal fixpoint, accepting each edit only on a strict oracle-state-lattice improvement with no regression). Re-invoke it once per LLM iteration via a single flag-gated hook. There is **exactly one** copy of the oracle gate in the engine — the post-pass inherits it verbatim; no second gate, no distance-only gate.

Flag defaults off → the default path and the 680-test suite are byte-identical.

## Design decisions (finalized)

| Decision | Choice | Why |
|---|---|---|
| Granularity | **Per-iteration**, one hook at `:4977` | The internal fixpoint batches all leaves exposed by that iteration's accepts; per-candidate would need 4 hook sites and re-run the expensive verify per accept for no added reach |
| Fixpoint vs single | **Fixpoint** (reuse the closure's own loop) | One leaf fix can cascade to expose another |
| Abstraction | **Minimal hook**, no RepairStep/driver | The closure is already re-invocable over the mutable `current_*` state; a driver is pure ceremony and risks the suite |
| Termination | **Independent backstops** (per-target `max_iterations`; `max_prepass_edits=200`; timeout) | Not a fragile potential-function proof (Φ-monotonicity is false in distance mode) |
| Oracle gate | Inherited verbatim; **hard-require `ACCEPTANCE_MODE=='oracle'`** | Both det & LLM become lattice-non-regressing → cannot undo each other → no thrash |
| **Repeat semantics** | **No attempt-cap reset** | Post-det mops up each iteration; the LLM does NOT re-swing an already-tried qualname. Simpler/cheaper first increment. (Reset is a documented follow-up requiring `max_iterations>1`.) |
| **Null outcome** | **Ship OFF as a measured negative** | Build, A/B, document; if null, stays flag-off and effort redirects to operator-coverage. Reverts via one flag. |

## Integration (concrete edits, all in `utils/reattach_source_code_object.py` unless noted)

1. `:3808` — **keep the name** `run_deterministic_prepass`; change signature to `def run_deterministic_prepass(*, phase="pre_llm", iteration_label=0) -> int:` (a test greps for this exact name; renaming breaks `test_deterministic_dispatch_coverage.py`).
2. `:3822-3825` — phase-aware enable + `return 0`: `enabled = SEMANTIC_DETERMINISTIC_PREPASS if phase=='pre_llm' else (SEMANTIC_POST_LLM_DETERMINISTIC and ACCEPTANCE_MODE=='oracle')`.
3. `:3826` — post-phase baseline refresh: `if phase=='post_llm': current_pylingual_verification = run_pylingual_verification(gt_pyc, current_pyc)` (closes stale-baseline hole).
4. `:3836` — `base_stem = derived_source.stem` (stable stem; no-op pre-LLM; prevents filename compounding).
5. `:3840-3841` — hard-timeout early exit returns `applied` (not `None`) → prevents `+= None` TypeError.
6. `:3944-3952` — move `next_source.write_text` inside the `try`; add `OSError` to the except (ENAMETOOLONG defense).
7. `:3967` — store `'iteration': iteration_label` and add `'phase': phase` to each step dict.
8. `:3994-3995` — `return applied` after the fixpoint loop.
9. `:3710` `record_best_state_if_improved` — oracle-aware promotion, **flag+mode gated** (default path untouched): also promote on `delta.improved and not delta.regressed` so oracle-only (distance-flat) post-det gains survive timeout rollback.
10. `:3704` — run-scoped `post_llm_det_edits_total = 0`.
11. `:4001` — leave the pre-LLM call as bare `run_deterministic_prepass()` (defaults; ignore return; preserves the test anchor).
12. `:4977` — **THE HOOK** (before the iteration-end timeout check): fire when `SEMANTIC_POST_LLM_DETERMINISTIC and ACCEPTANCE_MODE=='oracle' and accepted_this_iteration>0 and not current_pylingual_verification.get('all_equal')`; wrap in `try/except` (on error keep LLM state); `post_llm_det_edits_total += applied`; `accepted_this_iteration += applied`; `if all_equal: break`.
13. `:4995` result dict — add `post_llm_deterministic_enabled` and `post_llm_deterministic_edits`.
14. Benchmark harness — run with `SEMANTIC_ACCEPTANCE_MODE=oracle` (feature no-ops in distance mode).

## Testing (no live LLM; `tests/test_post_llm_deterministic.py`)

1. `test_flag_off_is_pure_noop` — flag off → edits==0, no `phase=='post_llm'` step, result matches baseline. Guards the suite.
2. `test_requires_oracle_mode` — flag on + distance mode → `enabled` False, edits==0.
3. `test_structural_edit_exposes_leaf` — **the hypothesis A/B**: stub fixer makes only a structural edit that reshapes a control-flow-masked object into "Different bytecode" with one wrong literal (verify the pre-pass provably defers on the original derived pyc); flag ON → `all_equal` + a `phase=='post_llm'` leaf accept; flag OFF → not `all_equal`. If no such fixture is constructible, that IS the finding — record it, don't fake it.
4. `test_stale_baseline_closed` — per-step verify off; post candidate improves-vs-initial but regresses-vs-current → refresh causes rejection.
5. `test_hook_exception_preserves_progress` — post-phase raise → function returns, state == LLM's last accepted.
6. `test_non_regression_gate_preserved` — operator returns an oracle-regressing candidate → no post accept.
7. `test_best_state_survives_timeout` — oracle-up/distance-flat post accept + forced timeout → gain retained.
8. `test_terminates_no_oscillation` — exposed-leaf case returns under a deadline; `max_iterations` respected.
9. `test_filename_no_compound` — two consecutive post accepts → no `prepass\d+_prepass\d+_` nesting.
10. `test_return_type_on_hard_timeout` — returns int on timeout path.
11. Regression: `test_deterministic_dispatch_coverage.py`, full `pytest`, `tools/validate_semantic_repair_versions.py`.

## Telemetry / A-B

Per-step `phase` + real `iteration` stamps; run-level `post_llm_deterministic_{enabled,edits}`. **A/B method:** run the benchmark twice with `SEMANTIC_ACCEPTANCE_MODE=oracle` fixed, `SEMANTIC_POST_LLM_DETERMINISTIC` 0 vs 1. Report (i) marginal file-perfects (files flipping to `all_equal` only flag-on), (ii) of those, the subset whose *last accepted step* has `phase=='post_llm'` (post-det tipped it), (iii) marginal final combined-distance (final-vs-final, not summed per-step — avoids rollback over-count).

## Risks (accepted)

- **Effectiveness may be null** (consistent with the prior 0/4 simulation). Built to measure + revert cleanly; null is informative, not a defect.
- With `max_iterations=1`, the LLM won't re-swing an already-tried qualname, so the expose→re-swing benefit needs `max_iterations>1` or a det-fixable residual. Documented.
- `compare_verifications` object-disappearance blind spot is inherited (not new); no superset assertion added (would break the legitimate `control_flow_renest` operator).
- One extra baseline-refresh verify + fixpoint per progressing iteration (bounded by the `accepted_this_iteration>0`/`all_equal` guards).
