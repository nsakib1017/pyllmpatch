# Corrective residual feedback for LLM repair (#54) — design

Date: 2026-07-12
Status: approved
Task: #54

## Problem

The semantic-repair loop retries an object's LLM repair across iterations. The current
retry feedback (`_format_rejected_attempt_summary`, `pipeline/code_object_repair_loop.py:1316`)
is abstract: last rejection reason, action-type guidance, `replacement_delta`, and the
after-distance *number*. It never tells the model **what its own edit still got wrong** at
the bytecode level, so the model re-guesses instead of correcting.

## Key enabler

At fragment-candidate evaluation (`utils/reattach_source_code_object.py:4444`) the loop
**already computes** `candidate_pylingual_verification = compare_pyc(gt_pyc, candidate_pyc)`
— the full GT-vs-candidate trace (message, `failed_offset`, `bc_a`/`bc_b`) for the acceptance
gate. The corrective signal is therefore already computed and discarded; we only capture and
render it.

## Design (approach A: corrective residual feedback)

Turn "rejected, distance N" into "your edit produced *this specific* remaining divergence
from GT — fix it."

**Data flow:**
1. **Capture (mechanics layer).** New pure helper
   `_candidate_residual_snapshot(verification, qualname) -> dict | None` in
   `reattach_source_code_object.py`. Finds the target object's trace in
   `verification["results"]`; returns `None` if that trace is `success` (target already
   matches GT — the rejection was for another object). Otherwise returns a compact
   `{"message", "failed_offset", "window"}` where `window` is a short GT-vs-candidate
   instruction diff around `failed_offset`, built from the existing `_instruction_records` +
   `_instruction_window` + `_format_instruction_window` helpers.
2. **Store.** `_remember_rejected_attempt` (`2884`) gains a `candidate_residual` param and
   stores it on the attempt dict. The fragment-path caller (`4588`) passes the snapshot taken
   from `best_candidate["pylingual_verification"]`. Capture happens **only when the flag is
   on** (below), so flag-off is byte-identical.
3. **Pass through.** `_compact_rejected_attempts_for_prompt` (`2749`) forwards
   `candidate_residual`.
4. **Render (pipeline layer).** `_format_rejected_attempt_summary` (`1316`) appends, for the
   most recent rejected attempt that has a residual:
   `"- your last edit still differs from ground truth at offset <o> (<message>):\n<window>\n
   Make the smallest change that resolves THIS divergence."`

**Scope:** the fragment (per-object) LLM path only — the dominant repair path. Module and
extra/missing paths are out of scope (YAGNI); the recorder param defaults to `None` there.

## Contracts

- **Prompt-only.** Touches only what the model sees, never the acceptance decision — the
  oracle gate is unchanged by construction. A richer prompt cannot regress correctness.
- **Flag-gated** `SEMANTIC_CORRECTIVE_FEEDBACK` (default off). Flag-off ⇒ no capture, no
  render, byte-identical prompt + telemetry.
- **Layer boundary respected:** mechanics produces the compact residual *data*; the pipeline
  layer formats it into prompt *text*.
- **Never raises:** the snapshot helper defers to `None` on any malformed trace.
- **Python 3.10+**, LOCAL Qwen only — unchanged (this is provider-agnostic prompt text).

## Testing (TDD)

1. `_candidate_residual_snapshot`: returns a compact residual (message + offset + window) for
   a failing target trace; returns `None` for a `success` trace and for a missing qualname;
   never raises on malformed input. Mock `bc_a`/`bc_b` instruction streams (existing
   `_Inst`/`_BC` pattern).
2. `_format_rejected_attempt_summary`: with a rejected attempt carrying `candidate_residual`
   and the flag ON, the summary includes the corrective window; flag OFF ⇒ excluded;
   no residual present ⇒ unchanged from today.
3. Regression: full suite green; flag-off produces the identical prompt string on a fixture
   repair_context (byte-identical guard).

## Measurement

A/B on hybrid-100 (GPU, after the best-shot run): `SEMANTIC_CORRECTIVE_FEEDBACK` on vs the
parity baseline. File-perfect up with no regressions ⇒ ship on; null ⇒ ship off as a measured
negative (every prior *prompt* lever has been null, but this is a feedback loop, not static
context — the one prompt lever worth testing).
