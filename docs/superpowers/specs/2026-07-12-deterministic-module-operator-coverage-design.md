# Deterministic module/operator coverage (Stage 1) — design

Date: 2026-07-12
Status: approved (pending spec review)
Task: #35 (reframed)

## Problem & evidence

Goal: raise the file-perfect (PyLingual `all_equal`) rate on 3.10+ semantic-repair
cases. A census of **1,095 unique files (685 failing)** partitioned by the repair
loop's structural surrender points found:

| Surrender class | Failing files | Addressable |
|---|---|---|
| Missing-object reconstruction | 22 | **0** — never the sole/dominant blocker (unmatched penalty is a fixed ~4; matched objects always diverge more). CONFIRMED NULL. |
| Module-body (`<module>` sole blocker) | 13 (10 near-misses, combined 1–32) | the live lever |
| Compute/timeout | 23 | measured separately (best-shot run) |
| Extra-object | 1 | 0 |
| Pure matched-object divergence | 625 (91%) | LLM/decompiler plateau |

**Root cause (established by reproduction, not assumption):** `<module>` is NOT
excluded from deterministic repair. `run_deterministic_prepass`
(`utils/reattach_source_code_object.py:3890`) already feeds `<module>` the whole
source through the 20-operator diffbc/struct chain. The near-misses survive because
**three divergence patterns have no operator** — and these patterns are uncovered
for functions too, not just `<module>`.

Reproduction (prepass in isolation, `fragment_fixer=None`, oracle mode, CPU) on the
4 locally-resolvable near-misses:

```
88bc3ebd7c (f-string):    combined 3 → 0,  all_equal=TRUE  ✅ via deterministic_prepass_fstring
5235c14286 (annotation):  12 → 12, all_equal=False  (no operator fires)
8eedb5648e (default-args): no operator fires
6bdb87bc50 (loop-shape):  16 → 16, all_equal=False  (no operator fires)
```

Two consequences:
1. **`fstring` already converts `88bc3ebd7c` today**, but the standing 37/99 parity
   baseline predates the module-prepass handling, so banked wins are uncounted. The
   true current deterministic baseline must be re-measured.
2. `annotation`, `default-args`, and `loop-shape` are genuine operator-coverage gaps.

## Scope

**In scope (Stage 1, CPU-only, no GPU):**
- Measure the true current deterministic-only hybrid-100 baseline.
- Add two deterministic operators: **default-args-restore** and **annotation-strip**.
  Both reconstruct verbatim from the oracle (GT bytecode) — no guessing.

**Out of scope (Stage 2, later, GPU):**
- The `GET_ITER`/`FOR_ITER` loop-shape case (`6bdb87bc50`) — structural; defer to LLM.
- Any LLM-path change.

## Hard contracts (unchanged)

- **Oracle never weakened.** Every candidate passes recompile + oracle-gated
  non-regression (`compare_verifications`, `delta.regressed or not delta.improved`).
- **Operators defer on ambiguity** — return `None` rather than guess. Values come
  only from the GT const tuple / GT code object, never inference.
- **Same interface** as the existing 20 operators:
  `candidate(bc_a, bc_b, fragment, repair_context) -> {"text": str, ...} | None`,
  independently unit-testable.
- **Python 3.10+ only.** Detection must cover **3.11–3.15** (see version matrix).
- Behavior stays behind the existing `SEMANTIC_DETERMINISTIC_OPERATORS` flag.
- Never mutate original decompiled inputs; all edits on copies under `results/`.

## Version matrix (verified via `uv run --python N`)

Detection is version-specific; the **source edit is version-independent** (the
recompile handles version bytecode specifics).

**Default args:**
| Version | GT (has defaults) | Derived (dropped) |
|---|---|---|
| 3.11, 3.12 | `LOAD_CONST (defaults tuple)` + `MAKE_FUNCTION 1 (defaults)` | `MAKE_FUNCTION 0` |
| 3.13, 3.14, 3.15 | `LOAD_CONST (defaults tuple)` + `MAKE_FUNCTION` + `SET_FUNCTION_ATTRIBUTE 1 (defaults)` | `MAKE_FUNCTION` (no `SET_FUNCTION_ATTRIBUTE`) |

**Annotated assignment:**
| Version | Bytecode signature |
|---|---|
| 3.11, 3.12, 3.13 | `SETUP_ANNOTATIONS` … `STORE_NAME n` … `LOAD_NAME __annotations__` / `STORE_SUBSCR` |
| 3.14, 3.15 | PEP 649: separate `__annotate__` code object + `STORE_NAME __annotate__` / `__conditional_annotations__`; NO `SETUP_ANNOTATIONS`/`STORE_SUBSCR` |

## Components

### 1. Measurement (revised after implementation)
A deterministic-*only* CPU measurement was attempted but abandoned: `fragment_fixer=None`
falls back to oracle-**source** fixing (needs the GT `.py`, unavailable — repair-only), and
a null fixer trips the module *fixer* path's unguarded `None`-candidate crash. More
fundamentally, the operator's value on real files (e.g. `8eedb5648e`: *functions* diverge
304 and need the LLM, *module* diverges 16) only shows in the **full det+LLM pipeline** —
deterministic-only can't convert it. So the A/B is a full det+LLM hybrid-100 run (GPU),
operator-ON vs the parity baseline, once the card frees. Per-operator correctness is proven
by unit + version-shape + e2e tests and an isolated recompile-to-fixpoint check on the real
bytecode (`8eedb5648e`: module 16→0, all 5 dropped defaults restored verbatim from GT).

### 2. EXTEND `utils/semantic_operators/decorator_default.py` (not a new operator)
Investigation found `decorator_default` **already** restores dropped positional
defaults — but its `_def_sites` recovery (line ~192) only fires when
`flags == 0x01` *exactly*, and it keys solely on the `MAKE_FUNCTION` flags arg. That
misses three real cases (all verified against `8eedb5648e`, where 5 functions have GT
`flags=0x5` (defaults+annotations) vs derived `flags=0x4` — clean dropped-defaults):

- **(a) Annotation/other-flag entanglement (3.11–3.12):** recover the defaults tuple
  when `flags & 0x01` even if other bits are set. Attributes are pushed in bit order
  (defaults, kwdefaults, annotations, closure) below the code load; locate the
  defaults `LOAD_CONST` tuple by counting set-bit attribute slots
  (`defaults @ code_idx - 1 - popcount(flags)`), validating it is a `LOAD_CONST`
  tuple. Fire only when the *sole* flag difference is the `0x01` bit
  (`gt.flags == der.flags | 0x01`) — already enforced at line 283.
- **(b) 3.13–3.15 `SET_FUNCTION_ATTRIBUTE` form:** in 3.13+ `MAKE_FUNCTION` takes no
  flags; defaults are set by a following `SET_FUNCTION_ATTRIBUTE` with oparg `0x01`
  (value = a `LOAD_CONST` tuple before the code load). Detect defaults-present via the
  trailing `SET_FUNCTION_ATTRIBUTE` chain; represent it in `_DefSite` as a synthesized
  `flags` bit so the downstream `_default_candidate` logic is unchanged.
- **(c) Multiple diverging functions:** fix ONE uniquely-resolvable diverging function
  per call and defer the rest — the prepass loops to fixpoint and re-invokes on the
  residual, so all get restored across iterations, each independently oracle-gated.

Reconstruct + source edit are unchanged (read GT defaults tuple verbatim, align to
trailing positional params via `co_argcount`/`co_varnames`, splice `=repr(value)`).
Defer on non-literal reprs, kw-only defaults, non-unique def, or param-name mismatch.

### 3. annotation-strip — DROPPED (verified non-deterministic)
Investigated `5235c14286` (the only annotation near-miss) directly: derived line 9 is
`integer: ppc.ipv4_address() = pp.Word(pp.nums)` where GT is `ipAddress = ppc.ipv4_address()`
— a catastrophic mis-decompilation (wrong target name, the RHS became the annotation, a
different value), NOT a clean `x: T = v` strip. Recovering GT requires re-associating the
name, value, and annotation — LLM territory, not a mechanical edit. So no annotation-strip
operator is built. (Same evidence-first discipline as the rejected jump-sense operator:
verify the pattern is genuinely deterministic before building.)

### 4. Wiring
Both register in the diffbc operator tuple in `run_deterministic_prepass`
(`utils/reattach_source_code_object.py:3942`), next to `decorator_default`. Ordered
after the existing high-precision operators; each guards its own regime and defers.

## Testing (TDD, version-parametrized)

For each operator, **failing test first**, built from real bytecode:
1. Unit tests compile a minimal source pair (GT with defaults/annotation vs derived
   without) across **3.11, 3.12, 3.13, 3.14, 3.15** via the repo's cross-version
   compile infra, run `compare_pyc`, and assert the operator fires and returns the
   correct source edit. Detection differences (flag vs SET_FUNCTION_ATTRIBUTE;
   SETUP_ANNOTATIONS vs `__annotate__`) are covered by explicit per-version cases.
2. Negative tests: operator returns `None` when GT has no defaults / no annotation
   mismatch, and on ambiguous (non-literal default, non-Name target) inputs.
3. End-to-end: `default_args` converts `8eedb5648e`, `annotation_strip` converts
   `5235c14286` (both on-disk), via the prepass, `all_equal=True`, zero regression.
4. CPU A/B re-measure (component 1) before/after: file-perfect strictly up, no
   regressions; full suite green.

## Outcome (as built)

Delivered: the **default-args extension** (component 2) — covers 3.11–3.15
(flag form + `SET_FUNCTION_ATTRIBUTE`), annotation-entanglement, and multiple diverging
functions. Proven on the real `8eedb5648e`: module 16→0, all 5 dropped defaults restored
verbatim from GT. +7 tests, full suite 691 green, oracle gate unchanged. It generalizes to
any code object with dropped positional defaults across the 3000-file run.

Dropped after investigation: annotation-strip (component 3 — the one near-miss is a mangled
mis-decompilation, LLM territory) and the loop-shape case `6bdb87bc50` (structural, LLM).

File-level A/B on hybrid-100 requires the full det+LLM run (GPU): `8eedb5648e`'s functions
need the LLM, so the operator's file-conversion only shows there. Expected: `8eedb5648e`
(and any other file whose sole residual after LLM function-repair is dropped module defaults)
converts. Honest ceiling is small on hybrid-100 (near-misses), but the operator is
proven-correct and the corpus generalization is the real value.
