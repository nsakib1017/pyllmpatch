# pyllmpatch

`pyllmpatch` repairs Python decompiler output. It has two complementary repair
pipelines:

- A syntactic repair pipeline that turns uncompilable decompiled Python into a
  compilable `.py`/`.pyc` pair.
- A semantic repair pipeline that compares ground-truth and derived `.pyc`
  files at the code-object level, edits localized source fragments, recompiles,
  and keeps candidates that improve bytecode distance or PyLingual verification.

The project is research-oriented: most commands operate against dataset rows
identified by `file_hash`, `source`, `error_type`, bytecode paths, and derived
source paths.

## Repository Layout

```text
main.py                         CLI entry point for syntactic and semantic repair
pipeline/
  config.py                     Environment-backed runtime configuration
  runner.py                     Dataset-level syntactic repair orchestration
  repair_engine.py              Syntax prompt construction and LLM dispatch
  code_object_repair_loop.py    Semantic repair CLI, dataset runner, fragment fixers
  dataset.py                    Dataset filtering and previous-run selection
  logging_utils.py              JSONL logging and compile-failure cleanup
model/
  loader.py                     Cached local model loading through Unsloth
  inference.py                  Local chat-template inference wrapper
utils/
  reattach_source_code_object.py Source mapping, source replacement, semantic loop core
  pyc_code_object_distance.py   Code-object bytecode and CFG distance metrics
  map_source_code_objects.py    AST source spans mapped to `.pyc` code objects
  file_helpers.py               Dataset path resolution, syntax context windows
  generate_bytecode.py          Version-aware Python compilation helpers
  delete_only_compilation.py    Non-LLM syntax repair by deleting bad spans
  providers.py                  API and local LLM provider registry
  mine_semantic_repair_actions.py
                                 Builds reusable action-pattern telemetry
finetuning/                     Fine-tuning scripts for syntax and semantic repair data
tools/                          Reporting and presentation helpers
tests/                          Focused unit tests for semantic token limits and dataset flow
```

## Setup

Create and activate a virtual environment, then install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Create a local environment file:

```bash
cp .env.example .env
```

At minimum, configure the project root, dataset, and decompiled-file roots:

```dotenv
PROJECT_ROOT_DIR=/absolute/path/to/this/repo
ROOT_FOR_FILES=/absolute/path/to/decompiled/file/root
BASE_DIR_PYTHON_FILES_PYLINGUAL=pylingual
BASE_DIR_PYTHON_FILES_PYPI=pypi
BASE_DATASET_NAME=your_dataset.csv
MAX_EXAMPLE_RUNTIME_MIN=1
```

LLM-backed repair also needs the provider key or local model paths implied by
`utils/providers.py`:

```dotenv
OPENAI_API_KEY=
DEEPSEEK_API_KEY=
GEMINI_API_KEY=
HUGGINGFACE_HUB_TOKEN=
```

Compilation uses `utils/generate_bytecode.py`. Native compilation is used when
the requested bytecode version matches the running interpreter. For Python
3.8-3.14, the code tries `uvx --python <version>` first and falls back to
`pyenv`; older or unavailable versions require `pyenv`.

## Running

Show the available commands:

```bash
python main.py --help
```

### Syntactic Repair

The default command runs the syntactic repair experiment:

```bash
python main.py
```

That is equivalent to:

```bash
python main.py syntactic-repair
```

Useful filters:

```bash
python main.py syntactic-repair --source PyPi --limit 25
```

Useful syntactic repair environment options:

```dotenv
USE_LOCAL_LLM=True
LOCAL_LLM_IDX=0
NO_OF_MAX_RETRIES=0
CONFIG_IDX_START=0
CONFIG_IDX_RANGE=1
MAX_WHOLE_FILE_BYTES=1048576
ENABLE_SYNTAX_EXPLANATION=1
ENABLE_WHOLE_FILE_REPAIR=1
DELETE_ONLY_MODE=false
ENABLE_DELETE_ONLY_FALLBACK=true
DELETE_ONLY_MAX_ITERS=5000
DELETE_ONLY_BASE_WINDOW=1
DELETE_ONLY_MAX_DELETED_RATIO=0.95
```

Run without LLM calls by enabling delete-only mode:

```bash
DELETE_ONLY_MODE=true python main.py syntactic-repair
```

Syntactic repair artifacts are written under:

```text
results/experiment_outputs/<timestamp>/<run_id>/
```

### Semantic Repair

Run semantic repair for one file by passing the ground-truth bytecode, derived
bytecode, and derived source file:

```bash
python main.py semantic-repair \
  /path/to/ground_truth.pyc \
  /path/to/derived.pyc \
  /path/to/derived_source.py \
  --output-dir results/semantic_repair/manual_case \
  --json-out results/semantic_repair/manual_case/result.json
```

By default, semantic repair uses the `oracle` fixer, which reads the
ground-truth source fragment and is useful for validating the reattachment and
acceptance loop. Use `--fixer llm` for generated fragment repair:

```bash
python main.py semantic-repair \
  /path/to/ground_truth.pyc \
  /path/to/derived.pyc \
  /path/to/derived_source.py \
  --fixer llm \
  --llm-provider Google \
  --llm-model gemini-2.5-flash-lite
```

Run semantic repair over `semantic_error` rows in the configured dataset:

```bash
python main.py semantic-repair --dataset-mode --limit 10
python main.py semantic-repair --dataset-mode --row-range 10:20
python main.py semantic-repair --dataset-mode --source PyPi --file-hash <hash>
```

Common semantic repair options:

```bash
--dataset-path /path/to/dataset.csv
--row-range 10:20
--file-hash <hash>
--skip-pylingual-verification
--skip-step-verification
--keep-non-improving
--max-iterations 2
--sample-timeout-seconds 3600
--sample-hard-timeout-seconds 10800
--sample-timeout-min-improvement-delta 1
--defer-preflight-risky-samples
--defer-timeout-no-improvement
--process-easy-cases-first
--output-dir /path/to/output_dir
--json-out /path/to/result.json
```

Dataset-mode semantic repair checks each sample every 3600 seconds by default.
If no combined-distance improvement has been made since the previous checkpoint,
the sample is skipped or deferred. If the combined distance improved, the
checkpoint is reset and the sample continues until the next checkpoint. The hard
cap defaults to 10800 seconds. Set either timeout to `0` to disable it.

## Architectural Design

### High-Level Orchestration

`main.py` is intentionally thin. It parses the command, imports the needed
pipeline lazily, and dispatches to one of two orchestration layers:

```text
main.py
  syntactic-repair -> pipeline.config.load_runtime_config
                   -> pipeline.runner.run_experiment

  semantic-repair  -> direct CodeObjectRepairLoop.run
                   -> or pipeline.code_object_repair_loop.run_dataset_repair_loop
```

The rest of the system is split by responsibility:

- `pipeline/*` owns command-level orchestration, runtime configuration, dataset
  filtering, and run output layout.
- `utils/*` owns reusable mechanics: path resolution, bytecode compilation,
  syntax context extraction, source/code-object mapping, distance metrics,
  and source reattachment.
- `model/*` owns local model loading and inference.
- `finetuning/*` and selected `utils/*` scripts build training, telemetry, and
  prompt-refresh datasets from accepted repairs.

### Configuration and Dataset Resolution

`pipeline/config.py` loads `.env` at import time and exposes the project-wide
paths used by both pipelines:

- `PROJECT_ROOT_DIR` anchors dataset and telemetry files.
- `ROOT_FOR_FILES`, `BASE_DIR_PYTHON_FILES_PYLINGUAL`, and
  `BASE_DIR_PYTHON_FILES_PYPI` anchor decompiled file storage.
- `BASE_DATASET_NAME` is resolved as
  `PROJECT_ROOT_DIR / "dataset" / BASE_DATASET_NAME`.
- `ACCEPTED_CODE_OBJECT_FILE` and `ACCEPTED_CODE_OBJECT_TELEMETRY_FILE` point to
  JSONL stores of accepted semantic repairs.

Dataset-backed commands expect rows with fields such as `file_hash`, `source`,
`error_type`, `file_path`, `file`, `bytecode_version`, and compile-error
metadata. The `utils/file_helpers.py` resolver handles the two currently
supported storage layouts:

- `source == "PyPi"`: original `.pyc` from `__pycache__`, derived source and
  derived `.pyc` from `decompiled_output_pylingual`.
- other sources: original top-level `.pyc`, derived `indented_*.py`, and derived
  `indented_*.pyc` from `decompiler_output`.

### Syntactic Repair Pipeline

The syntactic path is optimized around compilation success.
`pipeline.runner.run_experiment` performs the dataset-level workflow:

1. Load runtime config from `.env`.
2. Read the base dataset, optionally subtract rows already present in a previous
   run log, filter to `syntactic_error`, and apply `--source` / `--limit`.
3. Shuffle the selected rows with a fixed seed.
4. For each row, resolve the source file, bytecode version, decompiler name, and
   output directory.
5. Skip files over `MAX_WHOLE_FILE_BYTES`.
6. Copy the input file into the run directory and repair only that copy.
7. Compile the candidate with `utils.generate_bytecode.compile_version`.
8. Append one JSONL record per sample to the run log.

There are two syntactic repair modes:

#### Delete-only mode

When `DELETE_ONLY_MODE=true`, no LLM is called. The runner sends the copied file
to `utils.delete_only_compilation.delete_lines_until_compilable_with_oracle`.
That utility repeatedly uses the current compiler error to propose small source
deletions, recompiles after each deletion, and stops when the file compiles or
when deletion guards are reached.

This mode is useful as a non-LLM baseline and as a fallback when LLM repair
exhausts its retry budget.

#### LLM mode

When delete-only mode is disabled, the runner selects an LLM config through
`pipeline.runner.select_llm`:

- `USE_LOCAL_LLM=True` selects from `utils.providers.OPEN_LLM_MODELS`.
- `USE_LOCAL_LLM=False` selects from `utils.providers.LLM_MODELS`.
- `LOCAL_LLM_IDX` chooses the entry, with index `0` as fallback.

`pipeline.repair_engine.attempt_repair` then tries one or more strategies:

- `syntax_context`: localize a fragment around the compiler error using
  `utils.file_helpers.segment_syntax_context`.
- `whole_file`: optionally repair the full file on late retries when
  `ENABLE_WHOLE_FILE_REPAIR=1`.

Syntax context selection is error-aware. `file_helpers` classifies delimiter,
indentation, numeric, and generic syntax errors, then tries line, block, or
delimiter windows with controlled expansion. The selected fragment is
dedented before prompting, and the returned replacement is reindented and
reattached with `align_indentation` and `reattach_block`.

For local LLMs, `repair_engine` can make a preliminary root-cause explanation
call when `ENABLE_SYNTAX_EXPLANATION=1`, then uses that explanation in the
repair prompt. API-backed models use the provider dispatch in
`utils.providers.make_llm_call`.

After every repair attempt, the runner recompiles the full copied source. Failed
attempts update the current error message and retry until the configured limit
or runtime guard is reached. If `ENABLE_DELETE_ONLY_FALLBACK=true`, the final
LLM candidate is sent through the delete-only compiler loop before the sample is
marked failed.

### Semantic Repair Pipeline

The semantic path is optimized around reducing bytecode drift between a
ground-truth `.pyc` and a derived `.pyc`.

There are two entry shapes:

- Direct mode: `main.py semantic-repair <gt_pyc> <derived_pyc> <derived_source>`.
- Dataset mode: `run_dataset_repair_loop` reads `semantic_error` rows, resolves
  paths from `file_hash` and `source`, and runs one repair loop per row.

At the center is `pipeline.code_object_repair_loop.CodeObjectRepairLoop`. It is
a small adapter around `utils.reattach_source_code_object.repair_mismatching_code_objects`.
The adapter supplies a `fragment_fixer` callback and collects LLM prompt/call
records when the fixer supports them.

The semantic loop follows this feedback cycle:

```text
ground-truth .pyc + derived .pyc + derived source
  -> compare code-object distances
  -> select repair targets
  -> map target qualname back to source span
  -> build repair context from metadata, bytecode windows, PyLingual failures,
     rejected attempts, and prior accepted repairs
  -> generate one or more candidate source fragments
  -> normalize and reattach candidate into the source file
  -> compile to a new .pyc
  -> recompute distance and optionally run PyLingual verification
  -> accept only improving candidates unless --keep-non-improving is set
  -> repeat over newly computed targets up to --max-iterations
```

#### Code-object distance and target selection

`utils.pyc_code_object_distance.py` loads `.pyc` files through PyLingual bytecode
utilities and computes:

- instruction edit distance,
- control-flow distance from basic-block graphs,
- unmatched-code-object penalties,
- a combined distance used for ranking and timeout progress.

`utils.reattach_source_code_object.py` consumes those rows to select target
sets:

- mismatched existing code objects,
- missing code objects,
- extra derived code objects,
- module-level mismatches,
- expression-child mismatches that can sometimes be repaired by editing the
  parent code object.

Targets are recomputed after accepted steps, so later repairs operate on the
current best source and bytecode rather than the original derived file.

#### Source mapping and reattachment

Semantic repair needs to edit a precise source span without corrupting the rest
of the file. `utils.map_source_code_objects.py` collects source code objects
from the AST and code objects from the `.pyc`, then maps them by qualname,
line-number evidence, occurrence index, sibling ordinal, and ordinal path.

For existing targets, `extract_source_segment` reads the mapped span and
`choose_best_reattachment` normalizes candidate indentation before replacing
the span. The loop also validates that reattaching a candidate does not damage
the surrounding code-object structure.

For missing targets, `insert_missing_source_segment` finds the nearest existing
parent and inserts a normalized function, async function, or class fragment.
Unsupported missing statement kinds are logged as rejected steps.

For extra targets, the loop deletes safely mapped function/class spans. If
plain deletion breaks syntax, it falls back to replacing the span with a `pass`
block when that is structurally valid.

For module-level repair, full-file rewriting is intentionally avoided. The loop
uses PyLingual failure context to find a failed line, localizes that line to a
top-level statement, and asks the fixer to repair only that statement.

#### Fragment fixers

Semantic repair is pluggable through the `FragmentFixer` interface:

- `OracleFragmentFixer` extracts the corresponding source fragment from the
  ground-truth source. This is used to validate mapping, reattachment,
  compilation, and acceptance mechanics.
- `LLMFragmentFixer` builds compact bytecode-aware prompts and asks an API or
  local model for candidate fragments.

The LLM fixer records every prompt under the row output directory, tracks token
counts, and skips provider calls before dispatch when the prompt exceeds the
configured token threshold. For normal existing-code-object repairs it can
generate multiple candidates in one semantic step, each with a different
strategy:

- retrieval-guided,
- control-flow residual,
- call/attribute residual,
- metadata/literal/name residual,
- conservative minimal delta.

Duplicate candidates are detected by both text hash and AST structural hash.
Rejected attempts are summarized into future prompts so repeated candidate
shapes are discouraged.

#### Semantic prompt inputs

`LLMFragmentFixer` composes prompts from several evidence sources:

- code-object metadata deltas such as names, constants, variables, argument
  shape, and flags,
- localized instruction diffs and fallback instruction windows,
- source line numbers and an indentation contract,
- PyLingual failure offsets and failed line numbers,
- rejected candidate history for the same target,
- accepted-case telemetry from `ACCEPTED_CODE_OBJECT_TELEMETRY_FILE`,
- mined action patterns from
  `results/semantic_repair_action_patterns/semantic_repair_action_patterns.jsonl`.

This design keeps the editable surface small while giving the model enough
bytecode evidence to prefer minimal semantic edits over broad rewrites.

#### Acceptance and verification

Every semantic candidate is evaluated by compiling the updated source and
recomputing code-object distance. When PyLingual verification is enabled, the
loop also runs PyLingual equality checks after each step and at finalization.

Candidates are accepted when they improve the measured state according to the
current summary and verification results. Rejected candidates are retained in
the step log with their replacement text, score deltas, prompt metadata, parse
status, structural validation status, and rejection reason. Passing
`--keep-non-improving` disables this acceptance filter and keeps generated
candidates for analysis.

Accepted semantic steps are appended to the configured accepted-code-object
JSONL stores. Those records are later used by prompt-refresh, action-pattern
mining, and fine-tuning scripts.

#### Dataset-mode controls

`run_dataset_repair_loop` adds operational controls around the per-file loop:

- `--process-easy-cases-first` preflights all selected rows with initial
  distance metrics and sorts lower-risk rows first.
- `--defer-preflight-risky-samples` sends rows with too many initial targets,
  high combined distance, missing targets, or extra targets to
  `semantic_repair_deferred_*.csv`.
- `--defer-timeout-no-improvement` records stalled rows in the deferred CSV
  instead of the main result CSV.
- timeout checkpoints stop rows that have made no distance progress and keep
  the best result for rows that improved before the hard cap.

Dataset-mode output includes:

```text
<output-dir>/
  run_log_<run_id>_<dataset>.jsonl
  semantic_repair_results_<dataset>.csv
  semantic_repair_deferred_<dataset>.csv
  semantic_repair/<source>/<file_hash>/
    result.json
    prompts/
    fragments/
    __pycache__/
    step<N>_<source_stem>.py
```

### LLM Provider Boundary

All provider configuration lives in `utils/providers.py`.

- `LLM_MODELS` contains API-backed OpenAI, DeepSeek, and Google model configs.
- `OPEN_LLM_MODELS` contains local checkpoint configs and tokenizer paths.
- `find_llm_config(provider, model)` validates command-line selections.
- `make_llm_call_from_config` dispatches to an API client or to
  `model.inference.call_llm_with_message`.

Local inference is loaded through `model.loader.load_model_once`, which caches
model/tokenizer pairs for the process and refuses to silently fall back to the
Hugging Face Hub when a configured local path is missing.

### Logging and Artifacts

Syntactic repair writes one compact JSONL record per sample plus repaired or
failed source artifacts in the run tree.

Semantic repair writes richer per-row artifacts:

- `result.json` with initial/final summaries, steps, targets, final source,
  final `.pyc`, PyLingual verification, and LLM calls,
- prompt JSON files for LLM semantic candidates,
- `.pyfrag` files for accepted or selected candidate fragments,
- accepted-code-object JSONL records for downstream mining and fine-tuning,
- result and deferred CSVs in dataset mode.

The system favors append-only run records. It does not mutate the original
dataset files or original decompiled inputs; all repair attempts are made in
run-specific output directories.

## Tests

Run the current unit test suite:

```bash
python -m pytest
```

The existing tests focus on semantic token-limit behavior, timeout/improvement
logic, and dataset-mode preflight/defer orchestration.

## Development Notes

- Prefer adding new runtime flags in `pipeline/config.py` or the relevant
  `main.py` subparser, then thread them through the orchestration layer.
- Add a new semantic repair backend by implementing `FragmentFixer` or
  `generate_candidates` in `pipeline/code_object_repair_loop.py`.
- Keep source edits localized. The semantic loop assumes candidate fragments can
  be normalized, reattached, compiled, and scored independently.
- When changing prompt shape, update prompt telemetry consumers such as
  `utils/rebuild_semantic_prompt_refresh_dataset.py` and action-pattern mining
  scripts if their reconstruction inputs change.
