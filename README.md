# pyllmpatch

`pyllmpatch` repairs Python decompiler output with two pipelines:

- **Syntactic repair**: make uncompilable decompiled Python compile again.
- **Semantic repair**: reduce bytecode drift between a ground-truth `.pyc` and
  a derived `.pyc` by editing localized source fragments.

The project is dataset-oriented. Most commands work from rows keyed by
`file_hash`, `source`, `error_type`, bytecode paths, and derived source paths.
Manual single-file semantic repair is also supported.

## Contents

- [Which Pipeline Should I Run?](#which-pipeline-should-i-run)
- [Repository Layout](#repository-layout)
- [Setup](#setup)
- [Running](#running)
- [Architecture](#architecture)
- [Artifacts and Logs](#artifacts-and-logs)
- [Tests](#tests)
- [Development Notes](#development-notes)

## Which Pipeline Should I Run?

| Goal | Command | Repair unit | Main success signal |
| --- | --- | --- | --- |
| Make decompiled source compile | `syntactic-repair` | Error-localized source window or whole file | Python compiler accepts the output |
| Validate semantic source reattachment with known-good fragments | `semantic-repair --fixer oracle` | Mapped code-object fragment | Combined bytecode distance improves |
| Generate semantic fixes with an LLM | `semantic-repair --fixer llm` | Mapped code-object fragment, insertion, deletion, or module statement | Combined distance and optional PyLingual checks improve |
| Run semantic repair over a dataset | `semantic-repair --dataset-mode` | One dataset row at a time | Result/deferred CSVs plus per-row `result.json` |

## Repository Layout

```text
main.py                         CLI entry point
pipeline/
  config.py                     Environment-backed runtime configuration
  runner.py                     Syntactic repair dataset runner
  repair_engine.py              Syntax prompt construction and LLM dispatch
  code_object_repair_loop.py    Semantic CLI, dataset runner, fragment fixers
  dataset.py                    Dataset filtering and previous-run selection
  logging_utils.py              JSONL logging and compile-failure cleanup
model/
  loader.py                     Cached local model loading through Unsloth
  inference.py                  Local chat-template inference wrapper
utils/
  reattach_source_code_object.py Semantic repair core: mapping, reattachment, acceptance
  pyc_code_object_distance.py   Instruction, CFG, unmatched-object distance metrics
  map_source_code_objects.py    AST source spans mapped to `.pyc` code objects
  file_helpers.py               Dataset path resolution and syntax context windows
  generate_bytecode.py          Version-aware Python compilation
  delete_only_compilation.py    Non-LLM syntax repair by deleting bad spans
  providers.py                  API and local LLM provider registry
  mine_semantic_repair_actions.py
                                 Action-pattern mining from accepted repairs
finetuning/                     Fine-tuning scripts for syntax and semantic repair data
tools/                          Reporting and presentation helpers
tests/                          Focused unit tests for semantic orchestration
```

## Setup

### Prerequisites

- Python with `venv`.
- `uvx` or `pyenv` if you need to compile bytecode for Python versions other
  than the interpreter running the command.
- Provider credentials for API-backed LLM repair, or local model checkpoints for
  local LLM repair.
- A dataset and decompiled-file tree matching the path conventions described in
  [Dataset and Path Resolution](#dataset-and-path-resolution).

### Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

### Required Environment

`pipeline/config.py` loads `.env` at import time. The dataset-backed commands
need these values:

| Variable | Used for |
| --- | --- |
| `PROJECT_ROOT_DIR` | Repository root. Dataset and telemetry files are resolved under this path. |
| `ROOT_FOR_FILES` | Root directory containing decompiled file trees. |
| `BASE_DIR_PYTHON_FILES_PYLINGUAL` | Subdirectory under `ROOT_FOR_FILES` for non-PyPi/PyLingual-style rows. |
| `BASE_DIR_PYTHON_FILES_PYPI` | Subdirectory under `ROOT_FOR_FILES` for PyPi rows. |
| `BASE_DATASET_NAME` | CSV filename resolved as `PROJECT_ROOT_DIR/dataset/<name>`. |
| `MAX_EXAMPLE_RUNTIME_MIN` | Syntactic repair runtime guard used by `pipeline.runner`. |

Example:

```dotenv
PROJECT_ROOT_DIR=/absolute/path/to/pyllmpatch
ROOT_FOR_FILES=/absolute/path/to/decompiled/file/root
BASE_DIR_PYTHON_FILES_PYLINGUAL=pylingual
BASE_DIR_PYTHON_FILES_PYPI=pypi
BASE_DATASET_NAME=your_dataset.csv
MAX_EXAMPLE_RUNTIME_MIN=1
```

LLM-backed repair uses these when the selected provider requires them:

```dotenv
OPENAI_API_KEY=
DEEPSEEK_API_KEY=
GEMINI_API_KEY=
HUGGINGFACE_HUB_TOKEN=
```

Local model paths are configured in `utils/providers.py` through
`OPEN_LLM_MODELS`.

## Running

Show the command surface:

```bash
python main.py --help
python main.py syntactic-repair --help
python main.py semantic-repair --help
```

### Syntactic Repair

The default command runs the syntactic repair dataset experiment:

```bash
python main.py
```

Equivalent explicit command:

```bash
python main.py syntactic-repair
```

Filter by source or limit the selected rows:

```bash
python main.py syntactic-repair --source PyPi --limit 25
```

Useful syntactic repair environment options:

```dotenv
USE_LOCAL_LLM=True
LOCAL_LLM_IDX=0
NO_OF_MAX_RETRIES=0
MAX_WHOLE_FILE_BYTES=1048576
ENABLE_SYNTAX_EXPLANATION=1
ENABLE_WHOLE_FILE_REPAIR=1
DELETE_ONLY_MODE=false
ENABLE_DELETE_ONLY_FALLBACK=true
DELETE_ONLY_MAX_ITERS=5000
DELETE_ONLY_BASE_WINDOW=1
DELETE_ONLY_MAX_DELETED_RATIO=0.95
```

Run a non-LLM baseline by enabling delete-only mode:

```bash
DELETE_ONLY_MODE=true python main.py syntactic-repair --limit 10
```

### Semantic Repair

Run direct semantic repair for one case:

```bash
python main.py semantic-repair \
  /path/to/ground_truth.pyc \
  /path/to/derived.pyc \
  /path/to/derived_source.py \
  --output-dir results/semantic_repair/manual_case \
  --json-out results/semantic_repair/manual_case/result.json
```

By default, direct semantic repair uses `--fixer oracle`, which extracts the
ground-truth source fragment. That is useful for validating mapping,
reattachment, compilation, and acceptance mechanics.

Use an LLM fixer for generated semantic fragments:

```bash
python main.py semantic-repair \
  /path/to/ground_truth.pyc \
  /path/to/derived.pyc \
  /path/to/derived_source.py \
  --fixer llm \
  --llm-provider Google \
  --llm-model gemini-2.5-flash-lite
```

Run semantic repair over dataset rows where `error_type == semantic_error`:

```bash
python main.py semantic-repair --dataset-mode --limit 10
python main.py semantic-repair --dataset-mode --row-range 10:20
python main.py semantic-repair --dataset-mode --source PyPi --file-hash <hash>
```

Common semantic repair controls:

```bash
--dataset-path /path/to/dataset.csv
--max-iterations 2
--skip-pylingual-verification
--skip-step-verification
--keep-non-improving
--sample-timeout-seconds 3600
--sample-hard-timeout-seconds 10800
--sample-timeout-min-improvement-delta 1
--process-easy-cases-first
--defer-preflight-risky-samples
--defer-timeout-no-improvement
--output-dir /path/to/output_dir
```

Dataset-mode semantic repair has a progress-aware timeout policy. At each
checkpoint, a sample continues only if combined bytecode distance improved by at
least `--sample-timeout-min-improvement-delta`. The hard timeout keeps the best
improving result or skips/deferred rows that never improved. Set either timeout
to `0` to disable it.

## Architecture

### Control Flow

`main.py` is intentionally thin: it parses arguments, imports the relevant
pipeline lazily, and dispatches.

```text
main.py
  syntactic-repair
    -> pipeline.config.load_runtime_config()
    -> pipeline.runner.run_experiment()
    -> pipeline.repair_engine.attempt_repair()
    -> utils.generate_bytecode.compile_version()

  semantic-repair direct mode
    -> pipeline.code_object_repair_loop.CodeObjectRepairLoop
    -> utils.reattach_source_code_object.repair_mismatching_code_objects()

  semantic-repair dataset mode
    -> pipeline.code_object_repair_loop.run_dataset_repair_loop()
    -> one CodeObjectRepairLoop per selected row
```

The boundary between `pipeline` and `utils` is deliberate:

- `pipeline/*` owns command orchestration, runtime options, dataset iteration,
  provider selection, and run-directory layout.
- `utils/*` owns reusable mechanics: path resolution, syntax localization,
  bytecode compilation, source-to-code-object mapping, distance metrics, and
  source reattachment.
- `model/*` owns local model loading and inference.
- `finetuning/*` consumes accepted repair telemetry to train local models.

### Dataset and Path Resolution

Dataset-backed repair starts from a CSV row. The important columns are:

| Column | Meaning |
| --- | --- |
| `file_hash` | Stable key used to locate files under the configured roots. |
| `source` | Chooses the expected file-tree layout, for example `PyPi`. |
| `error_type` | Selects `syntactic_error` or `semantic_error` rows. |
| `file_path` / `file` | Optional direct path/name used by syntactic repair. |
| `bytecode_version` | Python version used for syntactic repair compilation. |
| `error`, `error_message`, `error_description` | Compiler error context for syntax prompts and delete-only repair. |

`utils/file_helpers.py` resolves files from `file_hash` and `source`.

For `source == "PyPi"`:

```text
<BASE_DIR_PYTHON_FILES_PYPI>/<file_hash>/
  __pycache__/*.cpython-310.pyc
  decompiled_output_pylingual/
    decompiled_*.py
    __pycache__/decompiled_*.cpython-310.pyc
```

For other sources:

```text
<BASE_DIR_PYTHON_FILES_PYLINGUAL>/<file_hash>/
  *.pyc
  decompiler_output/
    indented_*.py
    indented_*.pyc
```

### Syntactic Repair Design

The syntactic pipeline is a compile-and-retry loop around a copied input file.
Original inputs are not mutated.

```text
dataset row
  -> resolve source file and bytecode version
  -> copy source into run directory
  -> choose repair mode
     -> delete-only repair
     -> or LLM repair of syntax-localized fragment / whole file
  -> compile candidate with the requested Python version
  -> retry with the new compiler error or write final artifact
  -> append JSONL run record
```

The LLM path localizes syntax errors before prompting. `file_helpers` classifies
the compiler message as delimiter, indentation, numeric, or generic, then tries
line, block, or delimiter windows with bounded expansion. The selected snippet
is dedented for the model and reattached with indentation alignment after the
model returns code.

The delete-only path uses `utils.delete_only_compilation` to repeatedly remove
small candidate spans guided by the latest compiler error. It can be used as the
primary mode with `DELETE_ONLY_MODE=true`, or as a fallback after LLM retries
with `ENABLE_DELETE_ONLY_FALLBACK=true`.

### Semantic Repair Design

Semantic repair operates on code objects rather than whole files. Each accepted
step creates a new source file and `.pyc`; subsequent targets are recomputed
against that current state.

```text
ground-truth .pyc + derived .pyc + derived source
  -> compare code-object distance
  -> select mismatched, missing, extra, and module targets
  -> map target qualname back to source span
  -> generate candidate fragment(s)
  -> normalize and reattach source
  -> compile candidate .pyc
  -> recompute distance and optionally run PyLingual verification
  -> accept improving candidates
  -> repeat until no targets remain or max iterations is reached
```

The core loop lives in
`utils.reattach_source_code_object.repair_mismatching_code_objects`. The
`pipeline.code_object_repair_loop.CodeObjectRepairLoop` wrapper supplies a
pluggable fragment fixer and captures LLM call records.

#### Distance and Target Selection

`utils.pyc_code_object_distance.py` loads editable bytecode through the local
PyLingual checkout and computes:

- instruction edit distance,
- control-flow distance from basic-block graphs,
- penalties for missing or extra code objects,
- a combined distance used for candidate ranking and timeout progress.

`utils.reattach_source_code_object` uses those rows to select:

- mismatched existing code-object fragments,
- missing code objects that may need insertion,
- extra derived code objects that may need deletion,
- module-level mismatches,
- expression-child mismatches where the parent may be the practical repair
  target.

#### Source Mapping and Reattachment

`utils.map_source_code_objects.py` maps AST source spans to `.pyc` code objects
using qualname, line evidence, occurrence index, sibling ordinal, and ordinal
path. Semantic repair uses that mapping differently for each operation:

| Operation | Behavior |
| --- | --- |
| `repair_source_fragment` | Extract mapped source, replace it with a normalized candidate, then validate structure. |
| `insert_missing` | Find nearest mapped parent and insert a normalized function, async function, or class fragment. |
| `delete_extra` | Delete a safely mapped function/class span, falling back to `pass` when needed. |
| `repair_module_statement` | Use PyLingual failure context to localize one top-level statement; full-file module rewrites are avoided. |

#### Fragment Fixers

Semantic candidates come from a `FragmentFixer`:

- `OracleFragmentFixer` extracts the matching ground-truth source fragment. Use
  it to validate the non-LLM mechanics.
- `LLMFragmentFixer` builds bytecode-aware prompts and asks an API or local
  model to return replacement fragments.

For normal existing-fragment repairs, `LLMFragmentFixer` can generate multiple
candidates with different strategies:

- retrieval-guided,
- control-flow residual,
- call/attribute residual,
- metadata/literal/name residual,
- conservative minimal delta.

Candidates are deduplicated by text hash and AST structural hash. Rejected
attempts are summarized back into later prompts so repeated candidate shapes are
discouraged.

#### Prompt Inputs

Semantic prompts combine:

- source fragment with line-number display,
- indentation and return contracts,
- compact ground-truth and derived code-object metadata,
- localized instruction diffs or fallback bytecode windows,
- PyLingual failed offset and failed line information,
- rejected-attempt feedback,
- accepted-case telemetry from `ACCEPTED_CODE_OBJECT_TELEMETRY_FILE`,
- mined action patterns from
  `results/semantic_repair_action_patterns/semantic_repair_action_patterns.jsonl`.

Before any provider call, the prompt is token-counted with provider/model
settings. Oversized prompts are skipped and logged without calling the provider.

### LLM Provider Boundary

All model selection is centralized in `utils/providers.py`.

| Config | Meaning |
| --- | --- |
| `LLM_MODELS` | API-backed OpenAI, DeepSeek, and Google models. |
| `OPEN_LLM_MODELS` | Local checkpoint configs with model and tokenizer paths. |
| `find_llm_config(provider, model)` | Validates CLI provider/model selections. |
| `make_llm_call_from_config(...)` | Dispatches to API clients or local inference. |

Local inference goes through `model.inference.call_llm_with_message`, which uses
`model.loader.load_model_once` to cache model/tokenizer pairs. Local paths are
validated before load so a typo does not silently fall back to downloading from
the Hugging Face Hub.

## Artifacts and Logs

Syntactic repair writes under:

```text
results/experiment_outputs/<timestamp>/<run_id>/
  run_log_<run_id>_<dataset>.jsonl
  <decompiler>/<dataset>/<bytecode_version>/<file_hash>/
    syntax_repaired_*.py
    syntax_repaired_*.pyc
    syntax_failed_repaired_*.py
    delete_only_*.jsonl
```

Semantic dataset mode writes under the selected output directory:

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

Accepted semantic steps are also appended to the configured dataset JSONL files:

```text
dataset/<ACCEPTED_CODE_OBJECT_FILE>
dataset/<ACCEPTED_CODE_OBJECT_TELEMETRY_FILE>
```

These accepted records feed action-pattern mining, prompt-refresh datasets, and
fine-tuning scripts.

## Tests

Run the suite:

```bash
python -m pytest
```

Run the focused semantic orchestration tests:

```bash
python -m pytest tests/test_semantic_token_limits.py -q
```

## Development Notes

- Add CLI flags in `main.py`, then thread them through the relevant pipeline
  layer rather than reading new environment variables deep in utility code.
- Add semantic repair backends by implementing `FragmentFixer` or
  `generate_candidates` in `pipeline/code_object_repair_loop.py`.
- Keep semantic edits localized. The acceptance loop assumes a candidate can be
  reattached, compiled, and scored as a small source change.
- When changing semantic prompt shape, check consumers that reconstruct or mine
  prompts, especially `utils/rebuild_semantic_prompt_refresh_dataset.py` and
  `utils/mine_semantic_repair_actions.py`.
- Avoid mutating original decompiled inputs. Repair attempts should stay inside
  run-specific output directories.
