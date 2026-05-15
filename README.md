# pyllmpatch

Utilities for repairing Python decompiler output with syntactic and semantic
repair pipelines.

## Setup

Create and activate a Python virtual environment, then install the project
dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Create a local environment file from the example:

```bash
cp .env.example .env
```

At minimum, set these values in `.env` before running the dataset-backed
pipelines:

```dotenv
PROJECT_ROOT_DIR=/absolute/path/to/this/repo
ROOT_FOR_FILES=/absolute/path/to/decompiled/file/root
BASE_DIR_PYTHON_FILES_PYLINGUAL=pylingual
BASE_DIR_PYTHON_FILES_PYPI=pypi
BASE_DATASET_NAME=your_dataset.csv
MAX_EXAMPLE_RUNTIME_MIN=1
```

If you run LLM-backed syntactic repair, also set the relevant provider key:

```dotenv
OPENAI_API_KEY=
DEEPSEEK_API_KEY=
GEMINI_API_KEY=
```

## Running

The default command runs the syntactic repair experiment:

```bash
python main.py
```

This is equivalent to:

```bash
python main.py syntactic-repair
```

Useful syntactic repair environment options:

```dotenv
USE_LOCAL_LLM=True
LOCAL_LLM_IDX=0
NO_OF_MAX_RETRIES=0
CONFIG_IDX_START=0
CONFIG_IDX_RANGE=1
MAX_WHOLE_FILE_BYTES=1048576
DELETE_ONLY_MODE=false
ENABLE_DELETE_ONLY_FALLBACK=true
```

To run without LLM calls, enable delete-only mode:

```bash
DELETE_ONLY_MODE=true python main.py syntactic-repair
```

Results are written under:

```text
results/experiment_outputs/<timestamp>/<run_id>/
```

## Semantic Repair

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

Run semantic repair over `semantic_error` rows in the configured dataset:

```bash
python main.py semantic-repair --dataset-mode --limit 10
python main.py semantic-repair --dataset-mode --row-range 10:20
```

Optional semantic repair flags:

```bash
--dataset-path /path/to/dataset.csv
--row-range 10:20
--file-hash <hash>
--skip-pylingual-verification
--skip-step-verification
--keep-non-improving
--sample-timeout-seconds 3600
--sample-timeout-patience-seconds 900
--sample-timeout-min-improvement-delta 1
--output-dir /path/to/output_dir
--json-out /path/to/result.json
```

Dataset-mode semantic repair checks each sample after 3600 seconds by default.
If no combined-distance improvement has been made when the timeout is reached,
the sample is skipped. If the combined distance has improved recently, the loop
gets a bounded patience window and then keeps the best partial result seen so
far. Override with `--sample-timeout-seconds`,
`--sample-timeout-patience-seconds`, `--sample-timeout-min-improvement-delta`,
or the matching `SEMANTIC_REPAIR_*` environment variables; use
`--sample-timeout-seconds 0` to disable the timeout.

For all CLI options:

```bash
python main.py --help
python main.py semantic-repair --help
```
