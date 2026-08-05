# pyllmpatch

Repairs Python decompiler output so it can be analysed:

- **Syntactic repair** — make uncompilable decompiled Python compile again.
- **Semantic repair** — reduce bytecode drift between a ground-truth `.pyc` and a
  derived `.pyc` by editing localized source fragments.

## Setup

Requires Python 3.12, plus CPython 3.10–3.15 available through
[`uv`](https://docs.astral.sh/uv/) or `pyenv` — the pipeline compiles candidate
fixes against each file's own bytecode version.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env          # then fill in the values below
```

A `pylingual` checkout must be importable from the project root: the distance
metrics import `pylingual.*` for the equivalence oracle.

### Minimum `.env`

| Variable | Meaning |
| --- | --- |
| `PROJECT_ROOT_DIR` | Absolute path to this checkout |
| `BASE_DATASET_NAME` | Dataset CSV filename; resolved as `dataset/<name>` |
| `ROOT_FOR_FILES` | Root under which the decompiled inputs live |
| `BASE_DIR_PYTHON_FILES_PYLINGUAL`, `BASE_DIR_PYTHON_FILES_PYPI` | Input subdirectories, relative to `ROOT_FOR_FILES` |
| `NO_OF_MAX_RETRIES` | LLM attempts per file in syntactic repair |
| `USE_LOCAL_LLM` | `True` to use a local model instead of an API |
| `LOCAL_LLM_IDX` | Index into `OPEN_LLM_MODELS` in `utils/providers.py` |

For a local vLLM server also set `SEMANTIC_VLLM_SERVER_URL`
(e.g. `http://127.0.0.1:8001/v1`) and `SEMANTIC_VLLM_SERVER_MODEL`. Otherwise set
the matching API key (`GEMINI_API_KEY`, `OPENAI_API_KEY`, `DEEPSEEK_API_KEY`).

## Running

Everything runs through `main.py`:

```bash
python main.py --help
```

### Syntactic repair

Runs over the dataset CSV, repairing files that do not compile:

```bash
python main.py syntactic-repair --limit 50
python main.py syntactic-repair --source VirusTotal
```

Writes one JSONL record per file to
`results/experiment_outputs/<timestamp>/<run_id>/run_log_*.jsonl`.

### Semantic repair — single file

```bash
python main.py semantic-repair GT.pyc DERIVED.pyc DERIVED.py --output-dir /tmp/repair
```

`--fixer` selects the backend and defaults to `llm`; `--fixer none` runs the
deterministic operators only, without a model. Add `--json-out result.json` to
save the full record.

The ground-truth `.pyc` is the repair target and the scoring oracle. Its
**source** is never read: repairs are generated from the derived source and the
GT *bytecode*, never from GT text.

### Semantic repair — dataset

```bash
python main.py semantic-repair --dataset-mode --limit 100
```

Writes a per-row `result.json` plus result/deferred CSVs under
`results/experiment_outputs/`.

## Useful flags

| Flag / variable | Effect |
| --- | --- |
| `--max-iterations N` | Repair passes per file (default 1) |
| `--sample-timeout-seconds N` | Per-file wall-clock cap |
| `SEMANTIC_DETERMINISTIC_OPERATORS=1` | Deterministic pre-pass before the LLM (default on) |
| `SEMANTIC_POST_LLM_DETERMINISTIC=1` | Re-run operators on the residual after each LLM pass |
| `SEMANTIC_ACCEPTANCE_MODE=oracle` | Oracle-gated acceptance (required by the post-pass) |
| `ENABLE_FORCED_COMPILE_FALLBACK=1` | Neutralise unparseable code instead of deleting it (default on) |

## Layout

```text
main.py       CLI entry point
pipeline/     dataset runners, config, LLM dispatch, logging
model/        local model loading and inference
utils/        repair core: mapping, reattachment, distance metrics, operators
```
