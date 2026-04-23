from __future__ import annotations

import os
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def current_run_timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def flag(name: str, default: str = "") -> bool:
    return str(os.getenv(name, default)).strip().lower() in {"1", "true", "yes", "y", "on"}


PROJECT_ROOT_DIR = Path(str(os.getenv("PROJECT_ROOT_DIR")))
ROOT_FOR_FILES = Path(str(os.getenv("ROOT_FOR_FILES")))
BASE_DIR_PYTHON_FILES_PYLINGUAL = ROOT_FOR_FILES / str(os.getenv("BASE_DIR_PYTHON_FILES_PYLINGUAL"))
BASE_DIR_PYTHON_FILES_PYPI = ROOT_FOR_FILES / str(os.getenv("BASE_DIR_PYTHON_FILES_PYPI"))
BASE_DATASET_NAME = str(os.getenv("BASE_DATASET_NAME"))
BASE_DATASET_PATH = PROJECT_ROOT_DIR / "dataset" / BASE_DATASET_NAME
MAX_EXAMPLE_RUNTIME_SEC = int(float(os.getenv("MAX_EXAMPLE_RUNTIME_MIN")) * 60 * 60)


@dataclass(frozen=True)
class RuntimeConfig:
    previous_run_log_path: Path | None
    delete_only_mode: bool
    enable_delete_only_fallback: bool
    use_local_llm: bool
    config_idx_start: int
    config_idx_range: int
    delete_only_infinite_iters: bool
    delete_only_max_iters: int
    delete_only_base_window: int
    delete_only_max_deleted_ratio: float
    max_whole_file_bytes: int
    max_retries_default: int
    local_llm_idx: int
    run_timestamp: str


def load_runtime_config() -> RuntimeConfig:
    previous_run_log_path = Path(str(os.getenv("PREVIOUS_RUN_LOG_PATH"))) if os.getenv("PREVIOUS_RUN_LOG_PATH") else None
    delete_only_infinite_iters = flag("DELETE_ONLY_INFINITE_ITERS")

    local_llm_idx_raw = os.getenv("LOCAL_LLM_IDX")
    try:
        local_llm_idx = int(local_llm_idx_raw) if local_llm_idx_raw is not None else 0
    except ValueError:
        local_llm_idx = 0

    return RuntimeConfig(
        previous_run_log_path=previous_run_log_path,
        delete_only_mode=flag("DELETE_ONLY_MODE"),
        enable_delete_only_fallback=flag("ENABLE_DELETE_ONLY_FALLBACK", "1"),
        use_local_llm=flag("USE_LOCAL_LLM", "True"),
        config_idx_start=int(os.getenv("CONFIG_IDX_START", 0)),
        config_idx_range=int(os.getenv("CONFIG_IDX_RANGE", 1)),
        delete_only_infinite_iters=delete_only_infinite_iters,
        delete_only_max_iters=(10**9) if delete_only_infinite_iters else int(os.getenv("DELETE_ONLY_MAX_ITERS", "5000")),
        delete_only_base_window=int(os.getenv("DELETE_ONLY_BASE_WINDOW", "1")),
        delete_only_max_deleted_ratio=float(os.getenv("DELETE_ONLY_MAX_DELETED_RATIO", "0.95")),
        max_whole_file_bytes=int(os.getenv("MAX_WHOLE_FILE_BYTES", str(1 * 1024 * 1024))),
        max_retries_default=int(os.getenv("NO_OF_MAX_RETRIES", "0")),
        local_llm_idx=local_llm_idx,
        run_timestamp=current_run_timestamp(),
    )


def build_run_paths(run_timestamp: str) -> tuple[str, Path, Path]:
    run_id = uuid.uuid4().hex
    log_base = Path(f"results/experiment_outputs/{run_timestamp}/{run_id}")
    log_base.mkdir(parents=True, exist_ok=True)
    log_file = log_base / f"run_log_{run_id}_{BASE_DATASET_PATH.name.split('.')[0]}.jsonl"
    return run_id, log_base, log_file
