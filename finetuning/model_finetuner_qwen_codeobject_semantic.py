from __future__ import annotations

import argparse
import gc
import inspect
import json
import os
import random
import time
import warnings
from pathlib import Path
from pprint import pprint
from typing import Any

import torch
from dotenv import load_dotenv


DEFAULT_MODEL_NAME = "/home/mxs220189/pylingual_collaboration/pylingual_download/code/finetuning/finetuned_models/Qwen3-Coder-30B-A3B-Instruct-Up/run_1778647681"
DEFAULT_DATASET_PATH = "dataset/accepted_codeobject_mining_prompt_refresh.jsonl"
DEFAULT_MAX_SEQ_LENGTH = 16384
DEFAULT_VALIDATION_RATIO = 0.05
DEFAULT_RANDOM_SEED = 42
DEFAULT_RESPONSE_TEMPLATE = ""


def env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def env_text(name: str, default: str) -> str:
    value = os.getenv(name)
    if value is None:
        return default
    return value.encode("utf-8").decode("unicode_escape")


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def resolve_path(path_value: str | os.PathLike[str]) -> Path:
    path = Path(path_value).expanduser()
    if path.is_absolute():
        return path
    return repo_root() / path


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on {path}:{line_no}: {exc}") from exc
    return records


def build_chat_samples(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []
    skipped = 0

    for record in records:
        prompt_record = record.get("prompt") or {}
        messages = prompt_record.get("messages")
        assistant_text = record.get("replacement_text")

        if not isinstance(messages, list) or len(messages) < 2 or not assistant_text:
            skipped += 1
            continue

        prompt_messages = [
            {
                "role": str(message.get("role", "")).strip(),
                "content": str(message.get("content", "")),
            }
            for message in messages
            if isinstance(message, dict) and message.get("role") and message.get("content")
        ]

        if len(prompt_messages) < 2:
            skipped += 1
            continue

        prompt_messages.append(
            {
                "role": "assistant",
                "content": str(assistant_text).strip("\n"),
            }
        )
        samples.append({"messages": prompt_messages})

    print(f"Built {len(samples)} chat samples; skipped {skipped} malformed records.")
    return samples


def apply_chat_template(
    tokenizer: Any,
    messages: list[dict[str, Any]],
    *,
    enable_thinking_template: bool,
) -> str:
    kwargs = {
        "tokenize": False,
        "add_generation_prompt": False,
        "enable_thinking": enable_thinking_template,
    }
    try:
        return tokenizer.apply_chat_template(messages, **kwargs)
    except TypeError:
        # Older/non-Qwen tokenizers do not expose enable_thinking.
        kwargs.pop("enable_thinking", None)
        return tokenizer.apply_chat_template(messages, **kwargs)


def formatting_func(example: dict[str, Any], tokenizer: Any, enable_thinking_template: bool) -> list[str]:
    messages_field = example["messages"]
    if (
        isinstance(messages_field, list)
        and messages_field
        and isinstance(messages_field[0], list)
    ):
        conversations = messages_field
    else:
        conversations = [messages_field]

    return [
        apply_chat_template(
            tokenizer,
            messages,
            enable_thinking_template=enable_thinking_template,
        )
        for messages in conversations
    ]


def print_sample(samples: list[dict[str, Any]], max_chars: int = 2000) -> None:
    if not samples:
        print("No samples available.")
        return

    messages = samples[0]["messages"]
    print("First sample roles:", [message["role"] for message in messages])
    print("First user prompt preview:")
    print(messages[1]["content"][:max_chars])
    print("First assistant output preview:")
    print(messages[-1]["content"][:max_chars])


def split_samples(
    samples: list[dict[str, Any]],
    *,
    validation_ratio: float,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not 0 < validation_ratio < 1:
        raise ValueError("--validation-ratio must be between 0 and 1.")

    shuffled = list(samples)
    random.Random(seed).shuffle(shuffled)
    val_size = max(1, int(round(len(shuffled) * validation_ratio)))
    val_size = min(val_size, len(shuffled) - 1)
    return shuffled[val_size:], shuffled[:val_size]


def load_model_and_tokenizer(args: argparse.Namespace):
    from unsloth import FastLanguageModel

    hf_token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")
    auth_kwargs = {"token": hf_token} if hf_token else {}

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model_name,
        max_seq_length=args.max_seq_length,
        dtype=None,
        load_in_4bit=args.load_in_4bit,
        device_map={"": 0},
        **auth_kwargs,
    )

    if hasattr(model, "peft_config") and model.peft_config:
        print("Loaded existing LoRA checkpoint.")
    else:
        model = FastLanguageModel.get_peft_model(
            model,
            r=args.lora_r,
            target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            bias="none",
            use_gradient_checkpointing="unsloth",
            random_state=args.seed,
            use_rslora=args.use_rslora,
            loftq_config=None,
        )

    if getattr(tokenizer, "pad_token_id", None) is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    model.config.use_cache = False

    print("Model device:", model.device)
    return model, tokenizer


def create_completion_data_collator(tokenizer: Any, args: argparse.Namespace) -> Any | None:
    if not args.assistant_only_loss:
        return None
    if not args.response_template:
        warnings.warn(
            "--assistant-only-loss was requested without --response-template; falling back to full-sequence loss.",
            RuntimeWarning,
            stacklevel=2,
        )
        return None
    try:
        from trl import DataCollatorForCompletionOnlyLM
    except (ImportError, AttributeError):
        warnings.warn(
            "TRL DataCollatorForCompletionOnlyLM is unavailable; falling back to full-sequence loss.",
            RuntimeWarning,
            stacklevel=2,
        )
        return None
    return DataCollatorForCompletionOnlyLM(
        response_template=args.response_template,
        tokenizer=tokenizer,
    )


def create_training_arguments(training_arguments_cls: Any, args: argparse.Namespace, output_dir: Path, *, bf16_supported: bool) -> Any:
    kwargs = {
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "warmup_steps": args.warmup_steps,
        "warmup_ratio": args.warmup_ratio,
        "num_train_epochs": args.num_train_epochs,
        "eval_strategy": "steps",
        "eval_steps": args.eval_steps,
        "save_strategy": "steps",
        "save_steps": args.save_steps,
        "learning_rate": args.learning_rate,
        "fp16": not bf16_supported,
        "bf16": bf16_supported,
        "optim": args.optim,
        "weight_decay": args.weight_decay,
        "max_grad_norm": args.max_grad_norm,
        "lr_scheduler_type": args.lr_scheduler_type,
        "seed": args.seed,
        "output_dir": str(output_dir),
        "logging_strategy": "steps",
        "logging_steps": args.logging_steps,
        "logging_first_step": True,
        "load_best_model_at_end": True,
        "metric_for_best_model": "eval_loss",
        "greater_is_better": False,
        "save_total_limit": args.save_total_limit,
        "report_to": args.report_to,
        "group_by_length": args.group_by_length,
    }

    supported = set(inspect.signature(training_arguments_cls.__init__).parameters)
    if "eval_strategy" not in supported and "evaluation_strategy" in supported:
        kwargs["evaluation_strategy"] = kwargs.pop("eval_strategy")
    filtered = {key: value for key, value in kwargs.items() if key in supported}
    dropped = sorted(set(kwargs) - set(filtered))
    if dropped:
        warnings.warn(
            f"TrainingArguments does not support these options in this environment; dropping: {', '.join(dropped)}",
            RuntimeWarning,
            stacklevel=2,
        )
    return training_arguments_cls(**filtered)


def create_trainer(
    *,
    model: Any,
    tokenizer: Any,
    train_dataset: Any,
    val_dataset: Any,
    output_dir: Path,
    args: argparse.Namespace,
):
    from functools import partial

    from transformers import EarlyStoppingCallback, TrainingArguments
    from trl import SFTTrainer
    from unsloth import is_bfloat16_supported
    bf16_supported = is_bfloat16_supported()

    trainer_kwargs = {
        "model": model,
        "tokenizer": tokenizer,
        "train_dataset": train_dataset,
        "eval_dataset": val_dataset,
        "formatting_func": partial(
            formatting_func,
            tokenizer=tokenizer,
            enable_thinking_template=args.enable_thinking_template,
        ),
        "max_seq_length": args.max_seq_length,
        "dataset_num_proc": args.dataset_num_proc,
        "packing": args.packing,
        "callbacks": [
            EarlyStoppingCallback(
                early_stopping_patience=args.early_stopping_patience,
                early_stopping_threshold=1e-3,
            )
        ],
        "args": create_training_arguments(
            TrainingArguments,
            args,
            output_dir,
            bf16_supported=bf16_supported,
        ),
    }
    data_collator = create_completion_data_collator(tokenizer, args)
    if data_collator is not None:
        trainer_kwargs["data_collator"] = data_collator
    return SFTTrainer(**trainer_kwargs)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fine-tune Qwen3-Coder on accepted semantic code-object repair "
            "prompt/output examples."
        )
    )
    parser.add_argument("--model-name", default=os.getenv("MODEL_NAME_FOR_FINETUNING", DEFAULT_MODEL_NAME))
    parser.add_argument(
        "--dataset-path",
        default=os.getenv("DATASET_PATH_FOR_FINETUNING", DEFAULT_DATASET_PATH),
    )
    parser.add_argument("--max-seq-length", type=int, default=int(os.getenv("MAX_SEQ_LENGTH_FOR_FINETUNING", DEFAULT_MAX_SEQ_LENGTH)))
    parser.add_argument("--validation-ratio", type=float, default=float(os.getenv("VALIDATION_RATIO_FOR_FINETUNING", DEFAULT_VALIDATION_RATIO)))
    parser.add_argument("--seed", type=int, default=DEFAULT_RANDOM_SEED)
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--load-in-4bit", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dry-run", action="store_true", help="Only validate and preview the JSONL examples.")
    parser.add_argument(
        "--max-sample-chars",
        type=int,
        default=int(os.getenv("MAX_SAMPLE_CHARS_FOR_FINETUNING", DEFAULT_MAX_SEQ_LENGTH * 4)),
        help="Drop extreme prompt+answer examples by character count before tokenization; 0 disables this heuristic.",
    )

    parser.add_argument("--per-device-train-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=16)
    parser.add_argument("--num-train-epochs", type=float, default=3)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--warmup-steps", type=int, default=0)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--eval-steps", type=int, default=100)
    parser.add_argument("--save-steps", type=int, default=100)
    parser.add_argument("--logging-steps", type=int, default=50)
    parser.add_argument("--save-total-limit", type=int, default=3)
    parser.add_argument("--early-stopping-patience", type=int, default=4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--max-grad-norm", type=float, default=0.3)
    parser.add_argument("--lr-scheduler-type", default="cosine")
    parser.add_argument("--optim", default="adamw_8bit")
    parser.add_argument("--report-to", default="none")
    parser.add_argument("--dataset-num-proc", type=int, default=1)
    parser.add_argument("--packing", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--group-by-length", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--assistant-only-loss", action=argparse.BooleanOptionalAction, default=env_bool("ASSISTANT_ONLY_LOSS_FOR_FINETUNING", False))
    parser.add_argument("--response-template", default=env_text("RESPONSE_TEMPLATE_FOR_FINETUNING", DEFAULT_RESPONSE_TEMPLATE))
    parser.add_argument("--enable-thinking-template", action=argparse.BooleanOptionalAction, default=env_bool("ENABLE_THINKING_TEMPLATE_FOR_FINETUNING", False))
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.0)
    parser.add_argument("--use-rslora", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--merge", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def filter_samples_by_char_budget(samples: list[dict[str, Any]], max_sample_chars: int) -> list[dict[str, Any]]:
    if max_sample_chars <= 0:
        return samples
    kept: list[dict[str, Any]] = []
    skipped = 0
    for sample in samples:
        total_chars = sum(len(str(message.get("content", ""))) for message in sample.get("messages", []))
        if total_chars <= max_sample_chars:
            kept.append(sample)
        else:
            skipped += 1
    if skipped:
        print(f"Skipped {skipped} examples over --max-sample-chars={max_sample_chars}.")
    return kept


def main() -> None:
    load_dotenv()
    args = parse_args()

    dataset_path = resolve_path(args.dataset_path)
    print("CUDA available:", torch.cuda.is_available())
    print("Dataset:", dataset_path)
    print("Model:", args.model_name)

    records = read_jsonl(dataset_path)
    samples = build_chat_samples(records)
    samples = filter_samples_by_char_budget(samples, args.max_sample_chars)
    if len(samples) < 2:
        raise ValueError("Need at least two valid samples for train/validation split.")
    if args.assistant_only_loss and args.packing:
        raise ValueError("--assistant-only-loss requires --no-packing so prompt/assistant masking stays valid.")

    print_sample(samples)
    if args.dry_run:
        return

    from datasets import Dataset

    train_samples, val_samples = split_samples(
        samples,
        validation_ratio=args.validation_ratio,
        seed=args.seed,
    )
    train_dataset = Dataset.from_list(train_samples)
    val_dataset = Dataset.from_list(val_samples)

    model, tokenizer = load_model_and_tokenizer(args)

    run_id = int(time.time())
    model_slug = args.model_name.strip("/").replace("/", "__")
    output_dir = repo_root() / "finetuning" / "finetuned_models" / model_slug / f"run_{run_id}"
    merged_dir = repo_root() / "finetuning" / "merged_models" / model_slug / f"run_{run_id}"
    output_dir.mkdir(parents=True, exist_ok=True)

    trainer = create_trainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        output_dir=output_dir,
        args=args,
    )

    torch.cuda.empty_cache()
    gc.collect()

    trainer_stats = trainer.train()
    pprint(trainer_stats)

    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(output_dir)
    print("Final adapter saved to:", output_dir)

    if args.merge:
        merged_dir.mkdir(parents=True, exist_ok=True)
        try:
            if hasattr(model, "save_pretrained_merged"):
                model.save_pretrained_merged(
                    str(merged_dir),
                    tokenizer,
                    save_method="merged_16bit",
                )
            else:
                model = model.merge_and_unload()
                model.save_pretrained(merged_dir)
                tokenizer.save_pretrained(merged_dir)
        except NotImplementedError as exc:
            raise RuntimeError(
                "The adapter was saved successfully, but saving the merged model failed. "
                "This can happen when merging a LoRA adapter into a 4-bit/bnb Qwen model "
                "with the current Transformers weight-conversion path. Re-run with "
                "`--no-merge` to keep only the adapter, or load the saved adapter into a "
                "non-quantized base model and merge from that."
            ) from exc
        print("Merged model saved to:", merged_dir)

    print("Adapter/checkpoint output saved to:", output_dir)


if __name__ == "__main__":
    main()
