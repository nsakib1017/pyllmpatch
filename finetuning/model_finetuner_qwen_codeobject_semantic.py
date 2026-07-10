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
DEFAULT_RESPONSE_TEMPLATE = "<|im_start|>assistant"


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
        print(
            "[completion-only-loss] --assistant-only-loss is OFF; training with "
            "FULL-SEQUENCE loss (loss on prompt + completion)."
        )
        return None
    if not args.response_template:
        warnings.warn(
            "--assistant-only-loss was requested without --response-template; falling back to full-sequence loss.",
            RuntimeWarning,
            stacklevel=2,
        )
        return None

    # trl 0.24.0 removed DataCollatorForCompletionOnlyLM, so we use a
    # self-contained, version-independent completion-only collator instead.
    # Robust to both run-as-module (finetuning.completion_masking) and
    # run-as-script (python finetuning/model_finetuner_...py -> completion_masking on sys.path).
    try:
        from finetuning.completion_masking import build_completion_only_collator
    except ModuleNotFoundError:
        from completion_masking import build_completion_only_collator

    collator = build_completion_only_collator(tokenizer, args.response_template)
    if collator is None:
        warnings.warn(
            f"Response template {args.response_template!r} tokenized to no ids; "
            "falling back to full-sequence loss.",
            RuntimeWarning,
            stacklevel=2,
        )
        return None
    print(
        "[completion-only-loss] enabled: masking prompt tokens through response "
        f"template {args.response_template!r} "
        f"({len(collator.response_template_ids)} token ids); loss on completion only."
    )
    return collator


def create_training_arguments(training_arguments_cls: Any, args: argparse.Namespace, output_dir: Path, *, bf16_supported: bool) -> Any:
    kwargs = {
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "warmup_steps": args.warmup_steps,
        "warmup_ratio": args.warmup_ratio,
        "num_train_epochs": args.num_train_epochs,
        # GB10 unified-memory crash guard: the GPU shares the 121 GB system RAM, so a
        # transient allocation spike is a *system* OOM that hangs the whole box (kernel
        # freeze -> reboot), not a recoverable CUDA OOM. The step-100 in-loop eval (full
        # forward at max_seq_length materializing a seq x vocab logits tensor per example)
        # is what wedged the machine on 2026-07-08, before any checkpoint could be saved.
        # Disable in-loop eval entirely (quality is judged by the end-to-end hybrid-100
        # bake-off, not eval_loss) and save early/often so a durable checkpoint lands fast.
        "eval_strategy": args.eval_strategy,
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
        # load_best_model_at_end requires eval; keep it only when eval is on.
        "load_best_model_at_end": args.eval_strategy != "no",
        "metric_for_best_model": "eval_loss",
        "greater_is_better": False,
        "save_total_limit": args.save_total_limit,
        "report_to": args.report_to,
        "group_by_length": args.group_by_length,
        # CRITICAL: trl 0.24 renamed SFTTrainer's max_seq_length -> SFTConfig.max_length
        # (default 1024). Passing max_seq_length= to SFTTrainer is silently DROPPED, so every
        # prior run truncated to 1024 tokens and lost the answer for ~90% of examples. Set
        # max_length here so long brief-ON prompts + their GT-source answer survive. Also keep
        # max_seq_length for any older trl; the signature filter below keeps whichever exists.
        "max_length": args.max_seq_length,
        "max_seq_length": args.max_seq_length,
        "packing": args.packing,
        "dataset_num_proc": args.dataset_num_proc,
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

    from transformers import EarlyStoppingCallback
    from trl import SFTConfig, SFTTrainer
    from unsloth import is_bfloat16_supported
    bf16_supported = is_bfloat16_supported()
    # Use SFTConfig (a superset of TrainingArguments) so max_length / packing actually take
    # effect in trl 0.24 — TrainingArguments has no max_length, so the value was being dropped.
    training_arguments_cls = SFTConfig

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
        # EarlyStoppingCallback asserts eval_strategy != "no" in on_train_begin (transformers
        # 5.3.0), so it MUST be omitted when in-loop eval is disabled (the GB10 unified-memory
        # crash guard) — otherwise trainer.train() AssertionErrors after the full model load.
        "callbacks": (
            [EarlyStoppingCallback(
                early_stopping_patience=args.early_stopping_patience,
                early_stopping_threshold=1e-3,
            )]
            if args.eval_strategy != "no"
            else []
        ),
        # max_length / packing / dataset_num_proc now live in the SFTConfig (args) so they
        # actually take effect in trl 0.24; do not also pass them as direct SFTTrainer kwargs.
        "args": create_training_arguments(
            training_arguments_cls,
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
    # Default eval OFF: in-loop eval spikes unified memory and hard-crashes the GB10 (see
    # create_training_arguments). Pass --eval-strategy steps to re-enable at your own risk.
    parser.add_argument("--eval-strategy", choices=["no", "steps", "epoch"], default="no")
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
    parser.add_argument("--run-dir", default=None,
                        help="Reuse this exact run dir and resume from its latest checkpoint (teardown-safe).")
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


def tokenized_text_length(tokenizer: Any, text: str) -> int:
    try:
        encoded = tokenizer(text, add_special_tokens=False, truncation=False)
    except TypeError:
        encoded = tokenizer(text, add_special_tokens=False)
    if isinstance(encoded, dict):
        input_ids = encoded.get("input_ids", [])
    elif hasattr(encoded, "input_ids"):
        input_ids = encoded.input_ids
    else:
        input_ids = encoded
    if hasattr(input_ids, "shape") and getattr(input_ids, "shape", None):
        return int(input_ids.shape[-1])
    if hasattr(input_ids, "numel"):
        return int(input_ids.numel())
    if input_ids and isinstance(input_ids[0], list):
        return max(len(row) for row in input_ids)
    return len(input_ids)


def sample_token_lengths(
    sample: dict[str, Any],
    tokenizer: Any,
    *,
    enable_thinking_template: bool,
) -> list[int]:
    texts = formatting_func(
        sample,
        tokenizer,
        enable_thinking_template=enable_thinking_template,
    )
    return [tokenized_text_length(tokenizer, text) for text in texts]


def filter_samples_by_token_budget(
    samples: list[dict[str, Any]],
    tokenizer: Any,
    *,
    max_seq_length: int,
    enable_thinking_template: bool,
) -> list[dict[str, Any]]:
    if max_seq_length <= 0:
        return samples
    kept: list[dict[str, Any]] = []
    skipped = 0
    max_seen = 0
    for sample in samples:
        lengths = sample_token_lengths(
            sample,
            tokenizer,
            enable_thinking_template=enable_thinking_template,
        )
        sample_max = max(lengths, default=0)
        max_seen = max(max_seen, sample_max)
        if sample_max <= max_seq_length:
            kept.append(sample)
        else:
            skipped += 1
    if skipped:
        print(
            f"Skipped {skipped} examples over --max-seq-length={max_seq_length} "
            f"after chat templating; max observed length was {max_seen} tokens."
        )
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

    model, tokenizer = load_model_and_tokenizer(args)
    samples = filter_samples_by_token_budget(
        samples,
        tokenizer,
        max_seq_length=args.max_seq_length,
        enable_thinking_template=args.enable_thinking_template,
    )
    if len(samples) < 2:
        raise ValueError("Need at least two valid samples within --max-seq-length for train/validation split.")

    train_samples, val_samples = split_samples(
        samples,
        validation_ratio=args.validation_ratio,
        seed=args.seed,
    )
    train_dataset = Dataset.from_list(train_samples)
    val_dataset = Dataset.from_list(val_samples)

    model_slug = args.model_name.strip("/").replace("/", "__")
    if getattr(args, "run_dir", None):
        output_dir = Path(args.run_dir)
        run_id = output_dir.name.replace("run_", "")
    else:
        run_id = int(time.time())
        output_dir = repo_root() / "finetuning" / "finetuned_models" / model_slug / f"run_{run_id}"
    merged_dir = repo_root() / "finetuning" / "merged_models" / model_slug / f"run_{run_id}"
    output_dir.mkdir(parents=True, exist_ok=True)
    # Teardown-safe resume: if the run dir already holds checkpoints, continue from the latest.
    _ckpts = sorted(output_dir.glob("checkpoint-*"),
                    key=lambda p: int(p.name.split("-")[-1]) if p.name.split("-")[-1].isdigit() else -1)
    resume_ckpt = str(_ckpts[-1]) if _ckpts else None
    if resume_ckpt:
        print("RESUMING from checkpoint:", resume_ckpt)

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

    trainer_stats = trainer.train(resume_from_checkpoint=resume_ckpt)
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
