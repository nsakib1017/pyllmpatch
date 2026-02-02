import os
import time
import gc
from pathlib import Path
from pprint import pprint

import torch
import pandas as pd
import numpy as np

from dotenv import load_dotenv
from tqdm import tqdm
from datasets import Dataset
from sklearn.model_selection import train_test_split

from unsloth import FastLanguageModel, is_bfloat16_supported

from trl import SFTTrainer
from transformers import TrainingArguments, EarlyStoppingCallback
from functools import partial
from unsloth.chat_templates import get_chat_template
# ===============================
# Environment & Globals
# ===============================

def setup_environment():
    load_dotenv()
    print("CUDA available:", torch.cuda.is_available())


#unsloth/phi-4-reasoning
MODEL_NAME = "unsloth/codellama-34b-bnb-4bit"
MAX_SEQ_LENGTH = 16384
LOAD_IN_4BIT = True
DTYPE = None

VALIDATION_RATIO = 0.1
RANDOM_SEED = 42


SYSTEM_PROMPT = (
    "You are an expert Python programmer and syntax repair specialist.\n"
    "Your task is to fix Python SYNTAX errors only.\n\n"

    "Make ONLY the minimal changes strictly required for the code to parse and compile successfully.\n"
    "Do NOT fix logic errors, runtime errors, or improve code quality unless doing so is absolutely necessary to resolve a syntax error.\n"
    "Do NOT refactor, reformat, rename symbols, or modify any code that is already syntactically valid.\n\n"

    "The provided error message is for reference only and may point to the wrong line. "
    "You must inspect surrounding lines and earlier code to identify the true root cause of any syntax error.\n\n"

    "If there are orphaned 'break', 'continue', or similar statements that cause invalid syntax, "
    "you may add the smallest possible syntactic wrapper (such as a dummy loop) solely to make the code syntactically valid.\n"

    "If missing or incomplete try/except blocks cause syntax errors, "
    "you may add the minimal required structure (including 'pass' where necessary) to restore syntactic correctness.\n\n"

    "If missing or incomplete match statements cause syntax errors, "
    "you may add the minimal required structure a default case (i.e., case _:, including 'pass' where necessary) to restore syntactic correctness.\n\n"

    "You must also fix any OTHER syntax errors present in the snippet beyond the one mentioned in the error message, "
    "as long as they can be resolved with minimal changes.\n\n"

    "Preserve all error-free lines exactly as they appear in the original code, including indentation.\n"
    "The final output must be a syntactically valid Python program.\n\n"

    "Output ONLY the corrected Python code.\n"
    "Do NOT include explanations, reasoning, comments, markdown, or formatting wrappers of any kind.\n"
    "Do NOT add any text before or after the code."
)

USER_PROMPT_TEMPLATE = (
    "Analyze the Python code snippet below and fix all syntax errors.\n\n"
    "Initial error message:\n"
    "{error_message}\n\n"
    "Initial code snippet:\n"
    "{code_snippet}\n\n"
    "If the code contains no syntax errors or the errors cannot be fixed, return the code unchanged."
)


# ===============================
# Model Setup
# ===============================

def load_model_and_tokenizer():
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=DTYPE,
        load_in_4bit=LOAD_IN_4BIT,
        device_map={"": "cuda:0"},
    )
    # model.to("cuda")

    tokenizer = get_chat_template(tokenizer,chat_template = "phi-4",)


    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_alpha=16,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=3407,
        use_rslora=False,
        loftq_config=None,
    )

    tokenizer.add_eos_token = True
    model.config.use_cache = False

    print("Model device:", model.device)
    return model, tokenizer


# ===============================
# Dataset Utilities
# ===============================

def read_csv_file(file_name: str) -> pd.DataFrame:
    csv_path = Path.cwd() / file_name
    print("Loading dataset:", csv_path)
    return pd.read_csv(csv_path)


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    df = df.dropna()
    df = df[df["old_code_full"].str.strip() != ""]
    df = df[df["new_code_full"].str.strip() != ""]
    return df


def build_chat_messages(
    code_snippet: str,
    error_message: str,
    system_prompt: str,
    user_prompt_template: str,
) -> list[dict]:
    user_prompt = user_prompt_template.format(
        error_message=error_message,
        code_snippet=code_snippet,
    )

    return [
        {"role": "system", "content": system_prompt.strip()},
        {"role": "user", "content": user_prompt.strip()},
    ]


def dataframe_to_chat_samples(
    df: pd.DataFrame,
    system_prompt: str,
    user_prompt_template: str,
    error_col: str = "syntactic_error_description",
) -> list[dict]:
    samples = []

    for _, row in df.iterrows():
        messages = build_chat_messages(
            code_snippet=row["old_code_full"].strip("\n"),
            error_message=row[error_col],
            system_prompt=system_prompt,
            user_prompt_template=user_prompt_template,
        )

        messages.append({
            "role": "assistant",
            "content": row["new_code_full"].strip("\n"),
        })

        samples.append({"messages": messages})

    return samples


def formatting_func(example, tokenizer):
    texts = []
    messages_field = example["messages"]

    if isinstance(messages_field, list) and messages_field and isinstance(messages_field[0], list):
        conversations = messages_field
    else:
        conversations = [messages_field]

    for messages in conversations:
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )
        texts.append(text)

    return texts


# ===============================
# Training
# ===============================

def create_trainer(
    model,
    tokenizer,
    train_dataset,
    val_dataset,
    output_dir,
):
    

    format_fn = partial(formatting_func, tokenizer=tokenizer)
    
    return SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        formatting_func=format_fn,
        max_seq_length=MAX_SEQ_LENGTH,
        dataset_num_proc=2,
        packing=False,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3, early_stopping_threshold=1e-3)],
        args=TrainingArguments(
            per_device_train_batch_size=4,
            gradient_accumulation_steps=4,
            warmup_steps=5,
            num_train_epochs=15,
            eval_strategy="steps",
            eval_steps=100,
            save_strategy="steps",
            save_steps=100,
            learning_rate=2e-4,
            fp16=not is_bfloat16_supported(),
            bf16=is_bfloat16_supported(),
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="linear",
            seed=3407,
            output_dir=output_dir,
            logging_strategy="steps",
            logging_steps=50,
            logging_first_step=True,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            save_total_limit=2,
            report_to="none",
        ),
    )


# ===============================
# Main
# ===============================

def main():
    setup_environment()
    model, tokenizer = load_model_and_tokenizer()

    # tokenizer = get_chat_template(tokenizer, chat_template = "gemma-3",)

    df = read_csv_file(f"{os.getenv('PROJECT_ROOT_DIR')}/dataset/{os.getenv('DATASET_PATH_FOR_FINETUNING_NAME')}")
    df = clean_dataset(df)

    train_df, val_df = train_test_split(
        df,
        test_size=VALIDATION_RATIO,
        random_state=RANDOM_SEED,
        shuffle=True,
    )

    train_samples = dataframe_to_chat_samples(
        train_df,
        system_prompt=SYSTEM_PROMPT,
        user_prompt_template=USER_PROMPT_TEMPLATE,
    )

    val_samples = dataframe_to_chat_samples(
        val_df,
        system_prompt=SYSTEM_PROMPT,
        user_prompt_template=USER_PROMPT_TEMPLATE,
    )

    train_dataset = Dataset.from_list(train_samples)
    val_dataset = Dataset.from_list(val_samples)
    run_id = int(time.time())
    output_dir = f"finetuned_models/{MODEL_NAME}/run_{run_id}"
    os.makedirs(output_dir, exist_ok=True)

    trainer = create_trainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        output_dir=output_dir,
    )

    torch.cuda.empty_cache()
    gc.collect()

    trainer_stats = trainer.train()
    pprint(trainer_stats)

    model = model.merge_and_unload()
    model.save_pretrained(f"merged_models/{MODEL_NAME}/run_{run_id}")
    tokenizer.save_pretrained(f"merged_models/{MODEL_NAME}/run_{run_id}")


if __name__ == "__main__":
    main()
