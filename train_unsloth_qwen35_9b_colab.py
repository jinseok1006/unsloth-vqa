# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Unsloth Qwen3.5-9B VQA Colab
#
# Colab A100 환경에서 빠르게 smoke test 또는 간단한 1 epoch 학습을 돌리기 위한 스크립트입니다.
#
# 전제:
# - 데이터는 상위 저장소 기준 자산과 같은 형태를 사용합니다.
# - `train.csv`, `test.csv`, `sample_submission.csv`, `train/`, `test/`가 한 루트 아래에 있어야 합니다.
# - 출력은 `sample_submission.csv` 형식을 그대로 따릅니다.

# %% [markdown]
# ## Optional Colab Setup
#
# 필요하면 먼저 Drive를 마운트합니다.
#
# ```python
# from google.colab import drive
# drive.mount('/content/drive')
# ```
#
# 런타임은 A100 + High-RAM 기준을 권장합니다.

# %%
from __future__ import annotations

import os
import random
import re
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from datasets import Dataset as HFDataset
from unsloth import FastVisionModel
from unsloth.trainer import UnslothVisionDataCollator
from trl import SFTConfig, SFTTrainer

Image.MAX_IMAGE_PIXELS = None


def validate_runtime() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA GPU is required for the Colab Unsloth workflow. "
            f"Detected torch build: {torch.__version__}."
        )


validate_runtime()


# %% [markdown]
# ## Config

# %%
EXPERIMENT_ID = "E_UNSLOTH_QWEN35_9B_COLAB_V1"
BASE_RUN_NAME = "none"
EXPERIMENT_PURPOSE = "Run simple Qwen3.5-9B VQA fine-tuning on Colab A100"
CHANGED_FIELDS = "platform=colab,framework=unsloth,model=Qwen3.5-9B"

MODEL_ID = "Qwen/Qwen3.5-9B"
IMAGE_SIZE = 384
MAX_NEW_TOKENS = 2
MAX_SEQ_LENGTH = 1024
SEED = 42

# Quick smoke test defaults for Colab.
USE_SUBSAMPLE = True
SUBSAMPLE_SIZE = 200

NUM_EPOCHS = 1
LR = 1e-4
GRAD_ACCUM = 4
BATCH_SIZE = 1
WARMUP_RATIO = 0.03

LOAD_IN_16BIT = True
LOAD_IN_4BIT = False
FULL_FINETUNING = False

LORA_R = 16
LORA_ALPHA = 16
LORA_DROPOUT = 0.0

USE_GRADIENT_CHECKPOINTING = "unsloth"
FINETUNE_VISION_LAYERS = True
FINETUNE_LANGUAGE_LAYERS = True
FINETUNE_ATTENTION_MODULES = True
FINETUNE_MLP_MODULES = True

device = "cuda"
print("Device:", device)
print("Torch:", torch.__version__)
print("GPU:", torch.cuda.get_device_name(0))

random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)


# %% [markdown]
# ## Paths

# %%
DATA_ROOT = Path(
    os.environ.get(
        "UNSLOTH_QWEN35_COLAB_DATA_ROOT",
        "/content",
    )
).resolve()
OUTPUT_ROOT = Path(
    os.environ.get(
        "UNSLOTH_QWEN35_COLAB_OUTPUT_ROOT",
        "/content/outputs/unsloth_qwen35",
    )
).resolve()

TRAIN_CSV = DATA_ROOT / "train.csv"
TEST_CSV = DATA_ROOT / "test.csv"
SAMPLE_SUBMISSION_CSV = DATA_ROOT / "sample_submission.csv"

MODEL_TAG = MODEL_ID.split("/")[-1].lower().replace(".", "_").replace("-", "_")
RUN_NAME = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{MODEL_TAG}_colab_ep{NUM_EPOCHS}_lr{LR:g}".replace(
    ".", "_"
)
RUN_DIR = OUTPUT_ROOT / "runs" / RUN_NAME
VALID_DIR = RUN_DIR / "valid"
SAVE_DIR = RUN_DIR / "checkpoints" / "final"
PREDICTIONS_DIR = RUN_DIR / "predictions"

CONFIG_PATH = RUN_DIR / "config.txt"
TRAIN_LOG_PATH = RUN_DIR / "train_log.txt"
SUMMARY_PATH = RUN_DIR / "summary.txt"
SUBMISSION_PATH = PREDICTIONS_DIR / "submission.csv"

VALID_TXT_PATH = VALID_DIR / "epoch_01_valid.txt"
VALID_CSV_PATH = VALID_DIR / "epoch_01_valid.csv"
VALID_TYPE_CSV_PATH = VALID_DIR / "epoch_01_question_type_accuracy.csv"
VALID_ERRORS_CSV_PATH = VALID_DIR / "epoch_01_errors.csv"

VALID_DIR.mkdir(parents=True, exist_ok=True)
SAVE_DIR.mkdir(parents=True, exist_ok=True)
PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)


def validate_paths() -> None:
    required_paths = [
        TRAIN_CSV,
        TEST_CSV,
        SAMPLE_SUBMISSION_CSV,
        DATA_ROOT / "train",
        DATA_ROOT / "test",
    ]
    missing_paths = [str(path) for path in required_paths if not path.exists()]
    if missing_paths:
        raise FileNotFoundError(
            "Required dataset assets were not found. Missing paths:\n- "
            + "\n- ".join(missing_paths)
        )


validate_paths()


# %% [markdown]
# ## Helpers

# %%
QUESTION_TYPE_PATTERNS = {
    "count": r"몇 개|개수|총 몇|총 몇 개|몇 개인가요|몇 개입니까",
    "color": r"색깔|색상|무슨 색|어느 색",
    "material": r"재질|소재",
    "location": r"어디|위치|어느 곳|놓여",
    "classification": r"분류|종류|무엇인가요|무엇입니까",
}

SYSTEM_INSTRUCT = (
    "You are a helpful visual question answering assistant. "
    "Answer using exactly one lowercase letter among a, b, c, or d. No explanation."
)


def classify_question_type(question: str) -> str:
    text = str(question)
    for label, pattern in QUESTION_TYPE_PATTERNS.items():
        if re.search(pattern, text):
            return label
    return "other"


def build_mc_prompt(question: str, a: str, b: str, c: str, d: str) -> str:
    return (
        f"{question}\n"
        f"(a) {a}\n(b) {b}\n(c) {c}\n(d) {d}\n\n"
        "정답을 반드시 a, b, c, d 중 하나의 소문자 한 글자로만 출력하세요."
    )


def extract_choice(text: str) -> str:
    text = str(text).strip().lower()
    if not text:
        return "a"

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if lines and lines[-1] in ["a", "b", "c", "d"]:
        return lines[-1]

    direct_match = re.search(r"\b([a-d])\b", text)
    if direct_match:
        return direct_match.group(1)

    paren_match = re.search(r"\(([a-d])\)", text)
    if paren_match:
        return paren_match.group(1)

    return "a"


def convert_row_to_messages(row: pd.Series, train: bool = True) -> dict[str, Any]:
    image = Image.open(DATA_ROOT / row["path"]).convert("RGB")
    user_text = build_mc_prompt(row["question"], row["a"], row["b"], row["c"], row["d"])
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": SYSTEM_INSTRUCT + "\n\n" + user_text},
                {"type": "image", "image": image},
            ],
        }
    ]
    if train:
        messages.append(
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": str(row["answer"]).strip().lower()}
                ],
            }
        )
    return {"messages": messages}


class VisionRecordDataset(Dataset):
    def __init__(self, records: list[dict[str, Any]]):
        self.records = records

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.records[index]


def generate_choice(
    model, processor, row: pd.Series, max_new_tokens: int = MAX_NEW_TOKENS
):
    image = Image.open(DATA_ROOT / row["path"]).convert("RGB")
    user_text = build_mc_prompt(row["question"], row["a"], row["b"], row["c"], row["d"])
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": SYSTEM_INSTRUCT + "\n\n" + user_text},
            ],
        }
    ]
    input_text = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(
        image,
        input_text,
        add_special_tokens=False,
        return_tensors="pt",
    ).to(device)

    with torch.no_grad():
        out_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            use_cache=True,
            do_sample=False,
        )

    output_text = processor.batch_decode(out_ids, skip_special_tokens=True)[0]
    return extract_choice(output_text), output_text


def save_config(train_size: int, valid_size: int) -> None:
    with CONFIG_PATH.open("w", encoding="utf-8") as f:
        f.write(f"run_name: {RUN_NAME}\n")
        f.write(f"experiment_id: {EXPERIMENT_ID}\n")
        f.write(f"base_run_name: {BASE_RUN_NAME}\n")
        f.write(f"experiment_purpose: {EXPERIMENT_PURPOSE}\n")
        f.write(f"changed_fields: {CHANGED_FIELDS}\n")
        f.write(f"model_id: {MODEL_ID}\n")
        f.write(f"image_size: {IMAGE_SIZE}\n")
        f.write(f"max_new_tokens: {MAX_NEW_TOKENS}\n")
        f.write(f"max_seq_length: {MAX_SEQ_LENGTH}\n")
        f.write(f"num_epochs: {NUM_EPOCHS}\n")
        f.write(f"learning_rate: {LR}\n")
        f.write(f"grad_accum: {GRAD_ACCUM}\n")
        f.write(f"batch_size: {BATCH_SIZE}\n")
        f.write(f"warmup_ratio: {WARMUP_RATIO}\n")
        f.write(f"seed: {SEED}\n")
        f.write(f"use_subsample: {USE_SUBSAMPLE}\n")
        f.write(f"subsample_size: {SUBSAMPLE_SIZE if USE_SUBSAMPLE else 'full'}\n")
        f.write(f"load_in_16bit: {LOAD_IN_16BIT}\n")
        f.write(f"load_in_4bit: {LOAD_IN_4BIT}\n")
        f.write(f"full_finetuning: {FULL_FINETUNING}\n")
        f.write(f"lora_r: {LORA_R}\n")
        f.write(f"lora_alpha: {LORA_ALPHA}\n")
        f.write(f"lora_dropout: {LORA_DROPOUT}\n")
        f.write(f"train_size: {train_size}\n")
        f.write(f"valid_size: {valid_size}\n")
        f.write(f"train_csv_path: {TRAIN_CSV}\n")
        f.write(f"test_csv_path: {TEST_CSV}\n")
        f.write(f"sample_submission_csv_path: {SAMPLE_SUBMISSION_CSV}\n")


def save_train_log(train_loss: float, valid_loss: float, valid_accuracy: float) -> None:
    with TRAIN_LOG_PATH.open("w", encoding="utf-8") as f:
        f.write(f"run_name: {RUN_NAME}\n")
        f.write("epoch\ttrain_loss\tvalid_loss\tvalid_accuracy\n")
        f.write(f"1\t{train_loss:.4f}\t{valid_loss:.4f}\t{valid_accuracy:.4f}\n")


def save_summary(
    valid_accuracy: float, type_acc_df: pd.DataFrame, errors_df: pd.DataFrame
) -> None:
    observations = [
        f"final_valid_accuracy: {valid_accuracy:.4f}",
        f"best_question_type: {type_acc_df.sort_values('accuracy', ascending=False).iloc[0]['question_type']}"
        if not type_acc_df.empty
        else "best_question_type: none",
        f"worst_question_type: {type_acc_df.sort_values('accuracy', ascending=True).iloc[0]['question_type']}"
        if not type_acc_df.empty
        else "worst_question_type: none",
    ]

    failure_patterns = []
    if not errors_df.empty:
        error_counts = errors_df["question_type"].value_counts().head(2)
        for question_type, count in error_counts.items():
            failure_patterns.append(f"{question_type}: {count} errors")
    if not failure_patterns:
        failure_patterns = [
            "no error pattern captured",
            "validation set too small for pattern analysis",
        ]

    lines = [
        f"run_name: {RUN_NAME}",
        f"experiment_id: {EXPERIMENT_ID}",
        f"experiment_purpose: {EXPERIMENT_PURPOSE}",
        f"base_run_name: {BASE_RUN_NAME}",
        f"changed_fields: {CHANGED_FIELDS}",
        f"final_valid_accuracy: {valid_accuracy:.4f}",
        f"best_epoch: {NUM_EPOCHS}",
        "question_type_accuracy:",
    ]
    for _, row in type_acc_df.iterrows():
        lines.append(
            f"- {row['question_type']}: {int(row['correct'])}/{int(row['total'])} ({row['accuracy']:.4f})"
        )
    lines.extend(
        [
            "핵심 관찰:",
            *[f"- {item}" for item in observations],
            "대표 실패 패턴:",
            *[f"- {item}" for item in failure_patterns],
            "이번 설정 유지 여부: pending",
            "다음 액션 제안: Compare against HF baseline and inspect classification/count performance",
        ]
    )
    SUMMARY_PATH.write_text("\n".join(lines), encoding="utf-8")


# %% [markdown]
# ## Load Data

# %%
train_df = pd.read_csv(TRAIN_CSV)
test_df = pd.read_csv(TEST_CSV)

if USE_SUBSAMPLE:
    train_df = train_df.sample(
        n=min(SUBSAMPLE_SIZE, len(train_df)), random_state=SEED
    ).reset_index(drop=True)

split_index = int(len(train_df) * 0.9)
train_subset = train_df.iloc[:split_index].reset_index(drop=True)
valid_subset = train_df.iloc[split_index:].reset_index(drop=True)

print(f"Data root: {DATA_ROOT}")
print(f"Output root: {OUTPUT_ROOT}")
print(f"Train rows used: {len(train_df):,}")
print(f"Train subset: {len(train_subset):,}")
print(f"Valid subset: {len(valid_subset):,}")
print(f"Test rows: {len(test_df):,}")

save_config(train_size=len(train_subset), valid_size=len(valid_subset))


# %% [markdown]
# ## Model

# %%
model, processor = FastVisionModel.from_pretrained(
    model_name=MODEL_ID,
    max_seq_length=MAX_SEQ_LENGTH,
    load_in_4bit=LOAD_IN_4BIT,
    load_in_16bit=LOAD_IN_16BIT,
    full_finetuning=FULL_FINETUNING,
)

model = FastVisionModel.get_peft_model(
    model,
    finetune_vision_layers=FINETUNE_VISION_LAYERS,
    finetune_language_layers=FINETUNE_LANGUAGE_LAYERS,
    finetune_attention_modules=FINETUNE_ATTENTION_MODULES,
    finetune_mlp_modules=FINETUNE_MLP_MODULES,
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    lora_dropout=LORA_DROPOUT,
    bias="none",
    random_state=SEED,
    use_gradient_checkpointing=USE_GRADIENT_CHECKPOINTING,
    target_modules="all-linear",
    modules_to_save=["lm_head", "embed_tokens"],
)


# %% [markdown]
# ## Dataset

# %%
train_records = [
    convert_row_to_messages(row, train=True) for _, row in train_subset.iterrows()
]
valid_records = [
    convert_row_to_messages(row, train=True) for _, row in valid_subset.iterrows()
]

train_dataset = HFDataset.from_list(train_records)
valid_dataset = VisionRecordDataset(valid_records)

data_collator = UnslothVisionDataCollator(
    model,
    processor,
    max_seq_length=MAX_SEQ_LENGTH,
    resize="min",
    train_on_responses_only=False,
    completion_only_loss=True,
)


# %% [markdown]
# ## Train

# %%
trainer = SFTTrainer(
    model=model,
    tokenizer=processor,
    data_collator=data_collator,
    train_dataset=train_dataset,
    args=SFTConfig(
        output_dir=str(RUN_DIR),
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=LR,
        warmup_ratio=WARMUP_RATIO,
        logging_steps=1,
        save_strategy="no",
        eval_strategy="no",
        optim="adamw_8bit",
        seed=SEED,
        max_seq_length=MAX_SEQ_LENGTH,
        dataset_num_proc=1,
        report_to=[],
        remove_unused_columns=False,
    ),
)

train_result = trainer.train()
train_loss = float(train_result.training_loss)


# %% [markdown]
# ## Validation Loss

# %%
valid_loader = DataLoader(
    valid_dataset,
    batch_size=1,
    shuffle=False,
    collate_fn=data_collator,
    num_workers=0,
)

model.eval()
FastVisionModel.for_inference(model)

valid_loss_sum = 0.0
valid_steps = 0
with torch.no_grad():
    for batch in tqdm(valid_loader, desc="Valid loss", unit="batch"):
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        valid_loss_sum += float(outputs.loss.item())
        valid_steps += 1

valid_loss = valid_loss_sum / valid_steps if valid_steps else 0.0


# %% [markdown]
# ## Validation Generation

# %%
valid_predictions = []
type_correct = defaultdict(int)
type_total = defaultdict(int)
correct = 0

for _, row in tqdm(
    valid_subset.iterrows(),
    total=len(valid_subset),
    desc="Valid generate",
    unit="sample",
):
    pred, raw_text = generate_choice(model, processor, row)
    answer = str(row["answer"]).strip().lower()
    question_type = classify_question_type(row["question"])
    is_correct = pred == answer

    correct += int(is_correct)
    type_total[question_type] += 1
    type_correct[question_type] += int(is_correct)

    valid_predictions.append(
        {
            "id": row["id"],
            "question_type": question_type,
            "question": row["question"],
            "pred": pred,
            "answer": answer,
            "correct": is_correct,
            "raw_text": raw_text.replace("\n", "\\n"),
        }
    )

valid_accuracy = correct / len(valid_predictions) if valid_predictions else 0.0
type_acc_rows = []
for question_type in sorted(type_total):
    type_acc_rows.append(
        {
            "question_type": question_type,
            "correct": type_correct[question_type],
            "total": type_total[question_type],
            "accuracy": round(
                type_correct[question_type] / type_total[question_type], 4
            ),
        }
    )

valid_df = pd.DataFrame(valid_predictions)
type_acc_df = pd.DataFrame(type_acc_rows)
errors_df = valid_df.loc[~valid_df["correct"]].reset_index(drop=True)

valid_df.to_csv(VALID_CSV_PATH, index=False, encoding="utf-8-sig")
type_acc_df.to_csv(VALID_TYPE_CSV_PATH, index=False, encoding="utf-8-sig")
errors_df.to_csv(VALID_ERRORS_CSV_PATH, index=False, encoding="utf-8-sig")

with VALID_TXT_PATH.open("w", encoding="utf-8") as f:
    f.write(f"epoch: {NUM_EPOCHS}\n")
    f.write(f"train_loss: {train_loss:.4f}\n")
    f.write(f"valid_size: {len(valid_predictions)}\n")
    f.write(f"valid_loss: {valid_loss:.4f}\n")
    f.write(f"valid_accuracy: {valid_accuracy:.4f}\n\n")
    f.write("question_type_accuracy\n")
    for item in type_acc_rows:
        f.write(
            f"{item['question_type']}: {item['correct']}/{item['total']} ({item['accuracy']:.4f})\n"
        )

save_train_log(
    train_loss=train_loss,
    valid_loss=valid_loss,
    valid_accuracy=valid_accuracy,
)


# %% [markdown]
# ## Save Checkpoint

# %%
model.save_pretrained(str(SAVE_DIR))
processor.save_pretrained(str(SAVE_DIR))
print("Saved checkpoint:", SAVE_DIR)


# %% [markdown]
# ## Test Inference

# %%
preds = []
sample_prediction_lines = []
for idx, row in tqdm(
    test_df.iterrows(), total=len(test_df), desc="Inference", unit="sample"
):
    pred, raw_text = generate_choice(model, processor, row)
    preds.append(pred)
    if idx < 20:
        sample_prediction_lines.append(
            f"id={row['id']} pred={pred} raw_text={raw_text.replace(chr(10), '\\n')}"
        )

submission = pd.read_csv(SAMPLE_SUBMISSION_CSV).copy()
submission["answer"] = preds
submission.to_csv(SUBMISSION_PATH, index=False, encoding="utf-8-sig")
(PREDICTIONS_DIR / "sample_predictions.txt").write_text(
    "\n".join(sample_prediction_lines), encoding="utf-8"
)


# %% [markdown]
# ## Summary

# %%
save_summary(
    valid_accuracy=valid_accuracy,
    type_acc_df=type_acc_df,
    errors_df=errors_df,
)

print("Run directory:", RUN_DIR)
print("Valid txt:", VALID_TXT_PATH)
print("Valid csv:", VALID_CSV_PATH)
print("Question type accuracy csv:", VALID_TYPE_CSV_PATH)
print("Summary:", SUMMARY_PATH)
print("Submission:", SUBMISSION_PATH)
