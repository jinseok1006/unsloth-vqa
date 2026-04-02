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
# # VQA Baseline Colab Unsloth
#
# Colab Pro+ 환경에서 VQA 멀티모달 모델을 학습하기 위한 baseline 노트북입니다.
#
# 기준:
# - 런타임: `T4`
# - Python: `3.11`
# - 환경 준비: 기존 `(260324)_baseline_colab.ipynb` 흐름 사용
# - 학습 구조: `train_unsloth_qwen35_9b_colab.py` 기반
# - Unsloth 연결 방식: `DAY13_효율적인_Fine_tuning_PEFT_sol.ipynb` 참고

# %% [markdown]
# ## Runtime Checklist
#
# 아래 조건을 먼저 맞춘 뒤 실행합니다.
#
# - `런타임 > 런타임 유형 변경`
# - `GPU` 선택
# - `T4` 선택
# - 먼저 smoke test로 정상 동작 확인

# %% [markdown]
# ## Environment Setup
#
# 설치 셀 실행 후 런타임 재시작이 필요할 수 있습니다.

# %%
# %%capture
# !pip install -U pip
# !pip install unsloth
# !pip install "transformers>=4.52.0" "accelerate>=1.7.0" "trl>=0.19.0" "datasets>=3.6.0" "peft>=0.13.2" "bitsandbytes>=0.46.0" pillow pandas tqdm

# %%
import os
import shutil
from pathlib import Path

import torch

print("Python executable:", os.sys.executable)
print("Torch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))

# %% [markdown]
# ## Mount Drive
#
# 데이터 원본 zip과 결과 저장 경로는 Drive에 둡니다.

# %%
from google.colab import drive

drive.mount("/content/drive")

# %% [markdown]
# ## Paths And Data Layout
#
# 권장 Drive 구조:
#
# ```text
# /content/drive/MyDrive/
#   colab.zip
#
# /content/drive/MyDrive/vqa/
#   outputs/
#
# /content/vqa_data/
#   train.csv
#   test.csv
#   sample_submission.csv
#   train/
#   test/
# ```

# %%
DRIVE_MOUNT_ROOT = Path("/content/drive/MyDrive")
DRIVE_ROOT = DRIVE_MOUNT_ROOT / "vqa"
DRIVE_OUTPUT_ROOT = DRIVE_ROOT / "outputs"

LOCAL_DATA_ROOT = Path("/content/vqa_data")
LOCAL_PREPROCESSED_ROOT = Path("/content/preprocessed")
LOCAL_OUTPUT_ROOT = Path("/content/outputs/unsloth_qwen35")

COLAB_ZIP_NAME = "colab.zip"

TRAIN_CSV_NAME = "train.csv"
TEST_CSV_NAME = "test.csv"
SAMPLE_SUBMISSION_NAME = "sample_submission.csv"

STAGE_TO_LOCAL = True

DRIVE_OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
LOCAL_OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

os.environ["UNSLOTH_QWEN35_COLAB_DATA_ROOT"] = str(LOCAL_DATA_ROOT)
os.environ["UNSLOTH_QWEN35_COLAB_OUTPUT_ROOT"] = str(LOCAL_OUTPUT_ROOT)
os.environ["HF_HOME"] = "/content/.cache/huggingface"
os.environ["TRANSFORMERS_CACHE"] = "/content/.cache/huggingface/transformers"
os.environ["HF_DATASETS_CACHE"] = "/content/.cache/huggingface/datasets"

print("Drive mount root:", DRIVE_MOUNT_ROOT)
print("Drive project root:", DRIVE_ROOT)
print("Expected zip location priority: /content/drive/MyDrive/colab.zip")
print("Local data root:", LOCAL_DATA_ROOT)
print("Local preprocessed root:", LOCAL_PREPROCESSED_ROOT)
print("Local output root:", LOCAL_OUTPUT_ROOT)

# %% [markdown]
# ## Extract Dataset In Colab Local Storage
#
# zip은 Drive에서 찾기만 하고, 실제 압축 해제는 `/content`에서 진행합니다.

# %%
import zipfile


def unzip_if_needed(zip_path: Path, target_dir: Path) -> None:
    if not zip_path.exists():
        print(f"Skip unzip, file not found: {zip_path}")
        return

    sentinel_paths = [
        target_dir / TRAIN_CSV_NAME,
        target_dir / TEST_CSV_NAME,
        target_dir / "train",
        target_dir / "test",
    ]
    if all(path.exists() for path in sentinel_paths):
        print(f"Dataset already available under: {target_dir}")
        return

    target_dir.mkdir(parents=True, exist_ok=True)
    print(f"Extracting {zip_path.name} -> {target_dir}")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(target_dir)


def find_file_by_name(search_root: Path, file_name: str) -> Path:
    direct_path = search_root / file_name
    if direct_path.exists():
        return direct_path

    matches = sorted(search_root.rglob(file_name))
    if not matches:
        raise FileNotFoundError(
            f"Could not locate {file_name} under {search_root}. "
            "Check the Drive upload location or update DRIVE_ROOT manually."
        )
    if len(matches) > 1:
        print("Multiple zip candidates found. Using:", matches[0])
    return matches[0]


def find_dataset_root(search_root: Path) -> Path:
    required_names = {
        TRAIN_CSV_NAME,
        TEST_CSV_NAME,
        SAMPLE_SUBMISSION_NAME,
        "train",
        "test",
    }

    if all((search_root / name).exists() for name in required_names):
        return search_root

    for path in sorted(search_root.rglob(TRAIN_CSV_NAME)):
        candidate = path.parent
        if all((candidate / name).exists() for name in required_names):
            return candidate

    raise FileNotFoundError(
        "Could not locate dataset root containing train.csv, test.csv, sample_submission.csv, train/, test/. "
        f"Check extracted contents under {search_root}."
    )


COLAB_ZIP_PATH = find_file_by_name(DRIVE_MOUNT_ROOT, COLAB_ZIP_NAME)
print("Resolved zip path:", COLAB_ZIP_PATH)
unzip_if_needed(COLAB_ZIP_PATH, LOCAL_DATA_ROOT)
RESOLVED_LOCAL_DATA_ROOT = find_dataset_root(LOCAL_DATA_ROOT)
print("Resolved local dataset root:", RESOLVED_LOCAL_DATA_ROOT)

# %% [markdown]
# ## Prepare Active Dataset Root
#
# 학습은 `/content` 로컬 경로를 기준으로 진행합니다.


# %%
if not STAGE_TO_LOCAL:
    raise RuntimeError(
        "This notebook is configured to use local /content storage for extracted data."
    )

os.environ["UNSLOTH_QWEN35_COLAB_DATA_ROOT"] = str(RESOLVED_LOCAL_DATA_ROOT)

print("Active data root:", os.environ["UNSLOTH_QWEN35_COLAB_DATA_ROOT"])

# %% [markdown]
# ## Training Imports

# %%
import random
import re
from collections import defaultdict
from datetime import datetime
from typing import Any

import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from PIL import ImageOps
from datasets import Dataset as HFDataset
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from unsloth import FastVisionModel
from unsloth.trainer import UnslothVisionDataCollator
from trl import SFTConfig, SFTTrainer

Image.MAX_IMAGE_PIXELS = None


def validate_runtime() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required for this Colab workflow.")


validate_runtime()
print("Using GPU:", torch.cuda.get_device_name(0))

# %% [markdown]
# ## User Controls
#
# 아래 값들만 바꾸면 주요 실험 설정을 제어할 수 있습니다.
# 초보자는 이 셀만 수정하고, 아래쪽 상세 구현은 건드리지 않는 것을 권장합니다.

# %%
# type: str
USER_MODEL_PROFILE = "qwen35_2b_unsloth"
# 선택 가능:
# - "qwen35_2b_unsloth": 반복 실험용 저비용 프로파일
# - "qwen35_9b_unsloth": A100 권장 대형 프로파일
# - "legacy_qwen3_vl_2b": 기존 Qwen3-VL-2B baseline

# type: str
USER_TRAIN_MODE = "smoke"
# 선택 가능:
# - "smoke": 작은 샘플로 빠르게 학습 + validation 수행
# - "train_valid_split": train 전체를 train/valid로 나눠서 학습 + validation 수행
# - "train_all_for_submission": train 전체로 학습하고 validation 생략 후 full inference 수행

# type: int
USER_SUBSAMPLE_SIZE = 200
# smoke 모드에서만 사용합니다.

# type: int | None
USER_IMAGE_SIZE = None
# None이면 모델 프로파일 기본값을 사용합니다.

# type: float
USER_CENTER_CROP_RATIO = 0.92
# type: bool
USER_BUILD_PREPROCESSED_CACHE = True
# type: int
USER_PREPROCESS_PREVIEW_SAMPLES = 6

# type: float
USER_LR = 1e-4
# type: int
USER_NUM_EPOCHS = 1
# type: int | None
USER_BATCH_SIZE = None
# type: int | None
USER_GRAD_ACCUM = None
# None이면 모델 프로파일 기본값을 사용합니다.

# trainer checkpoint 저장 설정
# type: bool
USER_SAVE_CHECKPOINTS_EACH_EPOCH = False
# True면 epoch가 끝날 때마다 trainer checkpoint를 저장합니다.

# type: int | None
USER_SAVE_TOTAL_LIMIT = 2
# 저장할 checkpoint 개수 제한입니다. None이면 제한 없이 저장합니다.

# type: bool
USER_SAVE_FINAL_MODEL = True
# True면 학습 종료 후 최종 모델/processor를 final 폴더에 저장합니다.

# %% [markdown]
# ## Config

# %%
EXPERIMENT_ID = "E_BASELINE_UNSLOTH_QWEN35_9B_COLAB_V1"
BASE_RUN_NAME = "baseline_colab"
EXPERIMENT_PURPOSE = "Profile-based VQA fine-tuning on Colab with Unsloth vision models"
CHANGED_FIELDS = "platform=colab,framework=unsloth,model=profile_selected"

MODEL_PROFILES = {
    "qwen35_2b_unsloth": {
        "model_id": "unsloth/Qwen3.5-2B",
        "load_in_16bit": False,
        "load_in_4bit": True,
        "full_finetuning": False,
        "image_size": 384,
        "max_seq_length": 1024,
        "batch_size": 1,
        "grad_accum": 8,
        "run_full_inference": False,
    },
    "qwen35_9b_unsloth": {
        "model_id": "unsloth/Qwen3.5-9B",
        "load_in_16bit": True,
        "load_in_4bit": False,
        "full_finetuning": False,
        "image_size": 384,
        "max_seq_length": 1024,
        "batch_size": 1,
        "grad_accum": 4,
        "run_full_inference": True,
    },
    "legacy_qwen3_vl_2b": {
        "model_id": "unsloth/Qwen3-VL-2B-Instruct-unsloth-bnb-4bit",
        "load_in_16bit": False,
        "load_in_4bit": True,
        "full_finetuning": False,
        "image_size": 384,
        "max_seq_length": 1024,
        "batch_size": 1,
        "grad_accum": 8,
        "run_full_inference": False,
    },
}

MODEL_PROFILE = USER_MODEL_PROFILE
if MODEL_PROFILE not in MODEL_PROFILES:
    raise ValueError(
        f"Unsupported USER_MODEL_PROFILE: {MODEL_PROFILE}. "
        f"Choose one of: {', '.join(sorted(MODEL_PROFILES))}"
    )

selected_profile = MODEL_PROFILES[MODEL_PROFILE]
TRAIN_MODE = USER_TRAIN_MODE
if TRAIN_MODE not in ["smoke", "train_valid_split", "train_all_for_submission"]:
    raise ValueError(
        f"Unsupported USER_TRAIN_MODE: {TRAIN_MODE}. "
        "Choose one of: smoke, train_valid_split, train_all_for_submission"
    )

MODEL_ID = selected_profile["model_id"]
IMAGE_SIZE = (
    selected_profile["image_size"] if USER_IMAGE_SIZE is None else USER_IMAGE_SIZE
)
MAX_NEW_TOKENS = 2
MAX_SEQ_LENGTH = selected_profile["max_seq_length"]
SEED = 42

PREPROCESS_IMAGES = True
PREPROCESS_MODE = "keep_ratio_pad"
BUILD_PREPROCESSED_CACHE = USER_BUILD_PREPROCESSED_CACHE
PREPROCESS_PREVIEW_SAMPLES = USER_PREPROCESS_PREVIEW_SAMPLES
CENTER_CROP_RATIO = USER_CENTER_CROP_RATIO

USE_SUBSAMPLE = TRAIN_MODE == "smoke"
SUBSAMPLE_SIZE = USER_SUBSAMPLE_SIZE

NUM_EPOCHS = USER_NUM_EPOCHS
LR = USER_LR
GRAD_ACCUM = (
    selected_profile["grad_accum"] if USER_GRAD_ACCUM is None else USER_GRAD_ACCUM
)
BATCH_SIZE = (
    selected_profile["batch_size"] if USER_BATCH_SIZE is None else USER_BATCH_SIZE
)
WARMUP_RATIO = 0.03
RUN_FULL_INFERENCE = TRAIN_MODE == "train_all_for_submission"
SAVE_CHECKPOINTS_EACH_EPOCH = USER_SAVE_CHECKPOINTS_EACH_EPOCH
SAVE_TOTAL_LIMIT = USER_SAVE_TOTAL_LIMIT
SAVE_FINAL_MODEL = USER_SAVE_FINAL_MODEL
SAVE_STRATEGY = "epoch" if SAVE_CHECKPOINTS_EACH_EPOCH else "no"

LOAD_IN_16BIT = selected_profile["load_in_16bit"]
LOAD_IN_4BIT = selected_profile["load_in_4bit"]
FULL_FINETUNING = selected_profile["full_finetuning"]

LORA_R = 16
LORA_ALPHA = 16
LORA_DROPOUT = 0.0

USE_GRADIENT_CHECKPOINTING = "unsloth"
FINETUNE_VISION_LAYERS = True
FINETUNE_LANGUAGE_LAYERS = True
FINETUNE_ATTENTION_MODULES = True
FINETUNE_MLP_MODULES = True

random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

print("Model profile:", MODEL_PROFILE)
print("Model id:", MODEL_ID)
print("Train mode:", TRAIN_MODE)
print("Use subsample:", USE_SUBSAMPLE)
print("Subsample size:", SUBSAMPLE_SIZE if USE_SUBSAMPLE else "not_used")
print("Num epochs:", NUM_EPOCHS)
print("Image size:", IMAGE_SIZE)
print("Center crop ratio:", CENTER_CROP_RATIO)
print("Load in 16bit:", LOAD_IN_16BIT)
print("Load in 4bit:", LOAD_IN_4BIT)
print("Run full inference:", RUN_FULL_INFERENCE)
print("Save checkpoints each epoch:", SAVE_CHECKPOINTS_EACH_EPOCH)
print("Save total limit:", SAVE_TOTAL_LIMIT)
print("Save final model:", SAVE_FINAL_MODEL)

# %% [markdown]
# ## Training Paths

# %%
DATA_ROOT = Path(os.environ["UNSLOTH_QWEN35_COLAB_DATA_ROOT"]).resolve()
OUTPUT_ROOT = Path(os.environ["UNSLOTH_QWEN35_COLAB_OUTPUT_ROOT"]).resolve()

TRAIN_CSV = DATA_ROOT / TRAIN_CSV_NAME
TEST_CSV = DATA_ROOT / TEST_CSV_NAME
SAMPLE_SUBMISSION_CSV = DATA_ROOT / SAMPLE_SUBMISSION_NAME

MODEL_TAG = MODEL_ID.split("/")[-1].lower().replace(".", "_").replace("-", "_")
RUN_NAME = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{MODEL_TAG}_colab_ep{NUM_EPOCHS}_lr{LR:g}".replace(
    ".",
    "_",
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
VALID_CONCEPT_CSV_PATH = VALID_DIR / "epoch_01_concept_accuracy.csv"
VALID_ERRORS_CSV_PATH = VALID_DIR / "epoch_01_errors.csv"
VALID_SAMPLES_TXT_PATH = VALID_DIR / "epoch_01_samples.txt"

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
print("Data root:", DATA_ROOT)
print("Output root:", OUTPUT_ROOT)

# %% [markdown]
# ## Image Preprocessing
#
# 원본 압축 해제는 한 번만 유지하고, 전처리는 설정별 캐시를 재사용합니다.

# %%
PREPROCESS_CACHE_DIR = LOCAL_PREPROCESSED_ROOT / f"{PREPROCESS_MODE}_{IMAGE_SIZE}"
PREPROCESS_CACHE_DIR.mkdir(parents=True, exist_ok=True)


def resolve_image_path(raw_path: str, root: Path | None = None) -> Path:
    search_root = DATA_ROOT if root is None else root
    normalized = str(raw_path).replace("\\", "/").lstrip("/")
    candidate = search_root / normalized
    if candidate.exists():
        return candidate

    fallback = search_root / Path(normalized).name
    if fallback.exists():
        return fallback

    return candidate


def preprocess_image(image: Image.Image) -> Image.Image:
    image = image.convert("RGB")

    if 0.0 < CENTER_CROP_RATIO < 1.0:
        width, height = image.size
        crop_width = max(1, int(width * CENTER_CROP_RATIO))
        crop_height = max(1, int(height * CENTER_CROP_RATIO))
        left = max(0, (width - crop_width) // 2)
        top = max(0, (height - crop_height) // 2)
        image = image.crop((left, top, left + crop_width, top + crop_height))

    if PREPROCESS_MODE == "force_resize":
        return image.resize((IMAGE_SIZE, IMAGE_SIZE))

    if PREPROCESS_MODE == "keep_ratio_pad":
        # Preserve aspect ratio and pad to square to avoid distorting tall/wide objects.
        return ImageOps.pad(
            image,
            (IMAGE_SIZE, IMAGE_SIZE),
            color=(0, 0, 0),
            method=Image.Resampling.BICUBIC,
        )

    raise ValueError(f"Unsupported PREPROCESS_MODE: {PREPROCESS_MODE}")


def build_preprocessed_cache(image_paths: list[str], desc: str) -> None:
    unique_paths = sorted(
        {str(path).replace("\\", "/").lstrip("/") for path in image_paths}
    )
    print(f"Building preprocessed cache for {desc}: {len(unique_paths)} files")

    for raw_path in tqdm(unique_paths, desc=f"Preprocess {desc}", unit="image"):
        source_path = resolve_image_path(raw_path, root=DATA_ROOT)
        output_path = PREPROCESS_CACHE_DIR / raw_path
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if output_path.exists():
            continue

        with Image.open(source_path) as image:
            processed = preprocess_image(image)
            processed.save(output_path)


def preview_preprocessed_samples(
    df: pd.DataFrame,
    split_name: str,
    sample_count: int = PREPROCESS_PREVIEW_SAMPLES,
) -> None:
    if df.empty:
        print(f"Skip preview, no rows available for split={split_name}")
        return

    sample_count = min(sample_count, len(df))
    sampled_df = df.sample(n=sample_count, random_state=SEED).reset_index(drop=True)

    fig, axes = plt.subplots(sample_count, 2, figsize=(10, 4 * sample_count))
    if sample_count == 1:
        axes = [axes]

    for row_axes, (_, row) in zip(axes, sampled_df.iterrows()):
        raw_image_path = resolve_image_path(row["path"], root=DATA_ROOT)
        processed_image_path = resolve_image_path(
            row["path"], root=PREPROCESS_CACHE_DIR
        )

        with Image.open(raw_image_path) as raw_image:
            raw_image = raw_image.convert("RGB")
            raw_size = raw_image.size
            row_axes[0].imshow(raw_image)
            row_axes[0].set_title(
                f"Original\n{raw_image_path.name} | {raw_size[0]}x{raw_size[1]}"
            )
            row_axes[0].axis("off")

        with Image.open(processed_image_path) as processed_image:
            processed_image = processed_image.convert("RGB")
            processed_size = processed_image.size
            row_axes[1].imshow(processed_image)
            row_axes[1].set_title(
                f"Preprocessed\n{processed_size[0]}x{processed_size[1]} | {PREPROCESS_MODE}"
            )
            row_axes[1].axis("off")

    plt.tight_layout()
    plt.show()


HAS_PREPROCESSED_CACHE = False
ACTIVE_IMAGE_ROOT = DATA_ROOT

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

CONCEPT_KEYWORDS = {
    "플라스틱": ["플라스틱", "페트", "pet", "비닐"],
    "종이": ["종이", "골판지", "박스", "상자", "봉투", "팩"],
    "유리": ["유리"],
    "금속": ["금속", "캔", "알루미늄"],
    "스티로폼": ["스티로폼"],
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


def extract_concepts(text: str) -> list[str]:
    normalized = str(text).strip().lower()
    matched = []
    for concept, keywords in CONCEPT_KEYWORDS.items():
        if any(keyword.lower() in normalized for keyword in keywords):
            matched.append(concept)
    return matched


def get_choice_text(row: pd.Series, choice: str) -> str:
    choice = str(choice).strip().lower()
    if choice not in ["a", "b", "c", "d"]:
        return ""
    return str(row[choice]).strip()


def configure_processor_image_size(processor: Any, image_size: int) -> None:
    image_processor = getattr(processor, "image_processor", None)
    if image_processor is None:
        return

    target_pixels = image_size * image_size
    if hasattr(image_processor, "size"):
        try:
            image_processor.size = {
                "shortest_edge": target_pixels,
                "longest_edge": target_pixels,
            }
        except Exception:
            pass
    if hasattr(image_processor, "min_pixels"):
        image_processor.min_pixels = target_pixels
    if hasattr(image_processor, "max_pixels"):
        image_processor.max_pixels = target_pixels


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


def load_resized_image(raw_path: str) -> Image.Image:
    image_path = resolve_image_path(raw_path, root=ACTIVE_IMAGE_ROOT)
    if not image_path.exists():
        raise FileNotFoundError(f"Image file not found for path={raw_path}")

    with Image.open(image_path) as image:
        if ACTIVE_IMAGE_ROOT == PREPROCESS_CACHE_DIR:
            return image.convert("RGB")
        return preprocess_image(image)


def convert_row_to_messages(row: pd.Series, train: bool = True) -> dict[str, Any]:
    image = load_resized_image(row["path"])
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

    def __getitem__(self, idx: int) -> dict[str, Any]:
        return self.records[idx]


def save_config(train_size: int, valid_size: int) -> None:
    with CONFIG_PATH.open("w", encoding="utf-8") as f:
        f.write(f"run_name: {RUN_NAME}\n")
        f.write(f"experiment_id: {EXPERIMENT_ID}\n")
        f.write(f"base_run_name: {BASE_RUN_NAME}\n")
        f.write(f"experiment_purpose: {EXPERIMENT_PURPOSE}\n")
        f.write(f"changed_fields: {CHANGED_FIELDS}\n")
        f.write(f"model_profile: {MODEL_PROFILE}\n")
        f.write(f"train_mode: {TRAIN_MODE}\n")
        f.write(f"model_id: {MODEL_ID}\n")
        f.write(f"image_size: {IMAGE_SIZE}\n")
        f.write(f"center_crop_ratio: {CENTER_CROP_RATIO}\n")
        f.write(f"max_new_tokens: {MAX_NEW_TOKENS}\n")
        f.write(f"max_seq_length: {MAX_SEQ_LENGTH}\n")
        f.write(f"num_epochs: {NUM_EPOCHS}\n")
        f.write(f"learning_rate: {LR}\n")
        f.write(f"grad_accum: {GRAD_ACCUM}\n")
        f.write(f"batch_size: {BATCH_SIZE}\n")
        f.write(f"save_strategy: {SAVE_STRATEGY}\n")
        f.write(f"save_total_limit: {SAVE_TOTAL_LIMIT}\n")
        f.write(f"save_final_model: {SAVE_FINAL_MODEL}\n")
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


def write_summary(train_loss: float, valid_loss: float, valid_accuracy: float) -> None:
    lines = [
        f"run_name: {RUN_NAME}",
        f"train_loss: {train_loss:.4f}",
        f"valid_loss: {valid_loss:.4f}",
        f"valid_accuracy: {valid_accuracy:.4f}",
        "이번 설정 유지 여부: pending",
        "다음 액션 제안: Increase data coverage after smoke test succeeds",
    ]
    SUMMARY_PATH.write_text("\n".join(lines), encoding="utf-8")


def generate_choice(model: Any, processor: Any, row: pd.Series) -> tuple[str, str]:
    image = load_resized_image(row["path"])
    prompt = build_mc_prompt(row["question"], row["a"], row["b"], row["c"], row["d"])
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": SYSTEM_INSTRUCT + "\n\n" + prompt},
            ],
        }
    ]

    input_text = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(
        image,
        input_text,
        add_special_tokens=False,
        return_tensors="pt",
    ).to("cuda")

    outputs = model.generate(
        **inputs,
        max_new_tokens=MAX_NEW_TOKENS,
        use_cache=True,
        do_sample=False,
    )
    raw_text = processor.batch_decode(outputs, skip_special_tokens=True)[0]
    return extract_choice(raw_text), raw_text


# %% [markdown]
# ## Load Data

# %%
train_df = pd.read_csv(TRAIN_CSV)
test_df = pd.read_csv(TEST_CSV)

if USE_SUBSAMPLE:
    train_df = train_df.sample(
        n=min(SUBSAMPLE_SIZE, len(train_df)), random_state=SEED
    ).reset_index(drop=True)

if TRAIN_MODE in ["smoke", "train_valid_split"]:
    split_index = int(len(train_df) * 0.9)
    train_subset = train_df.iloc[:split_index].reset_index(drop=True)
    valid_subset = train_df.iloc[split_index:].reset_index(drop=True)
else:
    train_subset = train_df.reset_index(drop=True)
    valid_subset = train_df.iloc[0:0].copy().reset_index(drop=True)

HAS_VALID_SPLIT = len(valid_subset) > 0

if PREPROCESS_IMAGES and BUILD_PREPROCESSED_CACHE:
    cache_paths = train_subset["path"].tolist() + valid_subset["path"].tolist()
    if RUN_FULL_INFERENCE:
        cache_paths.extend(test_df["path"].tolist())
    build_preprocessed_cache(cache_paths, desc="active train/valid split")

HAS_PREPROCESSED_CACHE = any(PREPROCESS_CACHE_DIR.rglob("*"))
ACTIVE_IMAGE_ROOT = PREPROCESS_CACHE_DIR if HAS_PREPROCESSED_CACHE else DATA_ROOT

print("PREPROCESS_IMAGES:", PREPROCESS_IMAGES)
print("PREPROCESS_MODE:", PREPROCESS_MODE)
print("BUILD_PREPROCESSED_CACHE:", BUILD_PREPROCESSED_CACHE)
print("Preprocess cache dir:", PREPROCESS_CACHE_DIR)
print("Using preprocessed cache:", HAS_PREPROCESSED_CACHE)
print("Active image root:", ACTIVE_IMAGE_ROOT)

if PREPROCESS_IMAGES and not HAS_PREPROCESSED_CACHE:
    raise RuntimeError(
        "Image preprocessing is enabled but the preprocessed cache was not created. "
        "Check BUILD_PREPROCESSED_CACHE and the selected dataset split."
    )

print(f"Train rows used: {len(train_df):,}")
print(f"Train subset: {len(train_subset):,}")
print(f"Valid subset: {len(valid_subset):,}")
if HAS_VALID_SPLIT:
    print(
        "Train/valid split ratio:",
        f"{len(train_subset)}:{len(valid_subset)}",
        f"(~{len(train_subset) / max(len(train_df), 1):.1%}:{len(valid_subset) / max(len(train_df), 1):.1%})",
    )
else:
    print("Train mode uses all train rows for fitting. Validation is skipped.")
print(f"Test rows: {len(test_df):,}")

# %% [markdown]
# ## Preview Preprocessed Samples
#
# 실제 학습에 사용하는 train subset 기준으로 원본과 전처리 결과를 함께 확인합니다.

# %%
preview_preprocessed_samples(train_subset, split_name="train")

sample_image_path = resolve_image_path(
    train_subset.iloc[0]["path"], root=ACTIVE_IMAGE_ROOT
)
print("First train image:", sample_image_path)
train_df.head(3)

save_config(train_size=len(train_subset), valid_size=len(valid_subset))

# %% [markdown]
# ## Load Model
#
# Day13의 Unsloth Vision 예시처럼 `FastVisionModel.from_pretrained` 와 `get_peft_model` 조합을 사용합니다.
#
# 기본값은 `unsloth/Qwen3-VL-2B-Instruct-unsloth-bnb-4bit` 입니다.
# T4 MVP에서는 Unsloth가 직접 제공하는 vision 4bit 체크포인트를 우선 사용합니다.
# 모델 전환은 상단 `USER_MODEL_PROFILE` 값을 바꿔서 진행합니다.

# %%
model, processor = FastVisionModel.from_pretrained(
    model_name=MODEL_ID,
    max_seq_length=MAX_SEQ_LENGTH,
    load_in_4bit=LOAD_IN_4BIT,
    load_in_16bit=LOAD_IN_16BIT,
    full_finetuning=FULL_FINETUNING,
)

configure_processor_image_size(processor, IMAGE_SIZE)
print("Configured processor image size:", IMAGE_SIZE)

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
)

# %% [markdown]
# ## Build Dataset

# %%
train_records = [
    convert_row_to_messages(row, train=True)
    for _, row in tqdm(
        train_subset.iterrows(),
        total=len(train_subset),
        desc="Build train dataset",
        unit="sample",
    )
]
valid_records = [
    convert_row_to_messages(row, train=True)
    for _, row in tqdm(
        valid_subset.iterrows(),
        total=len(valid_subset),
        desc="Build valid dataset",
        unit="sample",
    )
]

train_dataset = HFDataset.from_list(train_records)
valid_dataset = VisionRecordDataset(valid_records)

data_collator = UnslothVisionDataCollator(
    model,
    processor,
    max_seq_length=MAX_SEQ_LENGTH,
    resize=IMAGE_SIZE,
    train_on_responses_only=False,
    completion_only_loss=True,
)

try:
    _sanity_batch = data_collator([train_records[0]])
    print("Collator smoke check passed.")
    print({k: tuple(v.shape) for k, v in _sanity_batch.items() if hasattr(v, "shape")})
    if "image_grid_thw" in _sanity_batch:
        print("image_grid_thw values:", _sanity_batch["image_grid_thw"].tolist())
except Exception as exc:
    raise RuntimeError(
        "Collator smoke check failed before training. Reduce IMAGE_SIZE further or increase MAX_SEQ_LENGTH."
    ) from exc

# %% [markdown]
# ## Train
#
# Day13의 Vision SFT 예시와 동일하게 `remove_unused_columns=False` 등 Vision trainer 필수 옵션을 유지합니다.

# %%
FastVisionModel.for_training(model)

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
        save_strategy=SAVE_STRATEGY,
        save_total_limit=SAVE_TOTAL_LIMIT,
        eval_strategy="no",
        optim="adamw_torch",
        seed=SEED,
        max_seq_length=MAX_SEQ_LENGTH,
        report_to="none",
        gradient_checkpointing=False,
        remove_unused_columns=False,
        dataset_text_field="",
        dataset_kwargs={"skip_prepare_dataset": True},
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
    ),
)

train_result = trainer.train()
train_loss = float(train_result.training_loss)

# %% [markdown]
# ## Validation Loss

# %%
valid_loss = 0.0
valid_accuracy = 0.0
valid_predictions = []

if HAS_VALID_SPLIT:
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
            batch = {k: v.to("cuda") for k, v in batch.items()}
            outputs = model(**batch)
            valid_loss_sum += float(outputs.loss.item())
            valid_steps += 1

    valid_loss = valid_loss_sum / valid_steps if valid_steps else 0.0
    print("Valid loss:", valid_loss)
else:
    print("Skipping validation loss because train mode is train_all_for_submission.")

# %% [markdown]
# ## Validation Generation

# %%
if HAS_VALID_SPLIT:
    type_correct = defaultdict(int)
    type_total = defaultdict(int)
    concept_correct = defaultdict(int)
    concept_total = defaultdict(int)
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
        answer_choice_text = get_choice_text(row, answer)
        pred_choice_text = get_choice_text(row, pred)
        answer_concepts = extract_concepts(answer_choice_text)
        pred_concepts = extract_concepts(pred_choice_text)

        correct += int(is_correct)
        type_total[question_type] += 1
        type_correct[question_type] += int(is_correct)

        if question_type in ["material", "classification"]:
            for concept in answer_concepts:
                concept_total[concept] += 1
                concept_correct[concept] += int(is_correct)

        valid_predictions.append(
            {
                "id": row["id"],
                "question_type": question_type,
                "question": row["question"],
                "pred": pred,
                "answer": answer,
                "pred_choice_text": pred_choice_text,
                "answer_choice_text": answer_choice_text,
                "pred_concepts": "|".join(pred_concepts),
                "answer_concepts": "|".join(answer_concepts),
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

    concept_acc_rows = []
    for concept in sorted(concept_total):
        concept_acc_rows.append(
            {
                "concept": concept,
                "correct": concept_correct[concept],
                "total": concept_total[concept],
                "accuracy": round(concept_correct[concept] / concept_total[concept], 4),
            }
        )

    valid_df = pd.DataFrame(valid_predictions)
    type_acc_df = pd.DataFrame(type_acc_rows)
    concept_acc_df = pd.DataFrame(concept_acc_rows)
    errors_df = valid_df.loc[~valid_df["correct"]].reset_index(drop=True)

    valid_df.to_csv(VALID_CSV_PATH, index=False, encoding="utf-8-sig")
    type_acc_df.to_csv(VALID_TYPE_CSV_PATH, index=False, encoding="utf-8-sig")
    concept_acc_df.to_csv(VALID_CONCEPT_CSV_PATH, index=False, encoding="utf-8-sig")
    errors_df.to_csv(VALID_ERRORS_CSV_PATH, index=False, encoding="utf-8-sig")

    with VALID_TXT_PATH.open("w", encoding="utf-8") as f:
        f.write(f"epoch: {NUM_EPOCHS}\n")
        f.write(f"train_mode: {TRAIN_MODE}\n")
        f.write(f"train_loss: {train_loss:.4f}\n")
        f.write(f"valid_size: {len(valid_predictions)}\n")
        f.write(f"valid_loss: {valid_loss:.4f}\n")
        f.write(f"valid_accuracy: {valid_accuracy:.4f}\n")
        if concept_acc_rows:
            f.write("concept_accuracy:\n")
            for item in concept_acc_rows:
                f.write(
                    f"- {item['concept']}: {item['accuracy']:.4f} ({item['correct']}/{item['total']})\n"
                )

    sample_lines = []
    for item in valid_predictions[:10]:
        sample_lines.append(
            " | ".join(
                [
                    f"id={item['id']}",
                    f"type={item['question_type']}",
                    f"pred={item['pred']}",
                    f"answer={item['answer']}",
                    f"correct={item['correct']}",
                    f"question={item['question']}",
                ]
            )
        )
    VALID_SAMPLES_TXT_PATH.write_text("\n".join(sample_lines), encoding="utf-8")

    write_summary(
        train_loss=train_loss, valid_loss=valid_loss, valid_accuracy=valid_accuracy
    )
    print("Valid accuracy:", valid_accuracy)
    print("Valid CSV saved:", VALID_CSV_PATH)
    print("Valid summary TXT saved:", VALID_TXT_PATH)
    print("Valid samples TXT saved:", VALID_SAMPLES_TXT_PATH)
    print("Valid question-type CSV saved:", VALID_TYPE_CSV_PATH)
    print("Valid concept CSV saved:", VALID_CONCEPT_CSV_PATH)
    print("Valid errors CSV saved:", VALID_ERRORS_CSV_PATH)
else:
    VALID_TXT_PATH.write_text(
        "\n".join(
            [
                f"epoch: {NUM_EPOCHS}",
                f"train_mode: {TRAIN_MODE}",
                f"train_loss: {train_loss:.4f}",
                "validation: skipped (train_all_for_submission)",
            ]
        ),
        encoding="utf-8",
    )
    write_summary(train_loss=train_loss, valid_loss=0.0, valid_accuracy=0.0)
    print(
        "Skipping validation generation because train mode is train_all_for_submission."
    )

# %% [markdown]
# ## Save Model

# %%
if SAVE_FINAL_MODEL:
    model.save_pretrained(str(SAVE_DIR))
    processor.save_pretrained(str(SAVE_DIR))
    print("Final model saved:", SAVE_DIR)
else:
    print("Skipping final model save by user setting.")

# %% [markdown]
# ## Optional Full Inference
#
# MVP 단계에서는 기본적으로 실행하지 않습니다.

# %%
if RUN_FULL_INFERENCE:
    submission_df = pd.read_csv(SAMPLE_SUBMISSION_CSV)
    preds = []

    for _, row in tqdm(
        test_df.iterrows(), total=len(test_df), desc="Test generate", unit="sample"
    ):
        pred, _ = generate_choice(model, processor, row)
        preds.append(pred)

    submission_df["answer"] = preds
    submission_df.to_csv(SUBMISSION_PATH, index=False, encoding="utf-8-sig")
    print("Submission saved:", SUBMISSION_PATH)
else:
    print("Skipping full test inference for MVP run.")

# %% [markdown]
# ## Backup Outputs To Drive
#
# 세션 종료 전 실행해 두면 안전합니다.

# %%
DRIVE_RUNS_ROOT = DRIVE_OUTPUT_ROOT / "runs"
DRIVE_RUNS_ROOT.mkdir(parents=True, exist_ok=True)
DRIVE_RUN_DIR = DRIVE_RUNS_ROOT / RUN_NAME

if DRIVE_RUN_DIR.exists():
    print(f"Drive run already exists: {DRIVE_RUN_DIR}")
else:
    shutil.copytree(RUN_DIR, DRIVE_RUN_DIR)
    print(f"Copied run to Drive: {DRIVE_RUN_DIR}")

# %% [markdown]
# ## Release Colab Runtime
#
# smoke 모드가 아닐 때는 작업 종료 후 Colab 런타임을 반환합니다.

# %%
if TRAIN_MODE in ["train_valid_split", "train_all_for_submission"]:
    try:
        from google.colab import runtime

        print("Releasing Colab runtime after completed run.")
        runtime.unassign()
    except ImportError:
        print(
            "google.colab.runtime is unavailable outside Colab. Skipping runtime release."
        )
else:
    print("Keeping Colab runtime alive because TRAIN_MODE is smoke.")
