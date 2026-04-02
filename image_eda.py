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
# # Image EDA
#
# 로컬 환경에서 VQA 이미지 데이터의 기본 EDA를 빠르게 확인하기 위한 노트북용 `.py` 파일입니다.
#
# 확인 항목:
# - `train.csv` 컬럼 및 샘플
# - 이미지 파일 존재 여부
# - 해상도 분포
# - 종횡비 분포
# - 질문 유형 분포
# - 이미지 샘플 시각화

# %%
from __future__ import annotations

from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image

plt.rcParams["figure.figsize"] = (8, 5)

# %% [markdown]
# ## Paths

# %%
WORKDIR = Path.cwd()
DATA_ROOT = WORKDIR
TRAIN_CSV = DATA_ROOT / "train.csv"
TEST_CSV = DATA_ROOT / "test.csv"
EDA_OUTPUT_DIR = WORKDIR / "eda_outputs"

EDA_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("WORKDIR:", WORKDIR)
print("DATA_ROOT:", DATA_ROOT)
print("TRAIN_CSV exists:", TRAIN_CSV.exists())
print("TEST_CSV exists:", TEST_CSV.exists())
print("EDA_OUTPUT_DIR:", EDA_OUTPUT_DIR)

# %% [markdown]
# ## Load CSV

# %%
train_df = pd.read_csv(TRAIN_CSV)
test_df = pd.read_csv(TEST_CSV) if TEST_CSV.exists() else pd.DataFrame()

print("train shape:", train_df.shape)
print("test shape:", test_df.shape)
print("train columns:", list(train_df.columns))
train_df.head(3)

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


def resolve_image_path(raw_path: str) -> Path:
    normalized = str(raw_path).replace("\\", "/").lstrip("/")
    candidate = DATA_ROOT / normalized
    if candidate.exists():
        return candidate

    fallback = DATA_ROOT / Path(normalized).name
    if fallback.exists():
        return fallback

    return candidate


def classify_question_type(question: str) -> str:
    text = str(question)
    for label, pattern in QUESTION_TYPE_PATTERNS.items():
        if pd.notna(text) and __import__("re").search(pattern, text):
            return label
    return "other"


def collect_image_metadata(df: pd.DataFrame, limit: int | None = None) -> pd.DataFrame:
    rows = []
    subset = df if limit is None else df.head(limit)

    for _, row in subset.iterrows():
        image_path = resolve_image_path(row["path"])
        exists = image_path.exists()
        width = None
        height = None
        mode = None

        if exists:
            with Image.open(image_path) as image:
                width, height = image.size
                mode = image.mode

        rows.append(
            {
                "id": row.get("id"),
                "path": row["path"],
                "resolved_path": str(image_path),
                "exists": exists,
                "width": width,
                "height": height,
                "aspect_ratio": round(width / height, 4) if width and height else None,
                "mode": mode,
                "question_type": classify_question_type(row.get("question", "")),
            }
        )

    return pd.DataFrame(rows)


# %% [markdown]
# ## File Existence Check

# %%
meta_df = collect_image_metadata(train_df, limit=500)
print("checked rows:", len(meta_df))
print("missing images:", int((~meta_df["exists"]).sum()))
meta_df.loc[~meta_df["exists"]].head(10)

meta_df.to_csv(
    EDA_OUTPUT_DIR / "image_meta_sample.csv", index=False, encoding="utf-8-sig"
)

# %% [markdown]
# ## Resolution Summary

# %%
valid_meta_df = meta_df.loc[meta_df["exists"]].copy()

print("resolution summary")
print(valid_meta_df[["width", "height", "aspect_ratio"]].describe())

top_sizes = Counter(zip(valid_meta_df["width"], valid_meta_df["height"])).most_common(
    20
)
top_sizes_df = pd.DataFrame(top_sizes, columns=["size", "count"])
top_sizes_df.head(10)

top_sizes_export_df = top_sizes_df.copy()
top_sizes_export_df[["width", "height"]] = pd.DataFrame(
    top_sizes_export_df["size"].tolist(), index=top_sizes_export_df.index
)
top_sizes_export_df = top_sizes_export_df[["width", "height", "count"]]
top_sizes_export_df.to_csv(
    EDA_OUTPUT_DIR / "top_image_sizes.csv", index=False, encoding="utf-8-sig"
)

valid_meta_df.to_csv(
    EDA_OUTPUT_DIR / "valid_image_meta_sample.csv", index=False, encoding="utf-8-sig"
)

# %% [markdown]
# ## Resolution Plots

# %%
fig, axes = plt.subplots(1, 3, figsize=(18, 4))

valid_meta_df["width"].hist(ax=axes[0], bins=20)
axes[0].set_title("Width Distribution")

valid_meta_df["height"].hist(ax=axes[1], bins=20)
axes[1].set_title("Height Distribution")

valid_meta_df["aspect_ratio"].hist(ax=axes[2], bins=20)
axes[2].set_title("Aspect Ratio Distribution")

plt.tight_layout()
plt.savefig(
    EDA_OUTPUT_DIR / "resolution_distributions.png", dpi=150, bbox_inches="tight"
)
plt.show()

# %% [markdown]
# ## Question Type Distribution

# %%
question_type_counts = valid_meta_df["question_type"].value_counts()
print(question_type_counts)

ax = question_type_counts.plot(kind="bar", rot=30, title="Question Type Distribution")
ax.set_xlabel("question_type")
ax.set_ylabel("count")
plt.tight_layout()
plt.savefig(
    EDA_OUTPUT_DIR / "question_type_distribution.png", dpi=150, bbox_inches="tight"
)
plt.show()

# %% [markdown]
# ## Distortion Simulation
#
# 현재 `224x224` 정사각형 강제 resize를 가정했을 때, 비율 유지 resize + pad와 얼마나 차이나는지 확인합니다.

# %%
TARGET_SIZE = 224

sim_rows = []
for _, row in valid_meta_df.head(30).iterrows():
    w = row["width"]
    h = row["height"]
    scale = min(TARGET_SIZE / w, TARGET_SIZE / h)
    resized_w = round(w * scale)
    resized_h = round(h * scale)
    sim_rows.append(
        {
            "path": row["path"],
            "orig_w": w,
            "orig_h": h,
            "forced_resize": f"{TARGET_SIZE}x{TARGET_SIZE}",
            "keep_ratio_resize": f"{resized_w}x{resized_h}",
            "pad_w": TARGET_SIZE - resized_w,
            "pad_h": TARGET_SIZE - resized_h,
        }
    )

sim_df = pd.DataFrame(sim_rows)
sim_df.head(10)

sim_df.to_csv(
    EDA_OUTPUT_DIR / "resize_distortion_simulation.csv",
    index=False,
    encoding="utf-8-sig",
)

# %% [markdown]
# ## Sample Images

# %%
sample_df = train_df.head(12).copy()

fig, axes = plt.subplots(3, 4, figsize=(14, 10))
axes = axes.flatten()

for ax, (_, row) in zip(axes, sample_df.iterrows()):
    image_path = resolve_image_path(row["path"])
    with Image.open(image_path) as image:
        ax.imshow(image.convert("RGB"))
    ax.set_title(f"id={row.get('id')}\n{Path(str(row['path'])).name}", fontsize=9)
    ax.axis("off")

for ax in axes[len(sample_df) :]:
    ax.axis("off")

plt.tight_layout()
plt.savefig(EDA_OUTPUT_DIR / "sample_images_grid.png", dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## Notes
#
# 아래를 보고 resize 전략을 결정하면 됩니다.
#
# - 종횡비가 거의 모두 같고 정사각형에 가깝다: 강제 정사각형 resize도 큰 문제 없을 수 있음
# - 종횡비가 다양하다: 비율 유지 + pad가 더 적절할 가능성이 큼
# - `count`, `location` 질문이 약하다: resize 왜곡 영향 가능성을 먼저 의심

# %% [markdown]
# ## Export Summary
#
# 아래 파일들을 저에게 보여주면 결과를 같이 해석할 수 있습니다.

# %%
summary_lines = [
    f"workdir: {WORKDIR}",
    f"data_root: {DATA_ROOT}",
    f"train_shape: {train_df.shape}",
    f"test_shape: {test_df.shape}",
    f"checked_image_rows: {len(meta_df)}",
    f"existing_images: {int(meta_df['exists'].sum())}",
    f"missing_images: {int((~meta_df['exists']).sum())}",
]

if not valid_meta_df.empty:
    summary_lines.extend(
        [
            f"width_min: {int(valid_meta_df['width'].min())}",
            f"width_max: {int(valid_meta_df['width'].max())}",
            f"height_min: {int(valid_meta_df['height'].min())}",
            f"height_max: {int(valid_meta_df['height'].max())}",
            f"aspect_ratio_min: {float(valid_meta_df['aspect_ratio'].min()):.4f}",
            f"aspect_ratio_max: {float(valid_meta_df['aspect_ratio'].max()):.4f}",
            f"top_question_types: {question_type_counts.to_dict()}",
        ]
    )

summary_path = EDA_OUTPUT_DIR / "eda_summary.txt"
summary_path.write_text("\n".join(summary_lines), encoding="utf-8")

print("Saved:", EDA_OUTPUT_DIR / "image_meta_sample.csv")
print("Saved:", EDA_OUTPUT_DIR / "valid_image_meta_sample.csv")
print("Saved:", EDA_OUTPUT_DIR / "top_image_sizes.csv")
print("Saved:", EDA_OUTPUT_DIR / "resize_distortion_simulation.csv")
print("Saved:", EDA_OUTPUT_DIR / "resolution_distributions.png")
print("Saved:", EDA_OUTPUT_DIR / "question_type_distribution.png")
print("Saved:", EDA_OUTPUT_DIR / "sample_images_grid.png")
print("Saved:", summary_path)
