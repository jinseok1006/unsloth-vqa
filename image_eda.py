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
# - 정답 클래스 불균형 분포
# - 이미지 샘플 시각화

# %%
from __future__ import annotations

from collections import Counter
from pathlib import Path
import re

import matplotlib.pyplot as plt
from matplotlib import font_manager
import pandas as pd
from PIL import Image

plt.rcParams["figure.figsize"] = (8, 5)
plt.rcParams["axes.unicode_minus"] = False

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

TOP_N_TARGET_CLASSES_FOR_PLOT = 30
TOP_N_PER_QUESTION_TYPE = 15
CHOICE_COLUMNS = ["a", "b", "c", "d"]
CONCEPT_KEYWORDS = {
    "플라스틱": ["플라스틱", "페트", "pet", "비닐"],
    "종이": ["종이", "골판지", "박스", "상자", "봉투", "팩"],
    "유리": ["유리"],
    "금속": ["금속", "캔", "알루미늄"],
    "스티로폼": ["스티로폼"],
}


def configure_korean_font() -> str:
    font_file_candidates = [
        Path("C:/Windows/Fonts/malgun.ttf"),
        Path("C:/Windows/Fonts/malgunbd.ttf"),
        Path("/usr/share/fonts/truetype/nanum/NanumGothic.ttf"),
        Path("/usr/share/fonts/truetype/nanum/NanumBarunGothic.ttf"),
        Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
    ]
    font_candidates = [
        "Malgun Gothic",
        "AppleGothic",
        "NanumGothic",
        "NanumBarunGothic",
        "Noto Sans CJK KR",
        "Noto Sans KR",
    ]

    for font_path in font_file_candidates:
        if font_path.exists():
            try:
                font_manager.fontManager.addfont(str(font_path))
            except RuntimeError:
                pass

    available_fonts = {font.name for font in font_manager.fontManager.ttflist}
    for font_name in font_candidates:
        if font_name in available_fonts:
            plt.rcParams["font.family"] = font_name
            return font_name

    return "default"


def plot_top_n_barh(
    df: pd.DataFrame,
    label_col: str,
    value_col: str,
    title: str,
    output_path: Path,
    top_n: int = TOP_N_TARGET_CLASSES_FOR_PLOT,
    color: str = "tab:blue",
) -> None:
    plot_df = df.head(top_n).copy().iloc[::-1]
    fig_height = max(6, top_n * 0.35)
    fig, ax = plt.subplots(figsize=(12, fig_height))
    ax.barh(plot_df[label_col], plot_df[value_col], color=color)
    ax.set_title(title)
    ax.set_xlabel(value_col)
    ax.set_ylabel(label_col)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.show()


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
        if pd.notna(text) and re.search(pattern, text):
            return label
    return "other"


def extract_correct_class(row: pd.Series) -> str | None:
    answer = str(row.get("answer", "")).strip().lower()
    if answer not in ["a", "b", "c", "d"]:
        return None
    return str(row.get(answer, "")).strip()


def normalize_answer_label(text: str | None) -> str | None:
    if text is None:
        return None

    normalized = str(text).strip()
    if not normalized:
        return None

    normalized = re.sub(r"\s+", " ", normalized)
    normalized = normalized.replace("( ", "(").replace(" )", ")")

    conservative_replacements = {
        "유리 병": "유리병",
        "종이 상자": "종이상자",
        "종이 박스": "종이박스",
        "플라스틱 병": "플라스틱병",
        "금속 캔": "금속캔",
        "음료 캔": "음료캔",
        "음료수 캔": "음료수캔",
        "플라스틱 용기": "플라스틱용기",
        "플라스틱 컵": "플라스틱컵",
        "플라스틱 빨대": "플라스틱빨대",
        "종이 봉투": "종이봉투",
        "종이 팩": "종이팩",
        "스티로폼 박스": "스티로폼박스",
        "페트 병": "페트병",
        "플라스틱 뚜껑": "플라스틱뚜껑",
        "플라스틱 포장지": "플라스틱포장지",
        "플라스틱 포장재": "플라스틱포장재",
    }
    normalized = conservative_replacements.get(normalized, normalized)
    return normalized


def build_normalization_candidates(labels: pd.Series) -> pd.DataFrame:
    rows = []
    for raw_label, count in labels.value_counts().items():
        normalized_label = normalize_answer_label(raw_label)
        rows.append(
            {
                "raw_label": raw_label,
                "normalized_label": normalized_label,
                "count": count,
            }
        )

    candidate_df = pd.DataFrame(rows)
    group_sizes = candidate_df.groupby("normalized_label")["raw_label"].transform(
        "nunique"
    )
    candidate_df["group_size"] = group_sizes
    candidate_df = candidate_df.sort_values(
        by=["group_size", "normalized_label", "count"],
        ascending=[False, True, False],
    ).reset_index(drop=True)
    return candidate_df


def extract_concepts(text: str | None) -> list[str]:
    normalized = str(text).strip().lower() if text is not None else ""
    matched = []
    for concept, keywords in CONCEPT_KEYWORDS.items():
        if any(keyword.lower() in normalized for keyword in keywords):
            matched.append(concept)
    return matched


def build_question_type_top_answers(df: pd.DataFrame, top_n: int) -> pd.DataFrame:
    rows = []
    for question_type, group_df in df.groupby("question_type"):
        counts = group_df["target_class"].value_counts(dropna=True).head(top_n)
        total = max(len(group_df), 1)
        for rank, (target_class, count) in enumerate(counts.items(), start=1):
            rows.append(
                {
                    "question_type": question_type,
                    "rank": rank,
                    "target_class": target_class,
                    "count": count,
                    "ratio_within_question_type": round(count / total, 4),
                }
            )
    return pd.DataFrame(rows)


def build_choice_concept_stats(df: pd.DataFrame) -> pd.DataFrame:
    choice_rows = []
    for _, row in df.iterrows():
        answer_letter = str(row.get("answer", "")).strip().lower()
        question_type = row.get("question_type", "other")
        for choice_col in CHOICE_COLUMNS:
            choice_text = row.get(choice_col, "")
            concepts = extract_concepts(choice_text)
            for concept in concepts:
                choice_rows.append(
                    {
                        "question_type": question_type,
                        "choice_col": choice_col,
                        "concept": concept,
                        "is_correct_choice": choice_col == answer_letter,
                    }
                )

    concept_df = pd.DataFrame(choice_rows)
    if concept_df.empty:
        return concept_df

    summary_df = (
        concept_df.groupby(["question_type", "concept"])
        .agg(
            total_choice_mentions=("concept", "size"),
            correct_choice_mentions=("is_correct_choice", "sum"),
        )
        .reset_index()
    )
    summary_df["correct_rate_within_mentions"] = (
        summary_df["correct_choice_mentions"]
        / summary_df["total_choice_mentions"].clip(lower=1)
    ).round(4)
    return summary_df.sort_values(
        by=["question_type", "total_choice_mentions"], ascending=[True, False]
    ).reset_index(drop=True)


def collect_image_metadata(df: pd.DataFrame, split: str) -> pd.DataFrame:
    rows = []

    for _, row in df.iterrows():
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
                "split": split,
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


SELECTED_FONT = configure_korean_font()
print("SELECTED_FONT:", SELECTED_FONT)


# %% [markdown]
# ## File Existence Check

# %%
train_meta_df = collect_image_metadata(train_df, split="train")
test_meta_df = (
    collect_image_metadata(test_df, split="test")
    if not test_df.empty
    else pd.DataFrame()
)
meta_df = pd.concat([train_meta_df, test_meta_df], ignore_index=True)

print("checked train rows:", len(train_meta_df))
print("train missing images:", int((~train_meta_df["exists"]).sum()))
if not test_meta_df.empty:
    print("checked test rows:", len(test_meta_df))
    print("test missing images:", int((~test_meta_df["exists"]).sum()))

meta_df.loc[~meta_df["exists"]].head(10)

train_meta_df.to_csv(
    EDA_OUTPUT_DIR / "train_image_meta.csv", index=False, encoding="utf-8-sig"
)
if not test_meta_df.empty:
    test_meta_df.to_csv(
        EDA_OUTPUT_DIR / "test_image_meta.csv", index=False, encoding="utf-8-sig"
    )

# %% [markdown]
# ## Resolution Summary

# %%
valid_meta_df = meta_df.loc[meta_df["exists"]].copy()
train_valid_meta_df = train_meta_df.loc[train_meta_df["exists"]].copy()
test_valid_meta_df = test_meta_df.loc[test_meta_df["exists"]].copy()

print("resolution summary (train + test)")
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

train_valid_meta_df.to_csv(
    EDA_OUTPUT_DIR / "train_valid_image_meta.csv", index=False, encoding="utf-8-sig"
)
if not test_valid_meta_df.empty:
    test_valid_meta_df.to_csv(
        EDA_OUTPUT_DIR / "test_valid_image_meta.csv", index=False, encoding="utf-8-sig"
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
# ## Target Class Imbalance
#
# 정답 기준 클래스 분포를 확인해 데이터 편향을 점검합니다.

# %%
train_analysis_df = train_df.copy()
train_analysis_df["question_type"] = train_analysis_df["question"].apply(
    classify_question_type
)
train_analysis_df["answer_letter"] = (
    train_analysis_df["answer"].astype(str).str.strip().str.lower()
)
train_analysis_df["target_class"] = train_analysis_df.apply(
    extract_correct_class, axis=1
)

answer_letter_counts = (
    train_analysis_df["answer_letter"].value_counts(dropna=True).sort_index()
)
answer_letter_df = answer_letter_counts.rename_axis("answer_letter").reset_index(
    name="count"
)
answer_letter_df["ratio"] = (
    answer_letter_df["count"] / answer_letter_df["count"].sum()
).round(4)

print("answer letter distribution")
print(answer_letter_df)

if not answer_letter_df.empty:
    ax = answer_letter_df.plot(
        kind="bar",
        x="answer_letter",
        y="count",
        rot=0,
        legend=False,
        title="Answer Letter Distribution",
        figsize=(8, 4),
        color="tab:orange",
    )
    ax.set_xlabel("answer_letter")
    ax.set_ylabel("count")
    plt.tight_layout()
    plt.savefig(
        EDA_OUTPUT_DIR / "answer_letter_distribution.png",
        dpi=150,
        bbox_inches="tight",
    )
    plt.show()

answer_letter_df.to_csv(
    EDA_OUTPUT_DIR / "answer_letter_distribution.csv",
    index=False,
    encoding="utf-8-sig",
)

answer_letter_by_question_type_df = pd.crosstab(
    train_analysis_df["answer_letter"],
    train_analysis_df["question_type"],
).reset_index()
answer_letter_by_question_type_df.to_csv(
    EDA_OUTPUT_DIR / "answer_letter_by_question_type.csv",
    index=False,
    encoding="utf-8-sig",
)

target_class_counts = train_analysis_df["target_class"].value_counts(dropna=True)
target_class_df = target_class_counts.rename_axis("target_class").reset_index(
    name="count"
)
target_class_df["ratio"] = (
    target_class_df["count"] / target_class_df["count"].sum()
).round(4)

print("target class distribution")
print(target_class_df)

if not target_class_df.empty:
    plot_top_n_barh(
        target_class_df,
        label_col="target_class",
        value_col="count",
        title=f"Target Class Distribution Top {min(TOP_N_TARGET_CLASSES_FOR_PLOT, len(target_class_df))}",
        output_path=EDA_OUTPUT_DIR / "target_class_distribution.png",
        top_n=TOP_N_TARGET_CLASSES_FOR_PLOT,
        color="tab:blue",
    )

target_class_df.to_csv(
    EDA_OUTPUT_DIR / "target_class_distribution.csv",
    index=False,
    encoding="utf-8-sig",
)

class_question_type_df = pd.crosstab(
    train_analysis_df["target_class"],
    train_analysis_df["question_type"],
).reset_index()
class_question_type_df.to_csv(
    EDA_OUTPUT_DIR / "target_class_by_question_type.csv",
    index=False,
    encoding="utf-8-sig",
)

top_target_class_df = target_class_df.head(20).copy()
top_target_class_df.to_csv(
    EDA_OUTPUT_DIR / "top_target_classes.csv",
    index=False,
    encoding="utf-8-sig",
)

# %% [markdown]
# ## Question-Type-Centric Target Analysis
#
# 질문 유형별로 정답 공간을 분리해 상위 정답과 개념 편향을 확인합니다.

# %%
question_type_top_answers_df = build_question_type_top_answers(
    train_analysis_df, top_n=TOP_N_PER_QUESTION_TYPE
)
question_type_top_answers_df.to_csv(
    EDA_OUTPUT_DIR / "question_type_top_answers.csv",
    index=False,
    encoding="utf-8-sig",
)

choice_concept_stats_df = build_choice_concept_stats(train_analysis_df)
choice_concept_stats_df.to_csv(
    EDA_OUTPUT_DIR / "choice_concept_stats.csv",
    index=False,
    encoding="utf-8-sig",
)

question_type_summary_rows = []
for question_type, group_df in train_analysis_df.groupby("question_type"):
    question_type_summary_rows.append(
        {
            "question_type": question_type,
            "rows": len(group_df),
            "unique_target_classes": int(group_df["target_class"].nunique()),
            "top_target_class": group_df["target_class"]
            .value_counts(dropna=True)
            .index[0],
            "top_target_class_count": int(
                group_df["target_class"].value_counts(dropna=True).iloc[0]
            ),
        }
    )

question_type_summary_df = pd.DataFrame(question_type_summary_rows).sort_values(
    by="rows", ascending=False
)
question_type_summary_df.to_csv(
    EDA_OUTPUT_DIR / "question_type_summary.csv",
    index=False,
    encoding="utf-8-sig",
)

# %% [markdown]
# ## Answer Label Normalization Candidates
#
# 띄어쓰기/표기 차이처럼 안전하게 합칠 수 있는 정답 라벨 후보를 먼저 확인합니다.

# %%
normalization_candidates_df = build_normalization_candidates(
    train_analysis_df["target_class"].dropna()
)
normalization_candidates_df.to_csv(
    EDA_OUTPUT_DIR / "answer_label_normalization_candidates.csv",
    index=False,
    encoding="utf-8-sig",
)

merged_candidate_df = normalization_candidates_df.loc[
    normalization_candidates_df["group_size"] > 1
].copy()
merged_candidate_df.to_csv(
    EDA_OUTPUT_DIR / "answer_label_normalization_merge_groups.csv",
    index=False,
    encoding="utf-8-sig",
)

train_analysis_df["normalized_target_class"] = train_analysis_df["target_class"].apply(
    normalize_answer_label
)
normalized_target_class_counts = train_analysis_df[
    "normalized_target_class"
].value_counts(dropna=True)
normalized_target_class_df = normalized_target_class_counts.rename_axis(
    "normalized_target_class"
).reset_index(name="count")
normalized_target_class_df["ratio"] = (
    normalized_target_class_df["count"] / normalized_target_class_df["count"].sum()
).round(4)
normalized_target_class_df.to_csv(
    EDA_OUTPUT_DIR / "normalized_target_class_distribution.csv",
    index=False,
    encoding="utf-8-sig",
)

if not normalized_target_class_df.empty:
    plot_top_n_barh(
        normalized_target_class_df,
        label_col="normalized_target_class",
        value_col="count",
        title=f"Normalized Target Class Distribution Top {min(TOP_N_TARGET_CLASSES_FOR_PLOT, len(normalized_target_class_df))}",
        output_path=EDA_OUTPUT_DIR / "normalized_target_class_distribution.png",
        top_n=TOP_N_TARGET_CLASSES_FOR_PLOT,
        color="tab:green",
    )

normalization_summary_df = pd.DataFrame(
    [
        {
            "raw_unique_labels": int(train_analysis_df["target_class"].nunique()),
            "normalized_unique_labels": int(
                train_analysis_df["normalized_target_class"].nunique()
            ),
            "labels_reduced": int(train_analysis_df["target_class"].nunique())
            - int(train_analysis_df["normalized_target_class"].nunique()),
        }
    ]
)
normalization_summary_df.to_csv(
    EDA_OUTPUT_DIR / "answer_label_normalization_summary.csv",
    index=False,
    encoding="utf-8-sig",
)

# %% [markdown]
# ## Distortion Simulation
#
# 현재 `384x384` 정사각형 강제 resize를 가정했을 때, 비율 유지 resize + pad와 얼마나 차이나는지 확인합니다.

# %%
TARGET_SIZE = 384

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
# - 특정 정답 위치(a/b/c/d)가 과도하게 많다: 모델이 내용보다 위치 prior를 학습할 수 있음
# - 특정 재활용품 클래스가 현저히 적다: 모델이 클래스 prior에 끌려 다수 클래스로 과예측할 수 있음
# - 같은 의미의 정답이 표기만 다르게 흩어져 있다: 정규화로 희소 라벨을 줄일 여지가 있음
# - 전체 분포보다 질문 유형 내부 분포가 더 중요할 수 있다: `count/color/material/...`를 분리해서 해석

# %% [markdown]
# ## Export Summary
#
# 아래 파일들을 저에게 보여주면 결과를 같이 해석할 수 있습니다.

# %%
summary_lines = [
    f"workdir: {WORKDIR}",
    f"data_root: {DATA_ROOT}",
    f"selected_font: {SELECTED_FONT}",
    f"train_shape: {train_df.shape}",
    f"test_shape: {test_df.shape}",
    f"checked_train_image_rows: {len(train_meta_df)}",
    f"checked_test_image_rows: {len(test_meta_df)}",
    f"checked_total_image_rows: {len(meta_df)}",
    f"existing_train_images: {int(train_meta_df['exists'].sum())}",
    f"missing_train_images: {int((~train_meta_df['exists']).sum())}",
    f"existing_test_images: {int(test_meta_df['exists'].sum()) if not test_meta_df.empty else 0}",
    f"missing_test_images: {int((~test_meta_df['exists']).sum()) if not test_meta_df.empty else 0}",
    f"existing_total_images: {int(meta_df['exists'].sum())}",
    f"missing_total_images: {int((~meta_df['exists']).sum())}",
]

if not answer_letter_df.empty:
    majority_answer_letter = answer_letter_df.loc[answer_letter_df["count"].idxmax()]
    minority_answer_letter = answer_letter_df.loc[answer_letter_df["count"].idxmin()]
    answer_letter_ratio = majority_answer_letter["count"] / max(
        minority_answer_letter["count"], 1
    )
    answer_letter_note = (
        "answer positions are roughly balanced"
        if answer_letter_ratio < 1.2
        else "answer positions are imbalanced enough to learn option position priors"
    )
    summary_lines.extend(
        [
            f"answer_letter_counts: {dict(zip(answer_letter_df['answer_letter'], answer_letter_df['count']))}",
            f"majority_answer_letter: {majority_answer_letter['answer_letter']} ({int(majority_answer_letter['count'])})",
            f"minority_answer_letter: {minority_answer_letter['answer_letter']} ({int(minority_answer_letter['count'])})",
            f"answer_letter_majority_to_minority_ratio: {answer_letter_ratio:.2f}",
            f"answer_letter_bias_note: {answer_letter_note}",
        ]
    )

if not target_class_df.empty:
    majority_class = target_class_df.iloc[0]
    minority_class = target_class_df.iloc[-1]
    imbalance_ratio = majority_class["count"] / max(minority_class["count"], 1)
    summary_lines.extend(
        [
            f"target_class_counts: {dict(zip(target_class_df['target_class'], target_class_df['count']))}",
            f"majority_class: {majority_class['target_class']} ({int(majority_class['count'])})",
            f"minority_class: {minority_class['target_class']} ({int(minority_class['count'])})",
            f"majority_to_minority_ratio: {imbalance_ratio:.2f}",
            "imbalance_note: minority classes may be under-learned and majority classes may dominate answer priors",
        ]
    )

if "normalized_target_class" in train_analysis_df:
    raw_unique_labels = int(train_analysis_df["target_class"].nunique())
    normalized_unique_labels = int(
        train_analysis_df["normalized_target_class"].nunique()
    )
    summary_lines.extend(
        [
            f"raw_unique_target_labels: {raw_unique_labels}",
            f"normalized_unique_target_labels: {normalized_unique_labels}",
            f"target_labels_reduced_by_normalization: {raw_unique_labels - normalized_unique_labels}",
        ]
    )

if not question_type_summary_df.empty:
    summary_lines.extend(
        [
            f"question_type_rows: {dict(zip(question_type_summary_df['question_type'], question_type_summary_df['rows']))}",
            f"question_type_unique_target_classes: {dict(zip(question_type_summary_df['question_type'], question_type_summary_df['unique_target_classes']))}",
            "question_type_analysis_note: compare answer distributions within each question type before interpreting overall label imbalance",
        ]
    )

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

print("Saved:", EDA_OUTPUT_DIR / "train_image_meta.csv")
if not test_meta_df.empty:
    print("Saved:", EDA_OUTPUT_DIR / "test_image_meta.csv")
print("Saved:", EDA_OUTPUT_DIR / "train_valid_image_meta.csv")
if not test_valid_meta_df.empty:
    print("Saved:", EDA_OUTPUT_DIR / "test_valid_image_meta.csv")
print("Saved:", EDA_OUTPUT_DIR / "top_image_sizes.csv")
print("Saved:", EDA_OUTPUT_DIR / "resize_distortion_simulation.csv")
print("Saved:", EDA_OUTPUT_DIR / "resolution_distributions.png")
print("Saved:", EDA_OUTPUT_DIR / "question_type_distribution.png")
print("Saved:", EDA_OUTPUT_DIR / "answer_letter_distribution.csv")
print("Saved:", EDA_OUTPUT_DIR / "answer_letter_distribution.png")
print("Saved:", EDA_OUTPUT_DIR / "answer_letter_by_question_type.csv")
print("Saved:", EDA_OUTPUT_DIR / "target_class_distribution.csv")
print("Saved:", EDA_OUTPUT_DIR / "target_class_distribution.png")
print("Saved:", EDA_OUTPUT_DIR / "target_class_by_question_type.csv")
print("Saved:", EDA_OUTPUT_DIR / "top_target_classes.csv")
print("Saved:", EDA_OUTPUT_DIR / "question_type_top_answers.csv")
print("Saved:", EDA_OUTPUT_DIR / "question_type_summary.csv")
print("Saved:", EDA_OUTPUT_DIR / "choice_concept_stats.csv")
print("Saved:", EDA_OUTPUT_DIR / "answer_label_normalization_candidates.csv")
print("Saved:", EDA_OUTPUT_DIR / "answer_label_normalization_merge_groups.csv")
print("Saved:", EDA_OUTPUT_DIR / "normalized_target_class_distribution.csv")
print("Saved:", EDA_OUTPUT_DIR / "normalized_target_class_distribution.png")
print("Saved:", EDA_OUTPUT_DIR / "answer_label_normalization_summary.csv")
print("Saved:", EDA_OUTPUT_DIR / "sample_images_grid.png")
print("Saved:", summary_path)
