import pandas as pd
import numpy as np
import re
import os

from sklearn.model_selection import GroupShuffleSplit


# ============================================================
# STEP 3 - LEAKAGE-AWARE PREPROCESSING & DATA SPLIT
# ============================================================

print("=" * 60)
print("STEP 3 - LEAKAGE-AWARE PREPROCESSING & DATA SPLIT")
print("=" * 60)


# ============================================================
# 1. LOAD DATASET
# ============================================================

INPUT_PATH = "data/labeled_data.csv"

df = pd.read_csv(INPUT_PATH)

print("\nDataset loaded successfully.")
print(f"Records : {len(df)}")
print(f"Columns : {len(df.columns)}")


# ============================================================
# 2. REQUIRED COLUMNS
# ============================================================

required_columns = [
    "tweet",
    "class"
]

missing_columns = [
    col for col in required_columns
    if col not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing required columns: {missing_columns}"
    )

print("\nRequired columns verified.")


# ============================================================
# 3. REMOVE EXACT DUPLICATES
# ============================================================

before = len(df)

df = df.drop_duplicates(
    subset=["tweet"]
).copy()

after = len(df)

print(f"\nExact duplicate tweets removed: {before - after}")


# ============================================================
# 4. HANDLE MISSING VALUES
# ============================================================

before = len(df)

df = df.dropna(
    subset=["tweet", "class"]
).copy()

after = len(df)

print(
    f"Rows removed due to missing values: "
    f"{before - after}"
)


# ============================================================
# 5. TEXT NORMALIZATION
# ============================================================

def normalize_text(text):

    text = str(text).lower()

    # Remove RT marker
    text = re.sub(
        r"\brt\b",
        " ",
        text
    )

    # Remove usernames
    text = re.sub(
        r"@\w+",
        " ",
        text
    )

    # Remove URLs
    text = re.sub(
        r"http\S+|www\S+",
        " ",
        text
    )

    # Decode common HTML entities
    text = text.replace(
        "&amp;",
        "and"
    )

    text = text.replace(
        "&lt;",
        " "
    )

    text = text.replace(
        "&gt;",
        " "
    )

    # Remove punctuation
    text = re.sub(
        r"[^a-z0-9\s]",
        " ",
        text
    )

    # Normalize whitespace
    text = re.sub(
        r"\s+",
        " ",
        text
    ).strip()

    return text


print("\nCreating normalized text...")

df["normalized_text"] = (
    df["tweet"]
    .apply(normalize_text)
)


# ============================================================
# 6. REMOVE EMPTY TEXT
# ============================================================

before = len(df)

df = df[
    df["normalized_text"].str.len() > 0
].copy()

after = len(df)

print(
    f"Empty normalized records removed: "
    f"{before - after}"
)


# ============================================================
# 7. CREATE MODEL TEXT
# ============================================================

# Keep the original tweet for analysis.
# The cleaned version is used by the models.

df["model_text"] = df["normalized_text"]


# ============================================================
# 8. LABEL MAPPING
# ============================================================

label_mapping = {
    0: "Hate_Speech",
    1: "Offensive_Language",
    2: "Neither"
}

df["label"] = df["class"].map(label_mapping)


if df["label"].isna().any():

    unknown_classes = (
        df.loc[
            df["label"].isna(),
            "class"
        ].unique()
    )

    raise ValueError(
        f"Unknown class values found: {unknown_classes}"
    )


# ============================================================
# 9. DATASET INFORMATION
# ============================================================

print("\n" + "=" * 60)
print("FINAL DATASET INFORMATION")
print("=" * 60)

print(f"\nRecords : {len(df)}")

print("\nClass Distribution:")

print(
    df["label"]
    .value_counts()
)


# ============================================================
# 10. LEAKAGE-AWARE GROUP SPLIT
# ============================================================

print("\n" + "=" * 60)
print("LEAKAGE-AWARE TRAIN/TEST SPLIT")
print("=" * 60)


# Every identical normalized text gets the same group ID.
# Therefore the same normalized text cannot appear
# in both training and testing.

groups = (
    df["normalized_text"]
    .factorize()[0]
)


splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=0.20,
    random_state=42
)


train_indices, test_indices = next(
    splitter.split(
        df,
        df["label"],
        groups=groups
    )
)


train_df = df.iloc[
    train_indices
].copy()

test_df = df.iloc[
    test_indices
].copy()


# ============================================================
# 11. VERIFY NO NORMALIZED TEXT LEAKAGE
# ============================================================

train_texts = set(
    train_df["normalized_text"]
)

test_texts = set(
    test_df["normalized_text"]
)

overlap = train_texts.intersection(
    test_texts
)


print("\nTrain/Test Statistics:")

print(
    f"Training samples : {len(train_df)}"
)

print(
    f"Testing samples  : {len(test_df)}"
)

print(
    f"Normalized text overlap : {len(overlap)}"
)


if len(overlap) == 0:

    print(
        "\n✓ Leakage check passed."
    )

else:

    print(
        "\nWARNING: Normalized text overlap detected."
    )


# ============================================================
# 12. CLASS DISTRIBUTION
# ============================================================

print("\n" + "=" * 60)
print("TRAINING CLASS DISTRIBUTION")
print("=" * 60)

print(
    train_df["label"]
    .value_counts()
)


print("\n" + "=" * 60)
print("TESTING CLASS DISTRIBUTION")
print("=" * 60)

print(
    test_df["label"]
    .value_counts()
)


# ============================================================
# 13. TEXT LENGTH FEATURES
# ============================================================

df["text_length"] = (
    df["tweet"]
    .astype(str)
    .str.len()
)

df["word_count"] = (
    df["tweet"]
    .astype(str)
    .str.split()
    .str.len()
)


# ============================================================
# 14. SAVE PROCESSED DATASET
# ============================================================

os.makedirs(
    "data/processed",
    exist_ok=True
)


# Save complete processed dataset

df.to_csv(
    "data/processed/processed_dataset.csv",
    index=False
)


# Save train/test datasets

train_df.to_csv(
    "data/processed/train.csv",
    index=False
)

test_df.to_csv(
    "data/processed/test.csv",
    index=False
)


# ============================================================
# 15. SAVE SPLIT SUMMARY
# ============================================================

summary = pd.DataFrame({

    "Metric": [
        "Total Records",
        "Training Records",
        "Testing Records",
        "Test Percentage",
        "Normalized Text Overlap",
        "Number of Classes"
    ],

    "Value": [
        len(df),
        len(train_df),
        len(test_df),
        round(
            len(test_df) / len(df) * 100,
            2
        ),
        len(overlap),
        df["label"].nunique()
    ]
})


os.makedirs(
    "results/preprocessing",
    exist_ok=True
)


summary.to_csv(
    "results/preprocessing/split_summary.csv",
    index=False
)


# ============================================================
# 16. SAVE CLASS MAPPING
# ============================================================

mapping_df = pd.DataFrame({
    "class_id": list(label_mapping.keys()),
    "label": list(label_mapping.values())
})


mapping_df.to_csv(
    "results/preprocessing/class_mapping.csv",
    index=False
)


# ============================================================
# 17. SAMPLE RECORDS
# ============================================================

print("\n" + "=" * 60)
print("PREPROCESSING EXAMPLES")
print("=" * 60)

for i in range(
    min(5, len(df))
):

    row = df.iloc[i]

    print("\nOriginal:")
    print(row["tweet"])

    print("\nNormalized:")
    print(row["normalized_text"])

    print("\nLabel:")
    print(row["label"])

    print("-" * 60)


# ============================================================
# 18. FINAL OUTPUT
# ============================================================

print("\n" + "=" * 60)
print("PREPROCESSING OUTPUT")
print("=" * 60)

print(
    "\nGenerated files:"
)

print(
    "- data/processed/processed_dataset.csv"
)

print(
    "- data/processed/train.csv"
)

print(
    "- data/processed/test.csv"
)

print(
    "- results/preprocessing/split_summary.csv"
)

print(
    "- results/preprocessing/class_mapping.csv"
)


print("\n" + "=" * 60)
print("FEATURE PREPROCESSING COMPLETED")
print("=" * 60)