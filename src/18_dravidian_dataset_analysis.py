# ================================================================
# STEP 18 - DRAVIDIAN DATASET ANALYSIS
# ================================================================
#
# Purpose:
#   Analyze the cleaned multilingual Dravidian dataset before
#   starting multilingual offensive-language modeling.
#
# Input:
#   data/dravidian_clean/dravidian_multilingual_clean.csv
#
# Output:
#   results/dravidian_dataset_analysis/
#
# No model training is performed in this step.
# ================================================================

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# ================================================================
# CONFIGURATION
# ================================================================

INPUT_FILE = "data/dravidian_clean/dravidian_multilingual_clean.csv"

OUTPUT_DIR = "results/dravidian_dataset_analysis"

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ================================================================
# HEADER
# ================================================================

print("=" * 70)
print("STEP 18 - DRAVIDIAN DATASET ANALYSIS")
print("=" * 70)

print("\nInput dataset:")
print(INPUT_FILE)

print("\nOutput directory:")
print(OUTPUT_DIR)


# ================================================================
# 1. LOAD DATASET
# ================================================================

print("\n" + "=" * 70)
print("1. LOADING CLEANED DRAVIDIAN DATASET")
print("=" * 70)

if not os.path.exists(INPUT_FILE):
    raise FileNotFoundError(
        f"\nDataset not found:\n{INPUT_FILE}\n\n"
        "Run Step 17 first."
    )

df = pd.read_csv(INPUT_FILE)

print(f"\nDataset loaded successfully.")
print(f"Total records : {len(df)}")
print(f"Total columns : {len(df.columns)}")

print("\nColumns:")
print(list(df.columns))


# ================================================================
# 2. BASIC DATASET INFORMATION
# ================================================================

print("\n" + "=" * 70)
print("2. BASIC DATASET INFORMATION")
print("=" * 70)

basic_info = pd.DataFrame({
    "Metric": [
        "Total Records",
        "Total Columns",
        "Unique Languages",
        "Unique Labels",
        "Unique Splits"
    ],
    "Value": [
        len(df),
        len(df.columns),
        df["language"].nunique(),
        df["mapped_label"].nunique(),
        df["split"].nunique()
    ]
})

print(basic_info.to_string(index=False))

basic_info.to_csv(
    os.path.join(OUTPUT_DIR, "dataset_basic_information.csv"),
    index=False
)


# ================================================================
# 3. LANGUAGE DISTRIBUTION
# ================================================================

print("\n" + "=" * 70)
print("3. LANGUAGE DISTRIBUTION")
print("=" * 70)

language_distribution = (
    df["language"]
    .value_counts()
    .rename_axis("language")
    .reset_index(name="count")
)

language_distribution["percentage"] = (
    language_distribution["count"] / len(df) * 100
)

print("\nLanguage distribution:")
print(language_distribution.to_string(index=False))

language_distribution.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "language_distribution.csv"
    ),
    index=False
)


# ================================================================
# 4. LABEL DISTRIBUTION
# ================================================================

print("\n" + "=" * 70)
print("4. LABEL DISTRIBUTION")
print("=" * 70)

label_distribution = (
    df["mapped_label"]
    .value_counts()
    .rename_axis("mapped_label")
    .reset_index(name="count")
)

label_distribution["percentage"] = (
    label_distribution["count"] / len(df) * 100
)

print("\nLabel distribution:")
print(label_distribution.to_string(index=False))

label_distribution.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "label_distribution.csv"
    ),
    index=False
)


# ================================================================
# 5. LANGUAGE × LABEL DISTRIBUTION
# ================================================================

print("\n" + "=" * 70)
print("5. LANGUAGE × LABEL DISTRIBUTION")
print("=" * 70)

language_label = pd.crosstab(
    df["language"],
    df["mapped_label"]
)

print("\nLanguage × Label:")
print(language_label)

language_label.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "language_label_distribution.csv"
    )
)


# ================================================================
# 6. LANGUAGE × LABEL PERCENTAGES
# ================================================================

print("\n" + "=" * 70)
print("6. LANGUAGE × LABEL PERCENTAGES")
print("=" * 70)

language_label_percent = pd.crosstab(
    df["language"],
    df["mapped_label"],
    normalize="index"
) * 100

print("\nPercentage within each language:")
print(language_label_percent.round(2))

language_label_percent.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "language_label_percentages.csv"
    )
)


# ================================================================
# 7. SPLIT DISTRIBUTION
# ================================================================

print("\n" + "=" * 70)
print("7. TRAIN / DEV / TEST DISTRIBUTION")
print("=" * 70)

split_distribution = pd.crosstab(
    df["language"],
    df["split"]
)

print("\nSplit distribution by language:")
print(split_distribution)

split_distribution.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "language_split_distribution.csv"
    )
)


# ================================================================
# 8. SPLIT × LANGUAGE × LABEL
# ================================================================

print("\n" + "=" * 70)
print("8. SPLIT × LANGUAGE × LABEL DISTRIBUTION")
print("=" * 70)

split_language_label = (
    df.groupby(
        ["split", "language", "mapped_label"]
    )
    .size()
    .reset_index(name="count")
)

print("\nSplit × Language × Label:")
print(split_language_label.to_string(index=False))

split_language_label.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "split_language_label_distribution.csv"
    ),
    index=False
)


# ================================================================
# 9. TEXT LENGTH ANALYSIS
# ================================================================

print("\n" + "=" * 70)
print("9. TEXT LENGTH ANALYSIS")
print("=" * 70)

# Use model_text because it is the standardized text column
TEXT_COLUMN = "model_text"

if TEXT_COLUMN not in df.columns:

    # Fallback to likely text columns
    possible_columns = [
        "text",
        "tweet",
        "sentence",
        "comment"
    ]

    found_column = None

    for column in possible_columns:
        if column in df.columns:
            found_column = column
            break

    if found_column is None:
        raise ValueError(
            "No suitable text column found in the dataset."
        )

    TEXT_COLUMN = found_column

print(f"\nText column used: {TEXT_COLUMN}")

df["_text_length_chars"] = (
    df[TEXT_COLUMN]
    .fillna("")
    .astype(str)
    .str.len()
)

df["_text_length_words"] = (
    df[TEXT_COLUMN]
    .fillna("")
    .astype(str)
    .str.split()
    .str.len()
)


# ================================================================
# 10. TEXT LENGTH BY LANGUAGE
# ================================================================

print("\n" + "=" * 70)
print("10. TEXT LENGTH BY LANGUAGE")
print("=" * 70)

text_length_language = (
    df.groupby("language")
    .agg(
        Records=(TEXT_COLUMN, "count"),
        Mean_Characters=("_text_length_chars", "mean"),
        Median_Characters=("_text_length_chars", "median"),
        Mean_Words=("_text_length_words", "mean"),
        Median_Words=("_text_length_words", "median")
    )
    .reset_index()
)

print(
    text_length_language.round(2).to_string(index=False)
)

text_length_language.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "text_length_by_language.csv"
    ),
    index=False
)


# ================================================================
# 11. TEXT LENGTH BY LABEL
# ================================================================

print("\n" + "=" * 70)
print("11. TEXT LENGTH BY LABEL")
print("=" * 70)

text_length_label = (
    df.groupby("mapped_label")
    .agg(
        Records=(TEXT_COLUMN, "count"),
        Mean_Characters=("_text_length_chars", "mean"),
        Median_Characters=("_text_length_chars", "median"),
        Mean_Words=("_text_length_words", "mean"),
        Median_Words=("_text_length_words", "median")
    )
    .reset_index()
)

print(
    text_length_label.round(2).to_string(index=False)
)

text_length_label.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "text_length_by_label.csv"
    ),
    index=False
)


# ================================================================
# 12. DUPLICATE ANALYSIS
# ================================================================

print("\n" + "=" * 70)
print("12. DUPLICATE ANALYSIS")
print("=" * 70)

duplicate_count = df.duplicated(
    subset=[TEXT_COLUMN]
).sum()

duplicate_percentage = (
    duplicate_count / len(df) * 100
)

duplicate_summary = pd.DataFrame({
    "Metric": [
        "Total Records",
        "Duplicate Text Records",
        "Unique Text Records",
        "Duplicate Percentage"
    ],
    "Value": [
        len(df),
        duplicate_count,
        df[TEXT_COLUMN].nunique(),
        round(duplicate_percentage, 2)
    ]
})

print(
    duplicate_summary.to_string(index=False)
)

duplicate_summary.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "duplicate_analysis.csv"
    ),
    index=False
)


# ================================================================
# 13. LANGUAGE CLASS BALANCE
# ================================================================

print("\n" + "=" * 70)
print("13. LANGUAGE CLASS BALANCE")
print("=" * 70)

balance_rows = []

for language in sorted(df["language"].unique()):

    language_df = df[
        df["language"] == language
    ]

    total = len(language_df)

    offensive_count = (
        language_df["mapped_label"]
        .eq("Offensive")
        .sum()
    )

    non_offensive_count = (
        language_df["mapped_label"]
        .eq("Non_Offensive")
        .sum()
    )

    offensive_percentage = (
        offensive_count / total * 100
        if total > 0 else 0
    )

    non_offensive_percentage = (
        non_offensive_count / total * 100
        if total > 0 else 0
    )

    balance_rows.append({
        "Language": language,
        "Total": total,
        "Offensive": offensive_count,
        "Non_Offensive": non_offensive_count,
        "Offensive_Percentage": offensive_percentage,
        "Non_Offensive_Percentage": non_offensive_percentage
    })

language_balance = pd.DataFrame(balance_rows)

print(
    language_balance.round(2).to_string(index=False)
)

language_balance.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "language_class_balance.csv"
    ),
    index=False
)


# ================================================================
# 14. VISUALIZATION - LANGUAGE DISTRIBUTION
# ================================================================

print("\n" + "=" * 70)
print("14. GENERATING VISUALIZATIONS")
print("=" * 70)

plt.figure(figsize=(8, 5))

plt.bar(
    language_distribution["language"],
    language_distribution["count"]
)

plt.title(
    "Dravidian Dataset - Language Distribution"
)

plt.xlabel("Language")
plt.ylabel("Number of Records")

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "language_distribution.png"
    ),
    dpi=300
)

plt.close()

print("Language distribution plot generated.")


# ================================================================
# 15. VISUALIZATION - LABEL DISTRIBUTION
# ================================================================

plt.figure(figsize=(8, 5))

plt.bar(
    label_distribution["mapped_label"],
    label_distribution["count"]
)

plt.title(
    "Dravidian Dataset - Label Distribution"
)

plt.xlabel("Label")
plt.ylabel("Number of Records")

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "label_distribution.png"
    ),
    dpi=300
)

plt.close()

print("Label distribution plot generated.")


# ================================================================
# 16. VISUALIZATION - LANGUAGE × LABEL
# ================================================================

plot_data = pd.crosstab(
    df["language"],
    df["mapped_label"]
)

plt.figure(figsize=(9, 6))

plot_data.plot(
    kind="bar",
    figsize=(9, 6)
)

plt.title(
    "Offensive vs Non-Offensive Distribution by Language"
)

plt.xlabel("Language")
plt.ylabel("Number of Records")

plt.xticks(rotation=0)

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "language_label_distribution.png"
    ),
    dpi=300
)

plt.close()

print("Language × label distribution plot generated.")


# ================================================================
# 17. VISUALIZATION - LANGUAGE CLASS BALANCE
# ================================================================

plt.figure(figsize=(9, 6))

x = np.arange(
    len(language_balance)
)

width = 0.35

plt.bar(
    x - width / 2,
    language_balance["Offensive"],
    width,
    label="Offensive"
)

plt.bar(
    x + width / 2,
    language_balance["Non_Offensive"],
    width,
    label="Non-Offensive"
)

plt.xticks(
    x,
    language_balance["Language"]
)

plt.xlabel("Language")
plt.ylabel("Number of Records")

plt.title(
    "Class Distribution Across Dravidian Languages"
)

plt.legend()

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "language_class_balance.png"
    ),
    dpi=300
)

plt.close()

print("Language class balance plot generated.")


# ================================================================
# 18. VISUALIZATION - TEXT LENGTH
# ================================================================

plt.figure(figsize=(9, 6))

plt.bar(
    text_length_language["language"],
    text_length_language["Mean_Words"]
)

plt.title(
    "Average Text Length by Language"
)

plt.xlabel("Language")
plt.ylabel("Average Number of Words")

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "average_text_length_by_language.png"
    ),
    dpi=300
)

plt.close()

print("Text length plot generated.")


# ================================================================
# 19. SAVE DATASET SUMMARY
# ================================================================

print("\n" + "=" * 70)
print("19. FINAL DATASET SUMMARY")
print("=" * 70)

summary = pd.DataFrame({
    "Metric": [
        "Total Records",
        "Total Languages",
        "Total Labels",
        "Tamil Records",
        "Malayalam Records",
        "Kannada Records",
        "Offensive Records",
        "Non-Offensive Records",
        "Duplicate Text Records"
    ],
    "Value": [
        len(df),
        df["language"].nunique(),
        df["mapped_label"].nunique(),
        (df["language"] == "Tamil").sum(),
        (df["language"] == "Malayalam").sum(),
        (df["language"] == "Kannada").sum(),
        (df["mapped_label"] == "Offensive").sum(),
        (df["mapped_label"] == "Non_Offensive").sum(),
        duplicate_count
    ]
})

print(
    summary.to_string(index=False)
)

summary.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "dravidian_dataset_summary.csv"
    ),
    index=False
)


# ================================================================
# 20. REMOVE TEMPORARY COLUMNS
# ================================================================

df.drop(
    columns=[
        "_text_length_chars",
        "_text_length_words"
    ],
    inplace=True,
    errors="ignore"
)


# ================================================================
# 21. FINAL OUTPUT LIST
# ================================================================

print("\n" + "=" * 70)
print("STEP 18 - DRAVIDIAN DATASET ANALYSIS COMPLETED")
print("=" * 70)

print("\nAll results saved in:")
print(OUTPUT_DIR)

print("\nGenerated files:")

for filename in sorted(
    os.listdir(OUTPUT_DIR)
):
    print(f"- {filename}")

print("\n" + "=" * 70)
print("NO MODEL TRAINING WAS PERFORMED")
print("=" * 70)

print(
    "\nStep 18 analyzed the cleaned multilingual dataset "
    "across languages, labels, splits, class balance, "
    "text characteristics, and duplicates."
)

print(
    "\nThe dataset is now ready for the next multilingual "
    "modeling stage."
)

print("=" * 70)