import pandas as pd
import re
import os

print("=" * 60)
print("STEP 2 - DUPLICATE & DATA LEAKAGE CHECK")
print("=" * 60)

# ------------------------------------------------------------
# 1. LOAD DATASET
# ------------------------------------------------------------

DATA_PATH = "data/labeled_data.csv"

df = pd.read_csv(DATA_PATH)

print("\nDataset loaded successfully.")
print(f"Records : {len(df)}")
print(f"Columns : {len(df.columns)}")

TEXT_COLUMN = "tweet"
LABEL_COLUMN = "class"

# ------------------------------------------------------------
# 2. BASIC DUPLICATE CHECK
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("EXACT DUPLICATE CHECK")
print("=" * 60)

duplicate_rows = df.duplicated().sum()
duplicate_tweets = df[TEXT_COLUMN].duplicated().sum()

print(f"\nDuplicate rows   : {duplicate_rows}")
print(f"Duplicate tweets : {duplicate_tweets}")

# ------------------------------------------------------------
# 3. NORMALIZE TEXT
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("NORMALIZED TEXT ANALYSIS")
print("=" * 60)


def normalize_text(text):
    text = str(text).lower()

    # Remove RT prefix
    text = re.sub(r"\brt\b", " ", text)

    # Remove usernames
    text = re.sub(r"@\w+", " ", text)

    # Remove URLs
    text = re.sub(r"http\S+|www\S+", " ", text)

    # Decode common HTML entities
    text = text.replace("&amp;", "and")
    text = text.replace("&lt;", " ")
    text = text.replace("&gt;", " ")

    # Remove punctuation
    text = re.sub(r"[^a-z0-9\s]", " ", text)

    # Normalize spaces
    text = re.sub(r"\s+", " ", text).strip()

    return text


df["normalized_text"] = df[TEXT_COLUMN].apply(normalize_text)

normalized_duplicates = df["normalized_text"].duplicated().sum()

print(f"\nNormalized duplicate records : {normalized_duplicates}")

unique_normalized = df["normalized_text"].nunique()

print(f"Unique normalized texts      : {unique_normalized}")

# ------------------------------------------------------------
# 4. FIND NORMALIZED TEXTS WITH DIFFERENT LABELS
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("CONFLICTING LABEL ANALYSIS")
print("=" * 60)

label_counts = (
    df.groupby("normalized_text")[LABEL_COLUMN]
    .nunique()
)

conflicting_texts = label_counts[label_counts > 1]

print(
    f"\nNormalized texts appearing with different labels : "
    f"{len(conflicting_texts)}"
)

# Save conflicting records
os.makedirs("results/leakage_analysis", exist_ok=True)

if len(conflicting_texts) > 0:

    conflict_df = df[
        df["normalized_text"].isin(conflicting_texts.index)
    ].sort_values("normalized_text")

    conflict_df.to_csv(
        "results/leakage_analysis/conflicting_labels.csv",
        index=False
    )

    print(
        "Saved: results/leakage_analysis/conflicting_labels.csv"
    )

else:
    print("No conflicting labels found.")

# ------------------------------------------------------------
# 5. MOST REPEATED NORMALIZED TEXTS
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("MOST REPEATED NORMALIZED TEXTS")
print("=" * 60)

repeated = (
    df["normalized_text"]
    .value_counts()
    .reset_index()
)

repeated.columns = ["normalized_text", "count"]

repeated = repeated[repeated["count"] > 1]

print(f"\nRepeated normalized texts : {len(repeated)}")

repeated.to_csv(
    "results/leakage_analysis/repeated_normalized_texts.csv",
    index=False
)

print(
    "Saved: results/leakage_analysis/repeated_normalized_texts.csv"
)

# ------------------------------------------------------------
# 6. RETWEET / TEMPLATE ANALYSIS
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("RETWEET / TEMPLATE ANALYSIS")
print("=" * 60)

retweet_count = df[TEXT_COLUMN].astype(str).str.contains(
    r"\brt\b", 
    case=False,
    regex=True
).sum()

print(f"\nRecords containing RT : {retweet_count}")

# ------------------------------------------------------------
# 7. CLASS DISTRIBUTION
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("CLASS DISTRIBUTION")
print("=" * 60)

class_distribution = (
    df[LABEL_COLUMN]
    .value_counts()
    .sort_index()
)

print("\n", class_distribution)

# ------------------------------------------------------------
# 8. SAVE SUMMARY
# ------------------------------------------------------------

summary = pd.DataFrame({
    "Metric": [
        "Total Records",
        "Exact Duplicate Rows",
        "Exact Duplicate Tweets",
        "Unique Normalized Texts",
        "Repeated Normalized Texts",
        "Conflicting Label Texts",
        "Records Containing RT"
    ],
    "Value": [
        len(df),
        duplicate_rows,
        duplicate_tweets,
        unique_normalized,
        len(repeated),
        len(conflicting_texts),
        retweet_count
    ]
})

summary.to_csv(
    "results/leakage_analysis/leakage_summary.csv",
    index=False
)

print("\nSummary saved:")
print("- results/leakage_analysis/leakage_summary.csv")

print("\n" + "=" * 60)
print("STEP 2 - DUPLICATE & DATA LEAKAGE CHECK COMPLETED")
print("=" * 60)