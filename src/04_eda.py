import pandas as pd
import os
import re
import matplotlib.pyplot as plt

print("=" * 60)
print("STEP 4 - EXPLORATORY DATA ANALYSIS")
print("=" * 60)

# ============================================================
# 1. LOAD PROCESSED DATA
# ============================================================

INPUT_PATH = "data/processed/processed_dataset.csv"

df = pd.read_csv(INPUT_PATH)

print("\nDataset loaded successfully.")
print(f"Rows    : {len(df)}")
print(f"Columns : {len(df.columns)}")


# ============================================================
# 2. BASIC INFORMATION
# ============================================================

print("\n" + "=" * 60)
print("BASIC DATASET INFORMATION")
print("=" * 60)

print("\nColumns:")
print(df.columns.tolist())

print("\nData Types:")
print(df.dtypes)

print("\nMissing Values:")
print(df.isnull().sum())

print("\nDuplicate Rows:")
print(df.duplicated().sum())


# ============================================================
# 3. CLASS DISTRIBUTION
# ============================================================

print("\n" + "=" * 60)
print("CLASS DISTRIBUTION")
print("=" * 60)

class_counts = df["label"].value_counts()

print(class_counts)

class_percent = (
    df["label"]
    .value_counts(normalize=True)
    .mul(100)
    .round(2)
)

print("\nPercentage:")
print(class_percent)


# ============================================================
# 4. PLATFORM / SOURCE INFORMATION
# ============================================================

# This dataset does not contain an actual platform column.
# Therefore we do not invent one.

print("\n" + "=" * 60)
print("SOURCE INFORMATION")
print("=" * 60)

print(
    "\nPlatform information is not available "
    "as a dataset column."
)


# ============================================================
# 5. TEXT LENGTH ANALYSIS
# ============================================================

df["character_count"] = (
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

print("\n" + "=" * 60)
print("TEXT LENGTH ANALYSIS")
print("=" * 60)

print("\nCharacter Statistics:")
print(
    df["character_count"]
    .describe()
)

print("\nWord Statistics:")
print(
    df["word_count"]
    .describe()
)


# ============================================================
# 6. TEXT LENGTH BY CLASS
# ============================================================

print("\n" + "=" * 60)
print("TEXT LENGTH BY CLASS")
print("=" * 60)

length_by_class = (
    df.groupby("label")
    .agg(
        Average_Characters=("character_count", "mean"),
        Median_Characters=("character_count", "median"),
        Average_Words=("word_count", "mean"),
        Median_Words=("word_count", "median")
    )
    .round(2)
)

print(length_by_class)


# ============================================================
# 7. OFFENSIVE VS NON-OFFENSIVE
# ============================================================

df["binary_category"] = df["label"].apply(
    lambda x:
        "Non_Offensive"
        if x == "Neither"
        else "Offensive"
)

print("\n" + "=" * 60)
print("OFFENSIVE VS NON-OFFENSIVE")
print("=" * 60)

binary_counts = (
    df["binary_category"]
    .value_counts()
)

print(binary_counts)

binary_percent = (
    df["binary_category"]
    .value_counts(normalize=True)
    .mul(100)
    .round(2)
)

print("\nPercentage:")
print(binary_percent)


# ============================================================
# 8. MOST FREQUENT WORDS
# ============================================================

print("\n" + "=" * 60)
print("MOST FREQUENT WORDS")
print("=" * 60)

all_text = " ".join(
    df["normalized_text"]
    .astype(str)
)

words = all_text.split()

# Basic stopwords
stopwords = {
    "the", "a", "an", "and", "or", "to",
    "of", "in", "is", "it", "for", "on",
    "you", "i", "this", "that", "with",
    "my", "me", "your", "are", "be",
    "was", "as", "at", "but", "have",
    "has", "from", "they", "we", "he",
    "she", "his", "her"
}

filtered_words = [
    word
    for word in words
    if word not in stopwords
    and len(word) > 2
]

word_counts = (
    pd.Series(filtered_words)
    .value_counts()
    .head(20)
)

print(word_counts)


# ============================================================
# 9. CLASS-SPECIFIC TOP WORDS
# ============================================================

print("\n" + "=" * 60)
print("CLASS-SPECIFIC WORD ANALYSIS")
print("=" * 60)

class_word_records = []

for label in sorted(df["label"].unique()):

    class_text = " ".join(
        df.loc[
            df["label"] == label,
            "normalized_text"
        ].astype(str)
    )

    class_words = class_text.split()

    class_words = [
        word
        for word in class_words
        if word not in stopwords
        and len(word) > 2
    ]

    top_words = (
        pd.Series(class_words)
        .value_counts()
        .head(10)
    )

    print(f"\n{label}:")
    print(top_words)

    for word, count in top_words.items():

        class_word_records.append({
            "label": label,
            "word": word,
            "frequency": count
        })


class_words_df = pd.DataFrame(
    class_word_records
)


# ============================================================
# 10. RETWEET ANALYSIS
# ============================================================

df["contains_rt"] = (
    df["tweet"]
    .astype(str)
    .str.contains(
        r"\brt\b",
        case=False,
        regex=True
    )
)

print("\n" + "=" * 60)
print("RETWEET ANALYSIS")
print("=" * 60)

rt_count = df["contains_rt"].sum()

print(
    f"\nTweets containing RT : {rt_count}"
)

print(
    f"Tweets without RT    : {len(df) - rt_count}"
)


# ============================================================
# 11. USERNAME / URL ANALYSIS
# ============================================================

df["contains_username"] = (
    df["tweet"]
    .astype(str)
    .str.contains(
        r"@\w+",
        regex=True
    )
)

df["contains_url"] = (
    df["tweet"]
    .astype(str)
    .str.contains(
        r"http\S+|www\S+",
        regex=True
    )
)

print("\n" + "=" * 60)
print("SOCIAL MEDIA CHARACTERISTICS")
print("=" * 60)

print(
    f"\nTweets containing usernames : "
    f"{df['contains_username'].sum()}"
)

print(
    f"Tweets containing URLs      : "
    f"{df['contains_url'].sum()}"
)


# ============================================================
# 12. CREATE RESULTS DIRECTORY
# ============================================================

os.makedirs(
    "results/eda",
    exist_ok=True
)


# ============================================================
# 13. SAVE NUMERICAL RESULTS
# ============================================================

class_distribution_df = (
    class_counts
    .rename("count")
    .reset_index()
)

class_distribution_df.columns = [
    "label",
    "count"
]

class_distribution_df[
    "percentage"
] = (
    class_distribution_df["count"]
    / len(df)
    * 100
).round(2)


class_distribution_df.to_csv(
    "results/eda/class_distribution.csv",
    index=False
)


length_by_class.to_csv(
    "results/eda/text_length_by_class.csv"
)


class_words_df.to_csv(
    "results/eda/class_specific_words.csv",
    index=False
)


# ============================================================
# 14. SAVE SUMMARY
# ============================================================

summary = pd.DataFrame({

    "Metric": [
        "Total Records",
        "Number of Classes",
        "Offensive Records",
        "Non-Offensive Records",
        "Average Characters",
        "Median Characters",
        "Average Words",
        "Median Words",
        "Tweets Containing RT",
        "Tweets Containing Username",
        "Tweets Containing URL"
    ],

    "Value": [
        len(df),
        df["label"].nunique(),
        int(
            (df["binary_category"] == "Offensive").sum()
        ),
        int(
            (df["binary_category"] == "Non_Offensive").sum()
        ),
        round(df["character_count"].mean(), 2),
        round(df["character_count"].median(), 2),
        round(df["word_count"].mean(), 2),
        round(df["word_count"].median(), 2),
        int(df["contains_rt"].sum()),
        int(df["contains_username"].sum()),
        int(df["contains_url"].sum())
    ]
})


summary.to_csv(
    "results/eda/eda_summary.csv",
    index=False
)


# ============================================================
# 15. CLASS DISTRIBUTION GRAPH
# ============================================================

plt.figure(figsize=(9, 6))

class_counts.plot(
    kind="bar"
)

plt.title(
    "Class Distribution"
)

plt.xlabel(
    "Class"
)

plt.ylabel(
    "Number of Tweets"
)

plt.xticks(
    rotation=20,
    ha="right"
)

plt.tight_layout()

plt.savefig(
    "results/eda/class_distribution.png",
    dpi=300
)

plt.close()


# ============================================================
# 16. TEXT LENGTH DISTRIBUTION
# ============================================================

plt.figure(figsize=(9, 6))

plt.hist(
    df["word_count"],
    bins=30
)

plt.title(
    "Tweet Word Count Distribution"
)

plt.xlabel(
    "Number of Words"
)

plt.ylabel(
    "Frequency"
)

plt.tight_layout()

plt.savefig(
    "results/eda/word_count_distribution.png",
    dpi=300
)

plt.close()


# ============================================================
# 17. TEXT LENGTH BY CLASS GRAPH
# ============================================================

plt.figure(figsize=(10, 6))

df.boxplot(
    column="word_count",
    by="label"
)

plt.title(
    "Tweet Length by Class"
)

plt.suptitle("")

plt.xlabel(
    "Class"
)

plt.ylabel(
    "Word Count"
)

plt.xticks(
    rotation=20
)

plt.tight_layout()

plt.savefig(
    "results/eda/word_count_by_class.png",
    dpi=300
)

plt.close()


# ============================================================
# 18. SAMPLE RECORDS
# ============================================================

print("\n" + "=" * 60)
print("SAMPLE RECORDS")
print("=" * 60)

sample_columns = [
    "tweet",
    "normalized_text",
    "label"
]

print(
    df[
        sample_columns
    ].head(10).to_string(
        index=False
    )
)


# ============================================================
# 19. FINAL OUTPUT
# ============================================================

print("\n" + "=" * 60)
print("EDA OUTPUT")
print("=" * 60)

print("\nGenerated files:")

print(
    "- results/eda/class_distribution.csv"
)

print(
    "- results/eda/text_length_by_class.csv"
)

print(
    "- results/eda/class_specific_words.csv"
)

print(
    "- results/eda/eda_summary.csv"
)

print(
    "- results/eda/class_distribution.png"
)

print(
    "- results/eda/word_count_distribution.png"
)

print(
    "- results/eda/word_count_by_class.png"
)

print("\n" + "=" * 60)
print("EXPLORATORY DATA ANALYSIS COMPLETED")
print("=" * 60)