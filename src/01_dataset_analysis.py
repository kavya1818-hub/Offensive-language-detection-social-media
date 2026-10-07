import pandas as pd
import os

print("=" * 60)
print("STEP 1 - DATASET ANALYSIS")
print("=" * 60)

DATA_PATH = "data/labeled_data.csv"

# ---------------------------------------------------------
# LOAD DATASET
# ---------------------------------------------------------

if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

df = pd.read_csv(DATA_PATH)

print("\nDataset loaded successfully.")
print(f"Records : {len(df)}")
print(f"Columns : {len(df.columns)}")

print("\nColumns:")
for col in df.columns:
    print(f"- {col}")

# ---------------------------------------------------------
# REQUIRED COLUMNS
# ---------------------------------------------------------

required_columns = ["tweet", "class"]

for col in required_columns:
    if col not in df.columns:
        raise ValueError(f"Required column missing: {col}")

print("\nRequired columns verified.")

# ---------------------------------------------------------
# DATASET INFORMATION
# ---------------------------------------------------------

print("\nDATA TYPES")
print("-" * 60)
print(df.dtypes)

# ---------------------------------------------------------
# MISSING VALUES
# ---------------------------------------------------------

print("\nMISSING VALUES")
print("-" * 60)
print(df.isnull().sum())

# ---------------------------------------------------------
# DUPLICATES
# ---------------------------------------------------------

duplicate_rows = df.duplicated().sum()
duplicate_tweets = df["tweet"].duplicated().sum()

print("\nDUPLICATE ANALYSIS")
print("-" * 60)
print(f"Duplicate rows   : {duplicate_rows}")
print(f"Duplicate tweets : {duplicate_tweets}")

# ---------------------------------------------------------
# CLASS DISTRIBUTION
# ---------------------------------------------------------

print("\nCLASS DISTRIBUTION")
print("-" * 60)

class_counts = df["class"].value_counts().sort_index()

class_names = {
    0: "Hate_Speech",
    1: "Offensive_Language",
    2: "Neither"
}

for label, count in class_counts.items():
    name = class_names.get(label, str(label))
    percentage = (count / len(df)) * 100
    print(f"{label} - {name:<20} : {count:>6} ({percentage:.2f}%)")

# ---------------------------------------------------------
# ANNOTATION INFORMATION
# ---------------------------------------------------------

print("\nANNOTATION INFORMATION")
print("-" * 60)

annotation_columns = [
    "count",
    "hate_speech",
    "offensive_language",
    "neither"
]

for col in annotation_columns:
    if col in df.columns:
        print(f"{col:<22}: available")

# ---------------------------------------------------------
# SAMPLE RECORDS
# ---------------------------------------------------------

print("\nSAMPLE RECORDS")
print("-" * 60)

for i in range(min(5, len(df))):
    row = df.iloc[i]

    label = class_names.get(row["class"], str(row["class"]))

    print(f"\nTweet {i + 1}:")
    print(row["tweet"])
    print(f"Label: {label}")

# ---------------------------------------------------------
# BASIC TEXT STATISTICS
# ---------------------------------------------------------

df["tweet_length"] = df["tweet"].astype(str).str.len()
df["word_count"] = df["tweet"].astype(str).str.split().str.len()

print("\nTEXT STATISTICS")
print("-" * 60)

print(f"Average characters : {df['tweet_length'].mean():.2f}")
print(f"Average words      : {df['word_count'].mean():.2f}")
print(f"Maximum characters : {df['tweet_length'].max()}")
print(f"Maximum words      : {df['word_count'].max()}")

# ---------------------------------------------------------
# SAVE ANALYSIS
# ---------------------------------------------------------

os.makedirs("results/eda", exist_ok=True)

class_distribution = pd.DataFrame({
    "class": class_counts.index,
    "class_name": [
        class_names.get(x, str(x))
        for x in class_counts.index
    ],
    "count": class_counts.values,
    "percentage": [
        round((x / len(df)) * 100, 2)
        for x in class_counts.values
    ]
})

class_distribution.to_csv(
    "results/eda/class_distribution.csv",
    index=False
)

print("\nGenerated file:")
print("- results/eda/class_distribution.csv")

print("\n" + "=" * 60)
print("DATASET ANALYSIS COMPLETED")
print("=" * 60)