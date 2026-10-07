import os
import pandas as pd

print("=" * 70)
print("STEP 16 - DRAVIDIAN DATASET ANALYSIS")
print("=" * 70)

DATA_DIR = "data/dravidian"
OUTPUT_DIR = "results/dravidian_analysis"

os.makedirs(OUTPUT_DIR, exist_ok=True)

files = [
    "kannada_offensive.csv",
    "kannada_offensive_train.csv",
    "kannada_offensive_dev.csv",
    "kannada_offensive_test.csv",
    "mal_full_offensive.csv",
    "mal_full_offensive_train.csv",
    "mal_full_offensive_dev.csv",
    "mal_full_offensive_test.csv",
    "tamil_offensive_full.csv",
    "tamil_offensive_full_train.csv",
    "tamil_offensive_full_dev.csv",
    "tamil_offensive_full_test.csv"
]

all_results = []

print("\nDATASET LOCATION")
print("-" * 70)
print(os.path.abspath(DATA_DIR))

print("\n" + "=" * 70)
print("FILE-BY-FILE ANALYSIS")
print("=" * 70)

for filename in files:

    path = os.path.join(DATA_DIR, filename)

    print("\n" + "-" * 70)
    print(filename)
    print("-" * 70)

    if not os.path.exists(path):
        print("FILE NOT FOUND")
        continue

    try:
        df = pd.read_csv(path)

        print("Rows       :", len(df))
        print("Columns    :", len(df.columns))
        print("Column names:")
        print(list(df.columns))

        print("\nFirst 3 records:")
        print(df.head(3).to_string(index=False))

        print("\nMissing values:")
        print(df.isnull().sum().to_string())

        print("\nData types:")
        print(df.dtypes.to_string())

        print("\nPossible label columns:")

        for col in df.columns:
            unique_count = df[col].nunique(dropna=True)

            if unique_count <= 30:
                print(
                    f"  {col}: "
                    f"{unique_count} unique values"
                )

                try:
                    print(df[col].value_counts(dropna=False).to_string())
                except Exception:
                    pass

        all_results.append({
            "File": filename,
            "Rows": len(df),
            "Columns": len(df.columns),
            "Column_Names": " | ".join(df.columns.astype(str))
        })

    except Exception as e:
        print("ERROR:", e)


print("\n" + "=" * 70)
print("DATASET SUMMARY")
print("=" * 70)

summary_df = pd.DataFrame(all_results)

if not summary_df.empty:
    print(summary_df.to_string(index=False))

    summary_path = os.path.join(
        OUTPUT_DIR,
        "dravidian_file_summary.csv"
    )

    summary_df.to_csv(summary_path, index=False)

    print("\nSummary saved:")
    print(summary_path)


print("\n" + "=" * 70)
print("CHECKING FULL DATASETS VS SPLITS")
print("=" * 70)

language_groups = {
    "Kannada": [
        "kannada_offensive.csv",
        "kannada_offensive_train.csv",
        "kannada_offensive_dev.csv",
        "kannada_offensive_test.csv"
    ],
    "Malayalam": [
        "mal_full_offensive.csv",
        "mal_full_offensive_train.csv",
        "mal_full_offensive_dev.csv",
        "mal_full_offensive_test.csv"
    ],
    "Tamil": [
        "tamil_offensive_full.csv",
        "tamil_offensive_full_train.csv",
        "tamil_offensive_full_dev.csv",
        "tamil_offensive_full_test.csv"
    ]
}

for language, language_files in language_groups.items():

    print("\n" + "-" * 70)
    print(language)
    print("-" * 70)

    for filename in language_files:

        path = os.path.join(DATA_DIR, filename)

        if not os.path.exists(path):
            continue

        try:
            df = pd.read_csv(path)

            print(
                f"{filename:<40} "
                f"{len(df):>7} rows"
            )

        except Exception as e:
            print(filename, "ERROR:", e)


print("\n" + "=" * 70)
print("STEP 16 - DRAVIDIAN DATASET ANALYSIS COMPLETED")
print("=" * 70)

print("\nGenerated:")
print("- results/dravidian_analysis/dravidian_file_summary.csv")

print("\nIMPORTANT:")
print("No files were modified.")
print("No datasets were merged.")
print("No preprocessing was performed.")
print("No model training was performed.")

print("\nNext step will be decided from the actual dataset structure.")