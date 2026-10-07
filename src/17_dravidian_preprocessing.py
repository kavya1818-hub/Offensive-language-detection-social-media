import os
import csv
import pandas as pd

print("=" * 75)
print("STEP 17 - DRAVIDIAN DATASET PREPROCESSING")
print("=" * 75)

DATA_DIR = "data/dravidian"
OUTPUT_DIR = "data/dravidian_clean"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ------------------------------------------------------------
# FILE CONFIGURATION
# ------------------------------------------------------------

DATASETS = {
    "Kannada": {
        "train": "kannada_offensive_train.csv",
        "dev": "kannada_offensive_dev.csv",
        "test": "kannada_offensive_test.csv"
    },
    "Malayalam": {
        "train": "mal_full_offensive_train.csv",
        "dev": "mal_full_offensive_dev.csv",
        "test": "mal_full_offensive_test.csv"
    },
    "Tamil": {
        "train": "tamil_offensive_full_train.csv",
        "dev": "tamil_offensive_full_dev.csv",
        "test": "tamil_offensive_full_test.csv"
    }
}


# ------------------------------------------------------------
# LABEL MAPPING
# ------------------------------------------------------------

def map_label(label):

    if pd.isna(label):
        return None

    label = str(label).strip()

    # Non-offensive
    if label.lower() == "not_offensive":
        return "Non_Offensive"

    # Offensive categories
    if label.lower().startswith("offensive_"):
        return "Offensive"

    # Language-mismatch records
    if label.lower() in [
        "not-kannada",
        "not-malayalam",
        "not-tamil"
    ]:
        return None

    return None


# ------------------------------------------------------------
# READ FILE ROBUSTLY
# ------------------------------------------------------------

def read_dravidian_file(path):

    rows = []

    with open(
        path,
        "r",
        encoding="utf-8-sig",
        errors="replace"
    ) as f:

        for line_number, raw_line in enumerate(f, start=1):

            line = raw_line.rstrip("\r\n")

            if not line.strip():
                continue

            text = None
            label = None

            # ------------------------------------------------
            # CASE 1: TAB-SEPARATED
            # ------------------------------------------------

            if "\t" in line:

                parts = line.split("\t")

                # Label is normally the final non-empty field.
                parts = [p.strip() for p in parts]

                while parts and parts[-1] == "":
                    parts.pop()

                if len(parts) >= 2:

                    label = parts[-1]

                    # Everything before label is text.
                    text = "\t".join(parts[:-1]).strip()

            # ------------------------------------------------
            # CASE 2: COMMA-SEPARATED / CSV
            # ------------------------------------------------

            if text is None:

                try:

                    parsed = next(
                        csv.reader([line])
                    )

                    if len(parsed) >= 2:

                        # Most reliable assumption:
                        # final field = label
                        label = parsed[-1].strip()

                        text = ",".join(
                            parsed[:-1]
                        ).strip()

                except Exception:
                    pass

            # ------------------------------------------------
            # IF PARSING FAILED
            # ------------------------------------------------

            if text is None or label is None:

                print(
                    f"Warning: Could not parse line "
                    f"{line_number} in {os.path.basename(path)}"
                )

                continue

            # Remove surrounding quotes
            text = text.strip().strip('"').strip("'")
            label = label.strip().strip('"').strip("'")

            if not text:
                continue

            rows.append({
                "text": text,
                "original_label": label
            })

    return pd.DataFrame(rows)


# ------------------------------------------------------------
# PROCESS ONE FILE
# ------------------------------------------------------------

def process_file(language, split, filename):

    path = os.path.join(DATA_DIR, filename)

    print("\n" + "-" * 75)
    print(f"{language} - {split.upper()}")
    print("-" * 75)

    print("Reading:")
    print(path)

    if not os.path.exists(path):

        print("FILE NOT FOUND")
        return None

    df = read_dravidian_file(path)

    print("Parsed records:", len(df))

    if df.empty:

        print("No valid records found.")
        return None

    print("\nOriginal label distribution:")

    print(
        df["original_label"]
        .value_counts(dropna=False)
        .to_string()
    )

    # --------------------------------------------------------
    # LANGUAGE-SPECIFIC INVALID RECORDS
    # --------------------------------------------------------

    before = len(df)

    df["mapped_label"] = df["original_label"].apply(map_label)

    removed = df["mapped_label"].isna().sum()

    df = df.dropna(
        subset=["mapped_label"]
    ).copy()

    print("\nRecords removed:", removed)
    print("Records retained:", len(df))

    # --------------------------------------------------------
    # ADD LANGUAGE AND SPLIT
    # --------------------------------------------------------

    df["language"] = language
    df["split"] = split

    # Standard column order
    df = df[
        [
            "text",
            "original_label",
            "mapped_label",
            "language",
            "split"
        ]
    ]

    print("\nMapped label distribution:")

    print(
        df["mapped_label"]
        .value_counts()
        .to_string()
    )

    # --------------------------------------------------------
    # SAVE
    # --------------------------------------------------------

    output_name = (
        f"{language.lower()}_{split}_clean.csv"
    )

    output_path = os.path.join(
        OUTPUT_DIR,
        output_name
    )

    df.to_csv(
        output_path,
        index=False,
        encoding="utf-8-sig"
    )

    print("\nSaved:")
    print(output_path)

    return df


# ------------------------------------------------------------
# PROCESS ALL DATASETS
# ------------------------------------------------------------

all_data = []

for language, splits in DATASETS.items():

    print("\n" + "=" * 75)
    print(language.upper())
    print("=" * 75)

    for split, filename in splits.items():

        df = process_file(
            language,
            split,
            filename
        )

        if df is not None:
            all_data.append(df)


# ------------------------------------------------------------
# COMBINE CLEAN DATA
# ------------------------------------------------------------

print("\n" + "=" * 75)
print("CREATING COMBINED DATASET")
print("=" * 75)

if all_data:

    combined = pd.concat(
        all_data,
        ignore_index=True
    )

    combined_path = os.path.join(
        OUTPUT_DIR,
        "dravidian_multilingual_clean.csv"
    )

    combined.to_csv(
        combined_path,
        index=False,
        encoding="utf-8-sig"
    )

    print("Total records:", len(combined))

    print("\nLanguage distribution:")
    print(
        combined["language"]
        .value_counts()
        .to_string()
    )

    print("\nLabel distribution:")
    print(
        combined["mapped_label"]
        .value_counts()
        .to_string()
    )

    print("\nLanguage × Label:")
    print(
        pd.crosstab(
            combined["language"],
            combined["mapped_label"]
        ).to_string()
    )

    print("\nSplit distribution:")
    print(
        pd.crosstab(
            combined["language"],
            combined["split"]
        ).to_string()
    )

    print("\nSaved:")
    print(combined_path)

else:

    print("No valid datasets were created.")


# ------------------------------------------------------------
# FINAL MESSAGE
# ------------------------------------------------------------

print("\n" + "=" * 75)
print("STEP 17 - DRAVIDIAN PREPROCESSING COMPLETED")
print("=" * 75)

print("\nOriginal files were NOT modified.")

print("\nClean dataset directory:")
print(OUTPUT_DIR)

print("\nPurpose of this preprocessing:")
print("1. Fix delimiter/CSV formatting problems")
print("2. Standardize text and labels")
print("3. Remove language-mismatch records")
print("4. Create common Offensive vs Non_Offensive labels")
print("5. Preserve original labels")
print("6. Preserve language information")
print("7. Preserve train/dev/test splits")

print("\nNo model training performed.")