"""
======================================================================
STEP 30 — EMOJI FEATURE ANALYSIS
======================================================================

Purpose:
    Analyze whether emojis provide useful information for offensive
    language detection in the Dravidian multilingual dataset.

IMPORTANT:
    - This is a SEPARATE emoji study.
    - It does NOT modify Step 20, Step 24, Step 26, or final results.
    - It does NOT train any model.
    - It does NOT change the multilingual final pipeline.
    - Results are saved ONLY under:
        results/emoji_analysis/step30_emoji_analysis

Dataset:
    data/dravidian_clean/dravidian_multilingual_clean.csv

Analyses:
    1. Emoji prevalence
    2. Emoji distribution by language
    3. Emoji distribution by label
    4. Offensive rate with/without emojis
    5. Most frequent emojis
    6. Most frequent emojis by language
    7. Most frequent emojis by class
    8. Emoji count per comment
    9. Emoji density
   10. Repeated emoji patterns
   11. Emoji diversity
   12. Language-wise emoji statistics
   13. Candidate emoji features
   14. Statistical association between emoji presence and label
   15. CSV reports
   16. Visualization plots
   17. Final text report

NO MODEL TRAINING.
======================================================================
"""

import os
import re
import unicodedata
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    from scipy.stats import chi2_contingency
    SCIPY_AVAILABLE = True
except Exception:
    SCIPY_AVAILABLE = False


# ======================================================================
# CONFIGURATION
# ======================================================================

ROOT = r"E:\Offensive Language Detection in Social Media_NLP"

DATA_FILE = os.path.join(
    ROOT,
    "data",
    "dravidian_clean",
    "dravidian_multilingual_clean.csv"
)

# Separate folder — DO NOT TOUCH multilingual_cross_dataset_final
OUT_DIR = os.path.join(
    ROOT,
    "results",
    "emoji_analysis",
    "step30_emoji_analysis"
)

os.makedirs(OUT_DIR, exist_ok=True)


# ======================================================================
# EMOJI DETECTION
# ======================================================================

# Main Unicode ranges containing commonly used emojis/symbols.
EMOJI_RANGES = [
    (0x1F000, 0x1FAFF),
    (0x1FC00, 0x1FFFD),
    (0x2600, 0x27BF),
    (0x2300, 0x23FF),
    (0x2B00, 0x2BFF),
    (0x2934, 0x2935),
    (0x3030, 0x303D),
    (0x3297, 0x3299),
]

# Characters that commonly participate in emoji sequences.
EMOJI_COMPONENTS = {
    0x200D,  # ZERO WIDTH JOINER
    0xFE0E,  # text presentation
    0xFE0F,  # emoji presentation
}


def is_emoji_codepoint(ch):
    """
    Detect whether a Unicode code point belongs to a common emoji range.
    """

    cp = ord(ch)

    for start, end in EMOJI_RANGES:
        if start <= cp <= end:
            return True

    # Regional indicator symbols used in flags
    if 0x1F1E6 <= cp <= 0x1F1FF:
        return True

    # Keycap base characters are handled through sequence logic.
    return False


def extract_emoji_codepoints(text):
    """
    Extract emoji-bearing Unicode characters from text.

    This function intentionally keeps the analysis broad so that
    multilingual social-media emoji usage is not lost.
    """

    if not isinstance(text, str):
        return []

    result = []

    for ch in text:
        if is_emoji_codepoint(ch):
            result.append(ch)

    return result


def extract_emoji_clusters(text):
    """
    Attempt to preserve multi-codepoint emoji sequences.

    Examples:
        ❤️
        😂
        🤦‍♂️
        👩‍💻
        👍🏽
        🇮🇳

    The regex package provides Unicode grapheme cluster support.
    If unavailable, a simpler codepoint-based fallback is used.
    """

    if not isinstance(text, str):
        return []

    try:
        import regex

        clusters = regex.findall(r"\X", text)

        emoji_clusters = []

        for cluster in clusters:
            if any(is_emoji_codepoint(ch) for ch in cluster):
                emoji_clusters.append(cluster)

        return emoji_clusters

    except Exception:
        # Fallback: individual emoji codepoints
        return extract_emoji_codepoints(text)


def contains_emoji(text):
    return len(extract_emoji_codepoints(text)) > 0


# ======================================================================
# TEXT FEATURES
# ======================================================================

def calculate_features(text):
    """
    Calculate emoji-related features for one comment.
    """

    if not isinstance(text, str):
        text = ""

    emojis = extract_emoji_codepoints(text)
    clusters = extract_emoji_clusters(text)

    emoji_count = len(emojis)

    unique_emojis = len(set(emojis))

    text_length = len(text)

    emoji_density = (
        emoji_count / text_length
        if text_length > 0
        else 0.0
    )

    # Count repeated emoji usage.
    counts = Counter(emojis)

    repeated_emoji_count = sum(
        count - 1
        for count in counts.values()
        if count > 1
    )

    has_repeated_emoji = int(repeated_emoji_count > 0)

    max_repeat = (
        max(counts.values())
        if counts
        else 0
    )

    # Number of emoji clusters.
    cluster_count = len(clusters)

    unique_clusters = len(set(clusters))

    return {
        "emoji_count": emoji_count,
        "unique_emoji_count": unique_emojis,
        "emoji_density": emoji_density,
        "repeated_emoji_count": repeated_emoji_count,
        "has_repeated_emoji": has_repeated_emoji,
        "max_emoji_repetition": max_repeat,
        "emoji_cluster_count": cluster_count,
        "unique_emoji_cluster_count": unique_clusters,
        "has_emoji": int(emoji_count > 0),
    }


# ======================================================================
# LABEL NORMALIZATION
# ======================================================================

def normalize_label(value):
    """
    Normalize dataset labels to:
        0 = Non-Offensive
        1 = Offensive
    """

    if pd.isna(value):
        return np.nan

    text = str(value).strip().lower()

    if text in {
        "0",
        "non_offensive",
        "non-offensive",
        "non offensive",
        "not offensive",
        "nonoffensive",
    }:
        return 0

    if text in {
        "1",
        "offensive",
    }:
        return 1

    return np.nan


def normalize_language(value):
    if pd.isna(value):
        return "Unknown"

    text = str(value).strip().lower()

    mapping = {
        "tamil": "Tamil",
        "malayalam": "Malayalam",
        "kannada": "Kannada",
    }

    return mapping.get(text, str(value).strip())


# ======================================================================
# DISPLAY HELPERS
# ======================================================================

def print_section(title):
    print()
    print("=" * 78)
    print(title)
    print("=" * 78)


def save_plot(path):
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()


# ======================================================================
# LOAD DATA
# ======================================================================

print("=" * 78)
print("STEP 30 — EMOJI FEATURE ANALYSIS")
print("=" * 78)

print()
print("Dataset:")
print(DATA_FILE)

print()
print("Output:")
print(OUT_DIR)

if not os.path.exists(DATA_FILE):
    raise FileNotFoundError(
        f"\nDataset not found:\n{DATA_FILE}\n\n"
        "Check the DATA_FILE path in the script."
    )

df = pd.read_csv(DATA_FILE)

print()
print("Raw shape:", df.shape)
print("Columns:", list(df.columns))


# ======================================================================
# FIND REQUIRED COLUMNS
# ======================================================================

text_candidates = [
    "text",
    "comment",
    "sentence",
    "content"
]

label_candidates = [
    "mapped_label",
    "label",
    "target"
]

language_candidates = [
    "language",
    "lang"
]

TEXT_COL = next(
    (c for c in text_candidates if c in df.columns),
    None
)

LABEL_COL = next(
    (c for c in label_candidates if c in df.columns),
    None
)

LANG_COL = next(
    (c for c in language_candidates if c in df.columns),
    None
)

if TEXT_COL is None:
    raise ValueError(
        f"Could not find text column. Available columns: {list(df.columns)}"
    )

if LABEL_COL is None:
    raise ValueError(
        f"Could not find label column. Available columns: {list(df.columns)}"
    )

if LANG_COL is None:
    raise ValueError(
        f"Could not find language column. Available columns: {list(df.columns)}"
    )


print()
print("Using columns:")
print("Text     :", TEXT_COL)
print("Label    :", LABEL_COL)
print("Language :", LANG_COL)


# ======================================================================
# CLEAN BASIC FIELDS
# ======================================================================

df = df.copy()

df[TEXT_COL] = df[TEXT_COL].fillna("").astype(str)
df["label_id"] = df[LABEL_COL].apply(normalize_label)
df["language_clean"] = df[LANG_COL].apply(normalize_language)

df = df.dropna(subset=["label_id"]).copy()
df["label_id"] = df["label_id"].astype(int)

df["label_name"] = df["label_id"].map({
    0: "Non-Offensive",
    1: "Offensive"
})


# ======================================================================
# REMOVE EXACT DUPLICATES FOR ANALYSIS CONSISTENCY
# ======================================================================

before_dedup = len(df)

df = df.drop_duplicates(
    subset=[TEXT_COL],
    keep="first"
).reset_index(drop=True)

after_dedup = len(df)

print()
print("Records before duplicate removal:", before_dedup)
print("Records after duplicate removal :", after_dedup)
print("Duplicates removed              :", before_dedup - after_dedup)


# ======================================================================
# CALCULATE EMOJI FEATURES
# ======================================================================

print_section("CALCULATING EMOJI FEATURES")

feature_records = []

for text in df[TEXT_COL]:
    feature_records.append(
        calculate_features(text)
    )

feature_df = pd.DataFrame(feature_records)

df = pd.concat(
    [
        df.reset_index(drop=True),
        feature_df.reset_index(drop=True)
    ],
    axis=1
)

print("Emoji feature extraction completed.")


# ======================================================================
# BASIC COUNTS
# ======================================================================

total_records = len(df)

emoji_records = int(df["has_emoji"].sum())

non_emoji_records = total_records - emoji_records

emoji_percentage = (
    emoji_records / total_records * 100
    if total_records > 0
    else 0
)


print_section("1. EMOJI PREVALENCE")

print(f"Total records             : {total_records:,}")
print(f"Records containing emojis : {emoji_records:,}")
print(f"Records without emojis    : {non_emoji_records:,}")
print(f"Emoji percentage          : {emoji_percentage:.2f}%")


# ======================================================================
# LABEL DISTRIBUTION
# ======================================================================

print_section("2. OVERALL LABEL DISTRIBUTION")

label_counts = (
    df["label_name"]
    .value_counts()
    .reindex(["Non-Offensive", "Offensive"])
    .fillna(0)
    .astype(int)
)

print(label_counts.to_string())


# ======================================================================
# EMOJI × LABEL
# ======================================================================

print_section("3. EMOJI PRESENCE × LABEL")

emoji_label_table = pd.crosstab(
    df["label_name"],
    df["has_emoji"]
)

emoji_label_table = emoji_label_table.reindex(
    index=["Non-Offensive", "Offensive"],
    columns=[0, 1],
    fill_value=0
)

emoji_label_table.columns = [
    "Without_Emoji",
    "With_Emoji"
]

print(emoji_label_table)


# ======================================================================
# OFFENSIVE RATE WITH/WITHOUT EMOJIS
# ======================================================================

emoji_groups = []

for has_emoji, group in df.groupby("has_emoji"):

    total = len(group)

    offensive = int(
        (group["label_id"] == 1).sum()
    )

    non_offensive = int(
        (group["label_id"] == 0).sum()
    )

    offensive_rate = (
        offensive / total * 100
        if total > 0
        else 0
    )

    emoji_groups.append({
        "emoji_presence": (
            "With Emoji"
            if has_emoji
            else "Without Emoji"
        ),
        "total_records": total,
        "non_offensive": non_offensive,
        "offensive": offensive,
        "offensive_rate_percent": offensive_rate
    })

emoji_group_df = pd.DataFrame(emoji_groups)

print_section("4. OFFENSIVE RATE WITH/WITHOUT EMOJIS")
print(emoji_group_df.to_string(index=False))


# ======================================================================
# LANGUAGE-WISE EMOJI STATISTICS
# ======================================================================

language_rows = []

for language, group in df.groupby("language_clean"):

    total = len(group)

    emoji_count = int(
        group["has_emoji"].sum()
    )

    non_emoji = total - emoji_count

    offensive_total = int(
        (group["label_id"] == 1).sum()
    )

    emoji_offensive = int(
        (
            (group["has_emoji"] == 1)
            & (group["label_id"] == 1)
        ).sum()
    )

    emoji_non_offensive = int(
        (
            (group["has_emoji"] == 1)
            & (group["label_id"] == 0)
        ).sum()
    )

    offensive_rate_emoji = (
        emoji_offensive / emoji_count * 100
        if emoji_count > 0
        else 0
    )

    language_rows.append({
        "language": language,
        "total_records": total,
        "emoji_records": emoji_count,
        "non_emoji_records": non_emoji,
        "emoji_percentage": (
            emoji_count / total * 100
            if total > 0
            else 0
        ),
        "emoji_non_offensive": emoji_non_offensive,
        "emoji_offensive": emoji_offensive,
        "offensive_rate_among_emoji_percent": offensive_rate_emoji,
        "total_offensive": offensive_total
    })

language_emoji_df = pd.DataFrame(
    language_rows
).sort_values(
    "language"
)

print_section("5. LANGUAGE-WISE EMOJI STATISTICS")
print(language_emoji_df.to_string(index=False))


# ======================================================================
# EXTRACT ALL EMOJIS
# ======================================================================

print_section("6. EXTRACTING INDIVIDUAL EMOJIS")

overall_counter = Counter()

language_counters = defaultdict(Counter)

label_counters = defaultdict(Counter)

emoji_record_map = defaultdict(list)

for _, row in df.iterrows():

    text = row[TEXT_COL]
    language = row["language_clean"]
    label = row["label_name"]

    emojis = extract_emoji_codepoints(text)

    for emoji in emojis:

        overall_counter[emoji] += 1
        language_counters[language][emoji] += 1
        label_counters[label][emoji] += 1

        emoji_record_map[emoji].append(
            {
                "language": language,
                "label": label,
                "text": text
            }
        )


# ======================================================================
# TOP EMOJIS OVERALL
# ======================================================================

top_overall = []

for emoji, count in overall_counter.most_common():

    records = emoji_record_map[emoji]

    offensive_count = sum(
        r["label"] == "Offensive"
        for r in records
    )

    non_offensive_count = sum(
        r["label"] == "Non-Offensive"
        for r in records
    )

    total = offensive_count + non_offensive_count

    offensive_rate = (
        offensive_count / total * 100
        if total > 0
        else 0
    )

    top_overall.append({
        "emoji": emoji,
        "total_occurrences": count,
        "non_offensive_occurrences": non_offensive_count,
        "offensive_occurrences": offensive_count,
        "offensive_rate_percent": offensive_rate
    })

top_overall_df = pd.DataFrame(top_overall)

print_section("7. TOP 30 EMOJIS OVERALL")

if len(top_overall_df) > 0:
    print(
        top_overall_df.head(30).to_string(
            index=False
        )
    )
else:
    print("No emojis detected.")


# ======================================================================
# TOP EMOJIS BY LABEL
# ======================================================================

print_section("8. TOP EMOJIS BY CLASS")

for label in ["Non-Offensive", "Offensive"]:

    print()
    print(f"--- {label} ---")

    counter = label_counters[label]

    if len(counter) == 0:
        print("No emojis.")
        continue

    for emoji, count in counter.most_common(20):
        print(
            f"{emoji}  ->  {count:,}"
        )


# ======================================================================
# TOP EMOJIS BY LANGUAGE
# ======================================================================

print_section("9. TOP EMOJIS BY LANGUAGE")

for language in sorted(language_counters.keys()):

    print()
    print(f"--- {language} ---")

    counter = language_counters[language]

    for emoji, count in counter.most_common(20):
        print(
            f"{emoji}  ->  {count:,}"
        )


# ======================================================================
# EMOJI CLASS ASSOCIATION
# ======================================================================

print_section("10. EMOJI-LEVEL OFFENSIVE ASSOCIATION")

association_rows = []

for emoji, records in emoji_record_map.items():

    total = len(records)

    offensive = sum(
        r["label"] == "Offensive"
        for r in records
    )

    non_offensive = total - offensive

    offensive_rate = (
        offensive / total * 100
        if total > 0
        else 0
    )

    association_rows.append({
        "emoji": emoji,
        "total_comments": total,
        "offensive_comments": offensive,
        "non_offensive_comments": non_offensive,
        "offensive_rate_percent": offensive_rate
    })

emoji_association_df = pd.DataFrame(
    association_rows
)

if len(emoji_association_df) > 0:

    emoji_association_df = (
        emoji_association_df
        .sort_values(
            [
                "offensive_rate_percent",
                "total_comments"
            ],
            ascending=[False, False]
        )
        .reset_index(drop=True)
    )

    print(
        emoji_association_df.head(30).to_string(
            index=False
        )
    )


# ======================================================================
# FREQUENT EMOJIS WITH MINIMUM SUPPORT
# ======================================================================

# Avoid claiming an emoji is associated with offensive language
# when it occurs only once or twice.

MIN_SUPPORT = 10

supported = emoji_association_df[
    emoji_association_df["total_comments"] >= MIN_SUPPORT
].copy()

supported = supported.sort_values(
    [
        "offensive_rate_percent",
        "total_comments"
    ],
    ascending=[False, False]
)

print_section(
    f"11. HIGHEST OFFENSIVE-RATE EMOJIS "
    f"(MINIMUM SUPPORT = {MIN_SUPPORT})"
)

if len(supported) > 0:
    print(
        supported.head(30).to_string(
            index=False
        )
    )
else:
    print("No emojis meet the minimum support threshold.")


# ======================================================================
# EMOJI COUNT DISTRIBUTION
# ======================================================================

print_section("12. EMOJI COUNT PER COMMENT")

emoji_comments = df[
    df["has_emoji"] == 1
].copy()

if len(emoji_comments) > 0:

    print(
        emoji_comments[
            "emoji_count"
        ].describe().to_string()
    )

    print()
    print(
        "Maximum emojis in one comment:",
        int(
            emoji_comments["emoji_count"].max()
        )
    )

    print(
        "Average emojis per emoji-containing comment:",
        f"{emoji_comments['emoji_count'].mean():.3f}"
    )

else:
    print("No emoji-containing comments.")


# ======================================================================
# REPEATED EMOJI ANALYSIS
# ======================================================================

print_section("13. REPEATED EMOJI PATTERNS")

repeated = df[
    df["has_repeated_emoji"] == 1
]

print(
    "Comments with repeated emojis:",
    len(repeated)
)

print(
    "Percentage of emoji comments:",
    (
        len(repeated) / emoji_records * 100
        if emoji_records > 0
        else 0
    ),
    "%"
)

if len(repeated) > 0:

    repeated_offensive = int(
        (repeated["label_id"] == 1).sum()
    )

    repeated_non_offensive = int(
        (repeated["label_id"] == 0).sum()
    )

    print(
        "Repeated-emoji comments that are Offensive:",
        repeated_offensive
    )

    print(
        "Repeated-emoji comments that are Non-Offensive:",
        repeated_non_offensive
    )


# ======================================================================
# EMOJI DENSITY
# ======================================================================

print_section("14. EMOJI DENSITY")

for label in ["Non-Offensive", "Offensive"]:

    group = df[
        df["label_name"] == label
    ]

    print()
    print(label)

    print(
        "Mean emoji density:",
        f"{group['emoji_density'].mean():.6f}"
    )

    print(
        "Median emoji density:",
        f"{group['emoji_density'].median():.6f}"
    )

    print(
        "Mean emoji count:",
        f"{group['emoji_count'].mean():.4f}"
    )


# ======================================================================
# STATISTICAL ASSOCIATION
# ======================================================================

print_section("15. STATISTICAL ASSOCIATION: EMOJI PRESENCE × LABEL")

chi_square_result = None

if SCIPY_AVAILABLE:

    table = pd.crosstab(
        df["has_emoji"],
        df["label_id"]
    )

    table = table.reindex(
        index=[0, 1],
        columns=[0, 1],
        fill_value=0
    )

    chi2, p_value, dof, expected = chi2_contingency(
        table
    )

    chi_square_result = {
        "chi_square": float(chi2),
        "p_value": float(p_value),
        "degrees_of_freedom": int(dof)
    }

    print("Chi-square statistic :", f"{chi2:.6f}")
    print("p-value              :", f"{p_value:.10f}")
    print("Degrees of freedom   :", dof)

    if p_value < 0.05:
        print(
            "\nResult: Emoji presence and label show "
            "a statistically significant association "
            "at the 0.05 level."
        )
    else:
        print(
            "\nResult: No statistically significant "
            "association detected at the 0.05 level."
        )

else:

    print(
        "SciPy is not available."
    )

    print(
        "Statistical test skipped."
    )


# ======================================================================
# LANGUAGE-SPECIFIC ASSOCIATION
# ======================================================================

print_section(
    "16. LANGUAGE-SPECIFIC EMOJI PRESENCE ASSOCIATION"
)

language_association_rows = []

for language in sorted(
    df["language_clean"].unique()
):

    subset = df[
        df["language_clean"] == language
    ]

    table = pd.crosstab(
        subset["has_emoji"],
        subset["label_id"]
    )

    table = table.reindex(
        index=[0, 1],
        columns=[0, 1],
        fill_value=0
    )

    if SCIPY_AVAILABLE:

        try:
            chi2, p_value, dof, expected = chi2_contingency(
                table
            )

            language_association_rows.append({
                "language": language,
                "chi_square": chi2,
                "p_value": p_value,
                "degrees_of_freedom": dof,
                "significant_at_0_05": (
                    p_value < 0.05
                )
            })

            print()
            print(language)
            print(
                "Chi-square:",
                f"{chi2:.6f}"
            )
            print(
                "p-value:",
                f"{p_value:.10f}"
            )

        except Exception as e:

            print()
            print(language)
            print(
                "Could not calculate test:",
                e
            )

language_association_df = pd.DataFrame(
    language_association_rows
)


# ======================================================================
# CANDIDATE FEATURE SUMMARY
# ======================================================================

print_section("17. CANDIDATE EMOJI FEATURES")

candidate_features = [
    {
        "feature": "has_emoji",
        "description": "Whether a comment contains at least one emoji",
        "type": "binary"
    },
    {
        "feature": "emoji_count",
        "description": "Total number of emoji codepoints",
        "type": "numeric"
    },
    {
        "feature": "unique_emoji_count",
        "description": "Number of distinct emojis",
        "type": "numeric"
    },
    {
        "feature": "emoji_density",
        "description": "Emoji count divided by text length",
        "type": "numeric"
    },
    {
        "feature": "repeated_emoji_count",
        "description": "Number of repeated emoji occurrences",
        "type": "numeric"
    },
    {
        "feature": "has_repeated_emoji",
        "description": "Whether an emoji is repeated",
        "type": "binary"
    },
    {
        "feature": "max_emoji_repetition",
        "description": "Maximum repetition count of one emoji",
        "type": "numeric"
    },
    {
        "feature": "emoji_cluster_count",
        "description": "Number of emoji grapheme clusters",
        "type": "numeric"
    },
    {
        "feature": "unique_emoji_cluster_count",
        "description": "Number of unique emoji grapheme clusters",
        "type": "numeric"
    }
]

candidate_feature_df = pd.DataFrame(
    candidate_features
)

print(
    candidate_feature_df.to_string(
        index=False
    )
)


# ======================================================================
# SAVE DATA WITH FEATURES
# ======================================================================

print_section("18. SAVING RESULTS")

df.to_csv(
    os.path.join(
        OUT_DIR,
        "emoji_enriched_dataset.csv"
    ),
    index=False,
    encoding="utf-8-sig"
)

emoji_group_df.to_csv(
    os.path.join(
        OUT_DIR,
        "emoji_presence_label_statistics.csv"
    ),
    index=False,
    encoding="utf-8-sig"
)

language_emoji_df.to_csv(
    os.path.join(
        OUT_DIR,
        "emoji_language_statistics.csv"
    ),
    index=False,
    encoding="utf-8-sig"
)

top_overall_df.to_csv(
    os.path.join(
        OUT_DIR,
        "top_emojis_overall.csv"
    ),
    index=False,
    encoding="utf-8-sig"
)

emoji_association_df.to_csv(
    os.path.join(
        OUT_DIR,
        "emoji_label_association.csv"
    ),
    index=False,
    encoding="utf-8-sig"
)

supported.to_csv(
    os.path.join(
        OUT_DIR,
        "supported_emoji_associations.csv"
    ),
    index=False,
    encoding="utf-8-sig"
)

candidate_feature_df.to_csv(
    os.path.join(
        OUT_DIR,
        "candidate_emoji_features.csv"
    ),
    index=False,
    encoding="utf-8-sig"
)

language_association_df.to_csv(
    os.path.join(
        OUT_DIR,
        "language_emoji_association.csv"
    ),
    index=False,
    encoding="utf-8-sig"
)


# ======================================================================
# PLOT 1 — EMOJI PRESENCE
# ======================================================================

plt.figure(figsize=(7, 5))

labels = [
    "Without Emoji",
    "With Emoji"
]

values = [
    non_emoji_records,
    emoji_records
]

plt.bar(labels, values)

plt.title("Emoji Presence in Dravidian Dataset")
plt.ylabel("Number of Comments")

save_plot(
    os.path.join(
        OUT_DIR,
        "emoji_presence.png"
    )
)


# ======================================================================
# PLOT 2 — EMOJI PERCENTAGE BY LANGUAGE
# ======================================================================

plt.figure(figsize=(8, 5))

plot_df = language_emoji_df.sort_values(
    "emoji_percentage",
    ascending=False
)

plt.bar(
    plot_df["language"],
    plot_df["emoji_percentage"]
)

plt.title(
    "Percentage of Emoji-Containing Comments by Language"
)

plt.ylabel("Emoji Comments (%)")

save_plot(
    os.path.join(
        OUT_DIR,
        "emoji_percentage_by_language.png"
    )
)


# ======================================================================
# PLOT 3 — OFFENSIVE RATE WITH/WITHOUT EMOJI
# ======================================================================

plt.figure(figsize=(7, 5))

plt.bar(
    emoji_group_df["emoji_presence"],
    emoji_group_df["offensive_rate_percent"]
)

plt.title(
    "Offensive Rate With vs Without Emojis"
)

plt.ylabel("Offensive Rate (%)")

save_plot(
    os.path.join(
        OUT_DIR,
        "offensive_rate_emoji_presence.png"
    )
)


# ======================================================================
# PLOT 4 — TOP 20 EMOJIS
# ======================================================================

if len(top_overall_df) > 0:

    top20 = top_overall_df.head(20).copy()

    plt.figure(figsize=(10, 7))

    plt.bar(
        range(len(top20)),
        top20["total_occurrences"]
    )

    plt.xticks(
        range(len(top20)),
        top20["emoji"],
        rotation=45,
        ha="right"
    )

    plt.title("Top 20 Most Frequent Emojis")
    plt.ylabel("Occurrences")

    save_plot(
        os.path.join(
            OUT_DIR,
            "top20_emojis.png"
        )
    )


# ======================================================================
# PLOT 5 — OFFENSIVE RATE OF SUPPORTED EMOJIS
# ======================================================================

if len(supported) > 0:

    top_supported = supported.head(20).copy()

    plt.figure(figsize=(10, 7))

    plt.bar(
        range(len(top_supported)),
        top_supported["offensive_rate_percent"]
    )

    plt.xticks(
        range(len(top_supported)),
        top_supported["emoji"],
        rotation=45,
        ha="right"
    )

    plt.title(
        f"Offensive Rate of Emojis "
        f"(Minimum Support = {MIN_SUPPORT})"
    )

    plt.ylabel("Offensive Rate (%)")

    save_plot(
        os.path.join(
            OUT_DIR,
            "supported_emoji_offensive_rate.png"
        )
    )


# ======================================================================
# CREATE TEXT REPORT
# ======================================================================

report_path = os.path.join(
    OUT_DIR,
    "step30_emoji_analysis_report.txt"
)

with open(
    report_path,
    "w",
    encoding="utf-8"
) as f:

    f.write("=" * 78 + "\n")
    f.write("STEP 30 — EMOJI FEATURE ANALYSIS\n")
    f.write("=" * 78 + "\n\n")

    f.write(
        "Purpose:\n"
        "Determine whether explicit emoji information provides useful "
        "signal for offensive-language detection.\n\n"
    )

    f.write(
        "IMPORTANT:\n"
        "- Separate analysis only.\n"
        "- No model was trained.\n"
        "- Step 20 was not modified.\n"
        "- Step 24 was not modified.\n"
        "- Step 26 was not modified.\n"
        "- Final multilingual results were not modified.\n\n"
    )

    f.write("DATASET\n")
    f.write("-" * 78 + "\n")
    f.write(f"Total records after deduplication: {total_records:,}\n")
    f.write(f"Emoji-containing records: {emoji_records:,}\n")
    f.write(f"Non-emoji records: {non_emoji_records:,}\n")
    f.write(f"Emoji percentage: {emoji_percentage:.2f}%\n\n")

    f.write("EMOJI × LABEL\n")
    f.write("-" * 78 + "\n")
    f.write(
        emoji_label_table.to_string()
    )
    f.write("\n\n")

    f.write("OFFENSIVE RATE WITH/WITHOUT EMOJIS\n")
    f.write("-" * 78 + "\n")
    f.write(
        emoji_group_df.to_string(index=False)
    )
    f.write("\n\n")

    f.write("LANGUAGE-WISE STATISTICS\n")
    f.write("-" * 78 + "\n")
    f.write(
        language_emoji_df.to_string(index=False)
    )
    f.write("\n\n")

    f.write("TOP 30 EMOJIS\n")
    f.write("-" * 78 + "\n")

    if len(top_overall_df) > 0:
        f.write(
            top_overall_df.head(30).to_string(
                index=False
            )
        )

    f.write("\n\n")

    f.write(
        "TOP EMOJIS BY CLASS\n"
    )
    f.write("-" * 78 + "\n")

    for label in [
        "Non-Offensive",
        "Offensive"
    ]:

        f.write(f"\n{label}\n")

        for emoji, count in label_counters[
            label
        ].most_common(20):

            f.write(
                f"{emoji} -> {count:,}\n"
            )

    f.write("\n")

    f.write(
        "CANDIDATE FEATURES\n"
    )
    f.write("-" * 78 + "\n")
    f.write(
        candidate_feature_df.to_string(
            index=False
        )
    )
    f.write("\n\n")

    if chi_square_result is not None:

        f.write(
            "STATISTICAL ASSOCIATION\n"
        )
        f.write("-" * 78 + "\n")

        f.write(
            f"Chi-square: "
            f"{chi_square_result['chi_square']:.6f}\n"
        )

        f.write(
            f"p-value: "
            f"{chi_square_result['p_value']:.10f}\n"
        )

        f.write(
            f"Degrees of freedom: "
            f"{chi_square_result['degrees_of_freedom']}\n"
        )

        if chi_square_result["p_value"] < 0.05:
            f.write(
                "Interpretation: Significant association "
                "at the 0.05 level.\n"
            )
        else:
            f.write(
                "Interpretation: No significant association "
                "at the 0.05 level.\n"
            )

        f.write("\n")

    f.write(
        "NEXT STEP\n"
    )
    f.write("-" * 78 + "\n")

    f.write(
        "If the analysis shows useful emoji signal, the next experiment "
        "can evaluate an emoji-aware feature branch separately from the "
        "final multilingual model.\n"
    )

    f.write(
        "The emoji branch must be evaluated as a separate experiment "
        "and must not overwrite the existing multilingual final results.\n"
    )


# ======================================================================
# FINAL SUMMARY
# ======================================================================

print_section("FINAL STEP 30 SUMMARY")

print(
    f"Total records analyzed       : {total_records:,}"
)

print(
    f"Emoji-containing records     : {emoji_records:,}"
)

print(
    f"Emoji percentage             : {emoji_percentage:.2f}%"
)

print()

print(
    "Language-wise emoji counts:"
)

for _, row in language_emoji_df.iterrows():

    print(
        f"  {row['language']:<12} : "
        f"{int(row['emoji_records']):,} "
        f"({row['emoji_percentage']:.2f}%)"
    )

print()

print(
    "Top emojis:"
)

for _, row in top_overall_df.head(15).iterrows():

    print(
        f"  {row['emoji']} : "
        f"{int(row['total_occurrences']):,} occurrences | "
        f"Offensive rate "
        f"{row['offensive_rate_percent']:.2f}%"
    )

print()

if chi_square_result is not None:

    print(
        "Emoji presence ↔ label statistical test:"
    )

    print(
        f"  Chi-square = "
        f"{chi_square_result['chi_square']:.6f}"
    )

    print(
        f"  p-value    = "
        f"{chi_square_result['p_value']:.10f}"
    )

    if chi_square_result["p_value"] < 0.05:
        print(
            "  Significant association detected."
        )
    else:
        print(
            "  No significant association detected."
        )

print()
print("Candidate features identified:")
print(
    ", ".join(
        candidate_feature_df["feature"].tolist()
    )
)

print()
print("Output folder:")
print(OUT_DIR)

print()
print("Generated files:")

for filename in sorted(
    os.listdir(OUT_DIR)
):

    print(
        "  ",
        filename
    )

print()
print("=" * 78)
print("STEP 30 COMPLETED")
print("=" * 78)