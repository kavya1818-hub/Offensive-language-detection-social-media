import pandas as pd
import os
import pickle

from sklearn.feature_extraction.text import TfidfVectorizer


# ============================================================
# STEP 5 - FEATURE ENGINEERING
# ============================================================

print("=" * 60)
print("STEP 5 - FEATURE ENGINEERING")
print("=" * 60)


# ============================================================
# 1. PATHS
# ============================================================

TRAIN_PATH = "data/processed/train.csv"
TEST_PATH = "data/processed/test.csv"

OUTPUT_DIR = "data/features"

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# 2. LOAD DATA
# ============================================================

train_df = pd.read_csv(TRAIN_PATH)
test_df = pd.read_csv(TEST_PATH)

print("\nTraining records :", len(train_df))
print("Testing records  :", len(test_df))


# ============================================================
# 3. VERIFY REQUIRED COLUMNS
# ============================================================

if "model_text" not in train_df.columns:
    raise ValueError("model_text column not found in training data.")

if "model_text" not in test_df.columns:
    raise ValueError("model_text column not found in testing data.")

if "label" not in train_df.columns:
    raise ValueError("label column not found in training data.")

if "label" not in test_df.columns:
    raise ValueError("label column not found in testing data.")


print("\nRequired columns verified.")


# ============================================================
# 4. PREPARE TEXT
# ============================================================

X_train_text = train_df["model_text"].fillna("").astype(str)
X_test_text = test_df["model_text"].fillna("").astype(str)

y_train = train_df["label"]
y_test = test_df["label"]


# ============================================================
# 5. WORD-LEVEL TF-IDF
# ============================================================

print("\nCreating WORD-level TF-IDF features...")

word_vectorizer = TfidfVectorizer(
    lowercase=True,
    ngram_range=(1, 2),
    min_df=2,
    max_df=0.95,
    sublinear_tf=True,
    max_features=20000
)

X_train_word = word_vectorizer.fit_transform(X_train_text)
X_test_word = word_vectorizer.transform(X_test_text)

print("Word TF-IDF training shape :", X_train_word.shape)
print("Word TF-IDF testing shape  :", X_test_word.shape)

print("Word vocabulary size        :", len(word_vectorizer.vocabulary_))


# ============================================================
# 6. CHARACTER-LEVEL TF-IDF
# ============================================================

print("\nCreating CHARACTER-level TF-IDF features...")

char_vectorizer = TfidfVectorizer(
    analyzer="char",
    ngram_range=(3, 5),
    min_df=2,
    max_features=20000,
    sublinear_tf=True
)

X_train_char = char_vectorizer.fit_transform(X_train_text)
X_test_char = char_vectorizer.transform(X_test_text)

print("Character TF-IDF training shape :", X_train_char.shape)
print("Character TF-IDF testing shape  :", X_test_char.shape)

print("Character vocabulary size        :",
      len(char_vectorizer.vocabulary_))


# ============================================================
# 7. SAVE VECTORIZERS
# ============================================================

with open(
    os.path.join(OUTPUT_DIR, "word_tfidf_vectorizer.pkl"),
    "wb"
) as f:
    pickle.dump(word_vectorizer, f)


with open(
    os.path.join(OUTPUT_DIR, "char_tfidf_vectorizer.pkl"),
    "wb"
) as f:
    pickle.dump(char_vectorizer, f)


# ============================================================
# 8. SAVE FEATURE MATRICES
# ============================================================

from scipy.sparse import save_npz

save_npz(
    os.path.join(OUTPUT_DIR, "X_train_word.npz"),
    X_train_word
)

save_npz(
    os.path.join(OUTPUT_DIR, "X_test_word.npz"),
    X_test_word
)

save_npz(
    os.path.join(OUTPUT_DIR, "X_train_char.npz"),
    X_train_char
)

save_npz(
    os.path.join(OUTPUT_DIR, "X_test_char.npz"),
    X_test_char
)


# ============================================================
# 9. SAVE LABELS
# ============================================================

y_train.to_csv(
    os.path.join(OUTPUT_DIR, "y_train.csv"),
    index=False
)

y_test.to_csv(
    os.path.join(OUTPUT_DIR, "y_test.csv"),
    index=False
)


# ============================================================
# 10. FEATURE SUMMARY
# ============================================================

summary = pd.DataFrame({
    "Feature_Type": [
        "Word TF-IDF",
        "Character TF-IDF"
    ],
    "Ngram_Range": [
        "(1,2)",
        "(3,5)"
    ],
    "Vocabulary_Size": [
        len(word_vectorizer.vocabulary_),
        len(char_vectorizer.vocabulary_)
    ],
    "Training_Features": [
        X_train_word.shape[1],
        X_train_char.shape[1]
    ],
    "Training_Samples": [
        X_train_word.shape[0],
        X_train_char.shape[0]
    ],
    "Testing_Samples": [
        X_test_word.shape[0],
        X_test_char.shape[0]
    ]
})

summary.to_csv(
    "results/feature_engineering_summary.csv",
    index=False
)


# ============================================================
# 11. FINAL OUTPUT
# ============================================================

print("\n" + "=" * 60)
print("FEATURE ENGINEERING SUMMARY")
print("=" * 60)

print("\nWORD TF-IDF")
print("Training shape :", X_train_word.shape)
print("Testing shape  :", X_test_word.shape)
print("Vocabulary     :", len(word_vectorizer.vocabulary_))

print("\nCHARACTER TF-IDF")
print("Training shape :", X_train_char.shape)
print("Testing shape  :", X_test_char.shape)
print("Vocabulary     :", len(char_vectorizer.vocabulary_))

print("\nGenerated files:")
print("- data/features/word_tfidf_vectorizer.pkl")
print("- data/features/char_tfidf_vectorizer.pkl")
print("- data/features/X_train_word.npz")
print("- data/features/X_test_word.npz")
print("- data/features/X_train_char.npz")
print("- data/features/X_test_char.npz")
print("- data/features/y_train.csv")
print("- data/features/y_test.csv")
print("- results/feature_engineering_summary.csv")

print("\n" + "=" * 60)
print("FEATURE ENGINEERING COMPLETED")
print("=" * 60)