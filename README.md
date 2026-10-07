# Offensive Language Detection in Social Media

## Overview

Offensive language on social media is difficult to detect because users express harmful or abusive content using noisy, multilingual, code-mixed, contextual, and intentionally obfuscated text.

This project develops a multilingual offensive-language detection framework using **XLM-RoBERTa (XLM-R)** combined with a semantic classification branch. The system is evaluated using English offensive-language data and multilingual Dravidian code-mixed data covering **Tamil, Malayalam, and Kannada**.

An additional inference-time robustness layer is implemented to examine whether predictions remain stable when complete sentences are subjected to generic text-obfuscation patterns.

---

## Objectives

The main objectives of this project are:

- Detect offensive and non-offensive social-media text.
- Develop a binary offensive-language classification system.
- Establish a classical machine-learning baseline.
- Use XLM-R for contextual multilingual representation.
- Combine contextual and semantic predictions using a hybrid approach.
- Evaluate performance across English, Tamil, Malayalam, and Kannada.
- Prevent duplicate and cross-split data leakage.
- Perform language-wise evaluation and error analysis.
- Analyze prediction stability under sentence-level text obfuscation.

---

## Key Contributions

### 1. Multilingual Cross-Dataset Framework

The system combines an English offensive-language dataset with multilingual Dravidian code-mixed data containing:

- Tamil
- Malayalam
- Kannada

This allows evaluation beyond a single language or dataset.

### 2. Contextual-Semantic Hybrid Classification

The final detection system combines:

- XLM-R contextual probability
- XLM-R semantic representation
- A semantic Logistic Regression classifier

The two prediction signals are combined using calibrated weights.

### 3. Leakage-Controlled Dataset Preparation

The multilingual dataset was cleaned using:

- Text normalization
- Exact duplicate removal
- Cross-split duplicate checking
- Train/validation/test separation

This reduces the possibility of artificially inflated evaluation results caused by duplicate sentences appearing across splits.

### 4. Multilingual and Language-Wise Evaluation

The final system is evaluated separately on:

- English
- Tamil
- Malayalam
- Kannada

This exposes differences in model behavior across languages instead of reporting only a single overall score.

### 5. Inference-Time Evasion-Aware Robustness Analysis

A final inference-time layer checks prediction stability under generic sentence-level text obfuscation patterns such as:

- Internal character spacing
- Punctuation insertion
- Repeated-character variation
- Zero-width character insertion

The underlying trained model is kept frozen. No additional model training is performed for this layer.

> The system identifies prediction instability; it does not determine the user's intention.

---

# System Architecture

```text
                    SOCIAL MEDIA TEXT
                           |
                           v
                  Text Preprocessing
                           |
                           v
              +------------+------------+
              |                         |
              v                         v
        XLM-RoBERTa              Semantic Branch
              |                         |
              v                         v
       XLM-R Probability       Mean + Max Pooling
              |                         |
              |                    Logistic Regression
              |                         |
              +------------+------------+
                           |
                           v
                    Hybrid Prediction
                           |
                           v
             Offensive / Non-Offensive
                           |
                           v
                Independent Evaluation
                           |
                           v
              Evasion-Aware Robustness
                           |
                           v
               Sentence-Level Variants
                           |
                           v
                 Same Frozen Model
                           |
                           v
              Stability / Review Flag
Datasets
Dataset 1 - English Offensive Language
File:
data/train.csv

The dataset contains 60,000 records with:
text
label

Original categories include:
- Non_Toxic
- Homophobia_LGBTQphobia
- Misogyny_Sexism
- Racism_Xenophobia
- Harassment_Stalking
- Religious_Hate
- Profanity_Insult
For the binary classification task:
Non_Toxic
    -> Non-Offensive

All other categories
    -> Offensive

Binary distribution:
Class	Samples
Non-Offensive	8,571
Offensive	51,429
Total	60,000


Dataset 2 - DravidianCodeMix
File:
data/dravidian_clean/dravidian_multilingual_clean.csv

The dataset contains multilingual/code-mixed social-media text in:
- Tamil
- Malayalam
- Kannada
The original offensive-language labels are mapped into:
0 -> Non-Offensive
1 -> Offensive

After duplicate and leakage cleaning, the final dataset contains approximately:
59,974 valid records

Final language distribution:
Language	Samples
Tamil	41,732
Malayalam	12,596
Kannada	5,646


Final class distribution:
Class	Samples
Non-Offensive	47,749
Offensive	12,225


Data Cleaning
The multilingual dataset undergoes the following preprocessing:
1. Text normalization
2. Exact duplicate removal
3. Cross-split duplicate detection
4. Removal of train-validation overlap
5. Removal of train-test overlap
6. Removal of validation-test overlap
Final independent split sizes used for evaluation:
Training   : 48,743
Validation : 5,643
Testing    : 5,571

The final test evaluation combines:
Dataset 1 test samples : 315
Dataset 2 test samples : 5,571

Total                  : 5,886

Models
Baseline
The classical baseline uses:
TF-IDF
   +
Logistic Regression

Baseline results:
Metric	Score
Accuracy	0.8382
Precision	0.8397
Recall	0.8382
F1-score	0.8379


XLM-RoBERTa
The transformer component uses:
xlm-roberta-base

XLM-R provides contextual multilingual representations for the input sentence.
Semantic Classification Branch
The XLM-R representation is processed using:
- Mean pooling
- Max pooling
The resulting semantic representation is used by:
StandardScaler
        +
Logistic Regression

Hybrid Prediction
The final prediction combines:
XLM-R probability
+
Semantic classifier probability

The calibrated configuration uses:
XLM-R weight       = 0.55
Semantic weight    = 0.45
Threshold          = 0.65

Final Evaluation
The final independent test evaluation contains:
5,886 samples

Overall performance:
Metric	Score
Accuracy	79.73%
Macro-F1	73.32%
Offensive Precision	60.31%
Offensive Recall	60.19%
Offensive F1	60.25%


Language-Wise Performance
Language	Accuracy	Macro-F1	Offensive F1
English	100.00%	100.00%	100.00%
Tamil	77.16%	68.49%	51.95%
Malayalam	86.13%	57.88%	23.38%
Kannada	77.70%	73.07%	61.90%
Overall	79.73%	73.32%	60.25%


The language-wise analysis demonstrates that multilingual offensive-language detection remains challenging, particularly for languages with fewer offensive examples.
Error Analysis
The final test set contains:
Total samples : 5,886
Errors        : 1,193
Error rate    : approximately 20.27%

Error analysis includes:
- False positives
- False negatives
- High-confidence errors
- Language-wise errors
- Confusion matrices
The analysis is used to identify limitations of the detector rather than relying only on overall accuracy.
Evasion-Aware Robustness Layer
The final inference-time layer operates on complete sentences.
Example:
Original:
You are an idiot.

Possible structural variants:
You are an id iot.
You are an id.iot.
You are an idiiot.

Each complete sentence is passed through the same frozen Step 24 detector.
The system compares:
Original prediction
        vs
Variant predictions

If substantial prediction instability is observed, the sentence can be marked for review.
Important
Step 32:
- Does not retrain XLM-R.
- Does not train another classifier.
- Does not use an offensive-word dictionary.
- Does not use manually defined offensive sentences.
- Does not determine user intention.
- Uses the existing frozen Step 24 model.
It is an inference-time robustness/evasion-aware analysis layer.
Project Pipeline
Dataset Collection
        |
        v
Data Cleaning
        |
        v
Binary Label Mapping
        |
        v
Duplicate / Leakage Removal
        |
        v
TF-IDF Baseline
        |
        v
XLM-R Training
        |
        v
Semantic Classifier
        |
        v
Hybrid Model
        |
        v
Independent Test Evaluation
        |
        v
Language-Wise Evaluation
        |
        v
Error Analysis
        |
        v
Inference-Time Evasion-Aware Analysis

Project Structure
Offensive-Language-Detection-in-Social-Media/
│
├── data/
│   ├── train.csv
│   └── dravidian_clean/
│       └── dravidian_multilingual_clean.csv
│
├── src/
│   ├── train_mbert.py
│   ├── step24_final.py
│   ├── step25_diagnosis.py
│   ├── step26_final_evaluation.py
│   ├── step29_error_analysis.py
│   └── step32_evasion_aware.py
│
├── results/
│   └── ...
│
├── README.md
├── .gitignore
└── requirements.txt

Generated model files, datasets, logs, and experiment outputs should not be uploaded unnecessarily to GitHub.
Technologies Used
- Python
- PyTorch
- Hugging Face Transformers
- XLM-RoBERTa
- Scikit-learn
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Joblib
Installation
Clone the repository:
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd Offensive-Language-Detection-in-Social-Media

Create a virtual environment:
python -m venv .venv

Activate on Windows:
.venv\Scripts\activate

Install dependencies:
pip install -r requirements.txt

Running the Project
Baseline
python src/train_baseline.py

Final evaluation
python src/step26_final_evaluation.py

Error analysis
python src/step29_error_analysis.py

Evasion-aware demonstration
python src/step32_evasion_aware.py

The final demonstration should use the already-trained model artifacts. The final inference and robustness stages do not require retraining the underlying XLM-R model.

Demonstration
The final demonstration should show three aspects:
1. Normal offensive/non-offensive prediction
Example:
You are an idiot.

2. Multilingual prediction
Examples from:
Tamil
Malayalam
Kannada

3. Evasion-aware robustness
A complete sentence is passed through the frozen detector and its automatically generated obfuscated variants are compared.
Limitations
The system has several limitations:
- Performance varies across languages.
- Contextual offensive language remains challenging.
- Multilingual and code-mixed text can produce prediction errors.
- Obfuscation analysis detects prediction instability but cannot determine user intention.
- The system does not guarantee perfect classification for every sentence.
- The final model is dependent on the quality and distribution of the training datasets.
Future Work
Possible future improvements include:
- Larger multilingual datasets
- Better low-resource language coverage
- More realistic social-media obfuscation generation
- Improved contextual classification
- Explainable offensive-language detection
- Human-in-the-loop moderation
- Adversarial robustness evaluation
- Deployment as a real-time moderation API
Conclusion
This project presents a multilingual offensive-language detection framework combining contextual XLM-R representations with a semantic classification branch. The system is evaluated using English and Dravidian code-mixed data and reports both overall and language-specific performance.
The project also includes an inference-time robustness layer that examines prediction stability under sentence-level text obfuscation without retraining the underlying detector.
The resulting framework provides a practical pipeline for multilingual offensive-language detection, evaluation, error analysis, and robustness assessment.
#   O f f e n s i v e - l a n g u a g e - d e t e c t i o n - s o c i a l - m e d i a  
 