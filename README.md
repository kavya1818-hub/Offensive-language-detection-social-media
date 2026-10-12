# Offensive Language Detection in Social Media

A multilingual offensive-language detection system using **XLM-RoBERTa (XLM-R)** and a semantic hybrid classifier, evaluated on English and Indian code-mixed social-media text. The project also includes a Streamlit dashboard for interactive prediction, multilingual testing, screenshot analysis, model evaluation, and inference-time robustness checking.

## Overview

Offensive language detection is challenging because social-media text can contain informal spelling, multilingual expressions, code-mixing, contextual language, and deliberately obfuscated words.

This project develops a binary classification pipeline to identify **Offensive** and **Non-Offensive** text. It combines contextual predictions from XLM-RoBERTa with a semantic Logistic Regression classifier and evaluates the resulting system across English, Tamil, Malayalam, and Kannada.

An additional inference-time robustness layer examines whether predictions change when complete sentences undergo generic text-obfuscation transformations. This layer reuses the existing trained model without retraining it.

## Key Features

- Binary offensive-language classification.
- Classical TF-IDF and Logistic Regression baseline.
- Multilingual XLM-RoBERTa model.
- Hybrid classification combining XLM-R and semantic predictions.
- English, Tamil, Malayalam, and Kannada evaluation.
- Dataset preprocessing, duplicate removal, and cross-split leakage checks.
- Language-wise performance analysis and confusion matrices.
- Error analysis covering false positives, false negatives, and high-confidence errors.
- Sentence-level obfuscation and prediction-stability analysis.
- Streamlit dashboard with interactive detection and multilingual testing.
- Screenshot-based comment extraction and CSV batch analysis.
- Prediction history and analytics.

## Key Contributions

### 1. Multilingual cross-dataset evaluation

The project combines an English offensive-language dataset with multilingual Dravidian code-mixed data containing Tamil, Malayalam, and Kannada.

### 2. Contextual-semantic hybrid classification

The final detector combines the offensive probability from XLM-R with a semantic classifier trained on pooled XLM-R representations.

### 3. Leakage-controlled evaluation

The data preparation pipeline performs duplicate removal and cross-split overlap checks to reduce evaluation leakage.

### 4. Language-wise analysis

Performance is reported separately for English, Tamil, Malayalam, and Kannada, making differences in classification performance visible.

### 5. Inference-time robustness analysis

The additional robustness layer tests whether predictions remain stable under generic sentence-level transformations, including internal spacing, punctuation insertion, repeated characters, and zero-width characters.

It identifies potential prediction instability; it does not establish whether a user intentionally attempted to evade detection.

### 6. Interactive dashboard

The Streamlit application provides an interface for trying text examples, examining predictions, checking robustness, reviewing performance, and analyzing uploaded screenshots or CSV files.

## System Architecture

```text
             Social-Media Text
                     |
                     v
              Preprocessing
                     |
          +----------+----------+
          |                     |
          v                     v
    XLM-RoBERTa          Semantic Branch
          |              Mean + Max Pooling
          |                     |
          v                     v
    XLM-R Probability     Logistic Regression
          |                     |
          +----------+----------+
                     |
                     v
              Hybrid Probability
                     |
                     v
           Offensive / Non-Offensive
                     |
                     v
            Independent Evaluation
                     |
                     v
        Inference-Time Robustness Check
                     |
                     v
         Sentence Variants and Comparison
                     |
                     v
          Stability / Review Indicator
```

## Datasets

### Dataset 1: English offensive-language data

Expected local path: `data/train.csv`

The dataset contains 60,000 records with `text` and `label` columns. Its original categories include Non_Toxic, Homophobia_LGBTQphobia, Misogyny_Sexism, Racism_Xenophobia, Harassment_Stalking, Religious_Hate, and Profanity_Insult.

For binary classification:

- `Non_Toxic` → `Non-Offensive` (0)
- All other categories → `Offensive` (1)

Original class distribution:

| Class | Records |
|---|---:|
| Non-Offensive | 8,571 |
| Offensive | 51,429 |
| Total | 60,000 |

### Dataset 2: DravidianCodeMix

Expected local path: `data/dravidian_clean/dravidian_multilingual_clean.csv`

This dataset contains multilingual and code-mixed social-media text in Tamil, Malayalam, and Kannada.

The original labels are mapped to:

- `0` → Non-Offensive
- `1` → Offensive

After the documented cleaning and deduplication process, the dataset contains 59,974 records.

| Language | Records |
|---|---:|
| Tamil | 41,732 |
| Malayalam | 12,596 |
| Kannada | 5,646 |
| Total | 59,974 |

| Class | Records |
|---|---:|
| Non-Offensive | 47,749 |
| Offensive | 12,225 |
| Total | 59,974 |

### Data preparation

The preparation pipeline includes:

1. Text normalization.
2. Exact duplicate removal.
3. Cross-split duplicate detection.
4. Removal of overlapping records across training, validation, and test splits.
5. Binary label mapping.
6. Separate evaluation on held-out test data.

Final Dataset 2 split sizes:

| Split | Records |
|---|---:|
| Training | 48,743 |
| Validation | 5,643 |
| Testing | 5,571 |

The final combined independent test evaluation contains 5,886 records: 315 English test records and 5,571 multilingual test records.

## Models and Methodology

### Classical baseline

The baseline uses TF-IDF vectorization followed by Logistic Regression.

| Metric | Score |
|---|---:|
| Accuracy | 83.82% |
| Precision | 83.97% |
| Recall | 83.82% |
| F1-score | 83.79% |

### Multilingual transformer

The transformer component uses `xlm-roberta-base` to produce contextual representations of input sentences.

### Semantic classification branch

The final semantic classifier uses mean and max pooling over XLM-R representations, followed by StandardScaler and Logistic Regression.

### Hybrid prediction

The final inference pipeline combines the XLM-R probability and semantic classifier probability using the saved calibration:

| Parameter | Value |
|---|---:|
| XLM-R weight | 0.55 |
| Semantic weight | 0.45 |
| Decision threshold | 0.65 |

The final prediction is based on the calibrated hybrid probability and threshold.

## Final Evaluation Results

The final independent test evaluation covers 5,886 samples.

| Metric | Score |
|---|---:|
| Accuracy | 79.73% |
| Macro-F1 | 73.32% |
| Offensive Precision | 60.31% |
| Offensive Recall | 60.19% |
| Offensive F1 | 60.25% |

### Language-wise performance

| Language | Accuracy | Macro-F1 | Offensive F1 |
|---|---:|---:|---:|
| English | 100.00% | 100.00% | 100.00% |
| Tamil | 77.16% | 68.49% | 51.95% |
| Malayalam | 86.13% | 57.88% | 23.38% |
| Kannada | 77.70% | 73.07% | 61.90% |
| **Overall** | **79.73%** | **73.32%** | **60.25%** |

The English test subset contains 315 records, so its reported score should be interpreted in light of that sample size. Performance also varies across the Dravidian languages, particularly for offensive-language recall and F1.

### Error analysis

The final test evaluation contains 1,193 errors, approximately 20.27% of the 5,886 samples.

The analysis includes:

- False positives and false negatives.
- High-confidence errors.
- Language-wise error patterns.
- Confusion matrices.
- Comparative performance visualizations.

## Evasion-Aware Robustness Analysis

The additional robustness layer operates on complete sentences using the frozen final detector.

Example:

Original sentence: `You are an idiot.`

Illustrative structural variants:

- `You are an id iot.`
- `You are an id.iot.`
- `You are an idiiot.`

The same detector evaluates the original sentence and its variants. The system compares the resulting probabilities and predictions to identify potential instability.

This component:

- Does not retrain XLM-R.
- Does not introduce another trained classifier.
- Does not rely on a manually defined offensive-word dictionary.
- Does not determine the user's intention.
- Reports possible prediction instability for review.

The robustness layer is an additional analysis feature, not a guarantee of correct classification for every transformed sentence.

## Streamlit Dashboard

The project includes a Streamlit dashboard in `dashboard/`.

Dashboard capabilities include:

- **Home:** Overall metrics and language-wise performance.
- **Detect:** Single-sentence prediction and probability display.
- **Multilingual Demo:** Batch testing of example sentences across languages.
- **Screenshot Analysis:** OCR-based extraction and analysis of social-media comments.
- **Robustness Check:** Comparison of original and obfuscated sentence predictions.
- **Model Performance:** Evaluation metrics and language-wise results.
- **History & Analytics:** Stored prediction history and visualizations.
- **Settings:** Model artifact status and password management.

## Repository Structure

```text
offensive-language-detection-social-media/
├── dashboard/
│   ├── app.py
│   ├── auth.py
│   ├── diagnose.py
│   ├── ocr_utils.py
│   ├── page_screenshot.py
│   ├── predictor.py
│   ├── robustness.py
│   └── requirements_dashboard.txt
├── src/
│   ├── 01_dataset_analysis.py
│   ├── 02_data_leakage_check.py
│   ├── ...
│   ├── 24_realtime_multilingual_prediction.py
│   ├── 25_diagnose_multilingual_model.py
│   ├── 26_evaluate_final_system.py
│   ├── ...
│   ├── 29_error_analysis.py
│   ├── 30_emoji_analysis.py
│   ├── 31_emoji_aware_experiment.py
│   └── 32_evasion_aware_moderation.py
├── README.md
├── .gitignore
└── requirements.txt
```

The `NOTES` file is optional and may remain local. The exact contents of each folder depend on the files included in the repository.

### Large files and local artifacts

The datasets, trained model files, and generated experiment results are excluded from GitHub through `.gitignore`.

Expected local paths include:

- `data/` — datasets.
- `models/` — optional model artifacts, if used by a local script.
- `results/` — trained Step 20/Step 24 artifacts, calibration files, evaluation outputs, and generated figures.

**Important:** A fresh GitHub clone does not automatically contain these ignored files. To run the final detector, provide the required datasets and trained model artifacts locally at the paths expected by the code. The dashboard requires the frozen transformer, semantic classifier, calibration, and configuration files.

## Installation

### Prerequisites

- Python 3.10 or a compatible version supported by the installed dependencies.
- Git.
- A Windows, Linux, or macOS environment.
- Access to the required local model artifacts for inference.

### 1. Clone the repository

```bash
git clone https://github.com/kavya1818-hub/offensive-language-detection-social-media.git
cd offensive-language-detection-social-media
```

### 2. Create a virtual environment

Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, use Command Prompt:

```bat
.venv\Scripts\activate.bat
```

### 3. Install project dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

For dashboard dependencies:

```bash
pip install -r dashboard/requirements_dashboard.txt
```

Install optional OCR dependencies only if required by the screenshot-analysis feature.

## Running the Dashboard

From the repository root:

```bash
streamlit run dashboard/app.py
```

Streamlit will display a local URL, usually `http://localhost:8501`.

The dashboard requires the expected trained model artifacts to be available locally. Installing Python dependencies alone does not recreate the ignored model files.

For security, change any default demonstration credentials before exposing the application to other users.

## Running Analysis Scripts

The numbered scripts in `src/` cover dataset analysis, preprocessing, baseline experiments, transformer experiments, multilingual evaluation, visualizations, error analysis, and robustness analysis.

Examples:

```bash
python src/01_dataset_analysis.py
python src/26_evaluate_final_system.py
python src/29_error_analysis.py
python src/32_evasion_aware_moderation.py
```

Run a script only after checking its expected input paths and dependencies. Training scripts may require datasets and substantial computation. Final inference and evaluation scripts may require previously generated model artifacts.

**The final model does not need to be retrained just to launch the dashboard if the required saved artifacts are already available.**

## Technologies Used

- Python
- PyTorch
- Hugging Face Transformers
- XLM-RoBERTa
- Scikit-learn
- Pandas and NumPy
- Matplotlib and Seaborn
- Plotly
- Streamlit
- Joblib
- EasyOCR / Tesseract for optional screenshot analysis

## Limitations

- Classification performance varies across languages.
- Offensive-language detection is sensitive to context, spelling variation, and code-mixing.
- The English test subset is relatively small compared with the multilingual test set.
- The semantic hybrid can still produce false positives and false negatives.
- The robustness layer measures prediction instability; it cannot establish user intent.
- Model performance depends on the training data and the distribution of real-world text.
- The dashboard cannot reproduce final predictions until the required model artifacts are available.

## Future Work

Potential improvements include:

- More representative offensive-language data for low-resource languages.
- Better handling of implicit and context-dependent offensive content.
- Broader adversarial robustness evaluation.
- Improved code-mixed language identification.
- Explainable predictions and human-in-the-loop moderation.
- Deployment as a secure real-time moderation service.

## Conclusion

This project implements a multilingual offensive-language detection pipeline combining XLM-R contextual predictions with a semantic classification branch. It evaluates English and Dravidian code-mixed text, reports language-wise results, performs error analysis, and includes an inference-time robustness layer for sentence-level obfuscation.

The Streamlit dashboard provides an interactive interface for exploring predictions, multilingual examples, robustness checks, and evaluation results. The project also documents the limitations of the model and the need for further improvement in language-specific offensive-language detection.

