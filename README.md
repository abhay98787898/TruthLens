# TruthLens — Fake News Detection using Machine Learning

<p align="center">
  <strong>Detecting misinformation through Natural Language Processing and Machine Learning</strong>
</p>

TruthLens is a machine learning-based fake news detection research project that analyzes news headlines and article text to classify content as **Fake** or **Real**. The project uses Natural Language Processing (NLP), TF-IDF feature extraction, and a Linear Support Vector Machine (Linear SVM) classifier trained on the WELFake dataset.

The project also explores data cleaning, duplicate removal, model evaluation, and robustness against data leakage caused by similar or duplicated news articles.

---

## Project Overview

Fake news detection is an important problem in the modern digital information ecosystem. The large volume of online information makes it increasingly difficult to distinguish misleading content from reliable reporting through manual verification alone.

TruthLens explores how machine learning techniques can identify linguistic patterns in news content and use them to classify articles.

### Objectives

- Build a machine learning system for fake news classification.
- Analyze news headlines and article body text.
- Apply Natural Language Processing and TF-IDF vectorization.
- Train and evaluate a Linear SVM classification model.
- Reduce data leakage caused by duplicate and near-duplicate articles.
- Compare model performance across different evaluation strategies.
- Develop an interactive web application for news text analysis.

---

## Features

- **Fake / Real Classification** — predicts whether submitted news text is classified as Fake or Real.
- **Headline & Article Analysis** — supports analysis using news headlines and article text.
- **NLP-based Processing** — transforms textual content into numerical features suitable for machine learning.
- **TF-IDF Feature Extraction** — represents text using word importance and frequency-based statistical features.
- **Linear SVM Classification** — uses a trained Linear Support Vector Machine model.
- **Prediction Confidence** — displays the model's prediction-related score or confidence information as implemented in the application.
- **Interactive Web Interface** — user-friendly interface powered by Streamlit.
- **Research Evaluation** — includes baseline experiments, group-based validation, duplicate analysis, and model evaluation scripts.

---

## Technology Stack

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| Streamlit | Interactive web application |
| Pandas | Data processing and analysis |
| NumPy | Numerical computing |
| Scikit-learn | Machine learning and text vectorization |
| TF-IDF | Text feature extraction |
| Linear SVM | News classification model |
| Joblib | Model serialization and loading |
| Git & GitHub | Version control and project hosting |

---

## Dataset

TruthLens uses the **WELFake Dataset**, a dataset designed for fake news detection research.

The dataset contains news articles labeled as Fake or Real and combines data from multiple sources.

### Dataset Information

| Property | Details |
|---|---|
| Dataset | WELFake |
| Original records | 72,134 |
| Classes | Fake / Real |
| Label 0 | Fake |
| Label 1 | Real |
| Input | News headline and article text |
| Language | English |

### Data Preparation

The research pipeline includes data preparation steps designed to improve data quality:

- Removal of empty text records.
- Text cleaning and normalization.
- Removal of exact duplicate records.
- Examination of duplicate and near-duplicate articles.
- Preparation of text inputs for feature extraction and model training.

After cleaning and exact deduplication, the research dataset contained approximately **63,104 usable unique records**.

> The original dataset is not stored in this GitHub repository due to its large size.

### Dataset Source

- [WELFake Dataset on Kaggle](https://www.kaggle.com/datasets/saurabhshahane/fake-news-classification)

---

## Methodology

TruthLens follows a text classification pipeline that converts news content into machine-readable features and uses a supervised learning model for classification.

### 1. Text Input

The application accepts news text, including a headline and article body.

The text fields are combined and processed as the input to the prediction pipeline.

### 2. Text Preprocessing

The research workflow includes cleaning and preparing textual data, handling empty entries, removing duplicates, and preparing documents for NLP-based feature extraction.

### 3. TF-IDF Feature Extraction

Term Frequency–Inverse Document Frequency (TF-IDF) transforms news text into numerical feature representations.

TF-IDF helps represent the importance of words within a document relative to their occurrence across the corpus.

### 4. Machine Learning Classification

The final saved model uses a **Linear Support Vector Machine (Linear SVM)** classifier trained on TF-IDF-based textual features.

The classifier learns patterns from labeled news examples and uses those learned patterns to predict the class of new input text.

### 5. Prediction

The processed text is passed through the trained model pipeline to produce a prediction:

- `0` → Fake
- `1` → Real

The model pipeline is saved using Joblib and loaded by the Streamlit application for inference.

---

## Research & Model Evaluation

TruthLens experiments investigate model performance under different data splitting and evaluation strategies.

The research includes:

- Baseline machine learning evaluation.
- Stratified train/test evaluation.
- Group-based splitting experiments.
- Text group split evaluation.
- Near-duplicate article auditing.
- Duplicate and leakage analysis.
- Robustness evaluation across multiple random seeds.

### Selected Experimental Results

The following results are from recorded research experiments and represent different experimental setups—not necessarily the exact performance of every deployed model instance.

| Experiment | Model | Accuracy | F1 Score |
|---|---|---:|---:|
| Baseline | Logistic Regression | 95.90% | 95.50% |
| Group Split | Logistic Regression | 96.22% | 95.85% |
| Text Group Split | Linear SVM | 96.51% | 96.14% |

<sub>Metrics are rounded from recorded experiment results. Results are dependent on the dataset preprocessing and evaluation split used in each experiment.</sub>

### Near-Duplicate Audit

The project also investigates near-duplicate articles to identify potential evaluation leakage.

The audit uses MinHash and Locality-Sensitive Hashing (LSH) techniques to identify candidate similar article pairs.

This is important because highly similar articles appearing in both training and testing sets can make evaluation results overly optimistic.

---

## Model

| Component | Implementation |
|---|---|
| Model Type | Linear Support Vector Machine |
| Feature Extraction | TF-IDF |
| Input Text | Headline + Article Body |
| Model Serialization | Joblib |
| Label Mapping | 0 = Fake, 1 = Real |
| Framework | Scikit-learn |

### Saved Model

The trained pipeline is stored at:

```text
artifacts/final_svm_pipeline.joblib
```

Model metadata is stored at:

```text
artifacts/model_metadata.json
```

The Streamlit application loads the saved pipeline instead of retraining the model every time the application starts.

---

## Limitations

TruthLens is an academic machine learning prototype and should not be treated as an authoritative fact-checking system.

- **Prediction is not fact verification:** A classification is not proof that an article is true or false.
- **Dataset limitations:** Training data may contain historical, geographical, political, or source-related biases.
- **Temporal limitations:** News language and misinformation techniques change over time.
- **Language limitations:** The model is designed for English text and is not validated for other languages.
- **Source limitations:** The model analyzes text and does not independently verify journalists, publishers, references, or sources.
- **Context limitations:** Sarcasm, satire, breaking news, incomplete reporting, and complex contextual information may affect predictions.
- **Generalization:** Performance on news sources outside the training dataset requires further independent validation.

TruthLens should therefore be used as an experimental analytical tool, not as a replacement for professional fact-checking or source verification.

---

## Installation & Local Setup

### Prerequisites

- Python 3.10 or later
- Git
- pip

### 1. Clone the repository

```bash
git clone https://github.com/abhay98787898/TruthLens.git
cd TruthLens
```

### 2. Create a virtual environment

**Windows:**

```powershell
python -m venv .venv
.venv\Scripts\activate
```

**Linux / macOS:**

```bash
python -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run TruthLens

```bash
streamlit run app.py
```

The application will open in your browser. If it doesn't open automatically, visit the local URL displayed in the terminal, usually:

```text
http://localhost:8501
```

---

## Running the Research Scripts

The repository also contains scripts used during data preparation, experimentation, model evaluation, and research analysis.

Research scripts are located in:

```text
scripts/
```

The trained model artifacts are located in:

```text
artifacts/
```

Dataset files are expected inside:

```text
data/
```

Refer to individual scripts and their configurations for the relevant experimental workflow.

---

## Deployment

TruthLens can be deployed using **Streamlit Community Cloud**.

### Deployment Steps

1. Push the TruthLens repository to GitHub.
2. Open [Streamlit Community Cloud](https://share.streamlit.io/).
3. Connect your GitHub account.
4. Select the `TruthLens` repository.
5. Choose the `main` branch.
6. Set the main application file to:

```text
app.py
```

7. Deploy the application.

Ensure the required dependencies are present in `requirements.txt` and the trained model file is included in the repository.

---

## Repository Structure

```text
TruthLens/
│
├── app.py
├── requirements.txt
├── README.md
├── LICENSE
├── .gitignore
│
├── artifacts/
│   ├── final_svm_pipeline.joblib
│   └── model_metadata.json
│
├── data/
│   └── WELFake_Dataset.csv       # Not included in Git
│
├── scripts/
│   ├── research_baseline.py
│   ├── research_group_split.py
│   ├── research_text_group_split.py
│   ├── research_near_duplicate_audit.py
│   └── ...
│
├── notebooks/
│   └── welfake_fake_news_detection.ipynb
│
└── reports/
    ├── final_report.pdf
    ├── presentation.pdf
    └── presentation.pptx
```

<sub>Structure shown is representative of the main project files. Additional research scripts and experiment outputs may be present.</sub>

---

## References

1. Verma, P., Agrawal, P., Amorim, I., & Prodan, R. (2021). *WELFake: Word embedding over linguistic features for fake news detection.* IEEE Transactions on Computational Social Systems, 8(4), 881–893.
2. Shu, K., Sliva, A., Wang, S., Tang, J., & Liu, H. (2017). *Fake news detection on social media: A data mining perspective.* ACM SIGKDD Explorations, 19(1), 22–36.
3. Pedregosa, F. et al. (2011). *Scikit-learn: Machine learning in Python.* Journal of Machine Learning Research, 12, 2825–2830.

---

## Project Team

**Developed by:** Abhay Pratap Singh

**Mentor:** Sudhesna Roy

**College:** BVIMIT

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

<p align="center">
  <strong>TruthLens</strong><br>
  Machine Learning Research Project — Fake News Detection
</p>
