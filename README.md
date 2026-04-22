# WELFake Fake News Detection

**Classical ML baseline for fake news classification, engineered around the failure mode that breaks most academic fake-news papers: source-style overfitting.**

![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![Models](https://img.shields.io/badge/classifiers-14-informational)
![F1](https://img.shields.io/badge/weighted%20F1-0.977-success)
![Validation](https://img.shields.io/badge/validated-5--fold%20CV%20%2B%20learning%20curves-brightgreen)
![License](https://img.shields.io/badge/license-MIT-lightgrey)

A comprehensive machine learning pipeline for fake news classification on the **WELFake Dataset** — a 72,134-article multi-source benchmark. Combines 23 handcrafted linguistic features with four text-vectorization schemes, trains 14 classifiers spanning linear, Bayesian, kernel, and ensemble families, and extends the analysis with unsupervised clustering, topic modeling, dimensionality reduction, and anomaly detection.

Final project for **ADSP 31009 — Machine Learning I**, MS in Applied Data Science, The University of Chicago.

---

## Why the dataset choice is the real story

Most fake-news classifiers published on single-source datasets (e.g., one Kaggle corpus) report very high accuracy. This is misleading. In earlier experiments on single-source data, training and validation F1 diverged as training set size grew — the classic overfitting signature indicating the model was memorizing **source-specific stylistic markers** (Reuters' bylines, Breitbart's headline style, etc.) rather than learning generalizable fake-news signals.

The response was to switch to **WELFake** (Verma et al., 2021), a benchmark explicitly constructed from four independent source datasets (Kaggle, McIntire, Reuters, BuzzFeed Political) to disrupt that memorization. On WELFake, training and validation F1 **converge** and plateau around 15,000 training examples — the honest signature of genuine generalization rather than memorization.

**The 97.72% weighted F1 reported below is therefore not a "too good to be true" number — it is the result of a model that has been explicitly tested against its most likely failure mode.**

Additional validation:

- **5-fold stratified cross-validation** on the top 5 models — confirms the single 80/20 test-set result is not a favorable-split artifact.
- **Learning curves** — converge rather than diverge with more data.
- **Dimensionality reduction sanity check** — a Logistic Regression on SVD-50 features achieves only ~50% F1 (essentially random), confirming the discriminative signal lives in high-dimensional sparse TF-IDF rather than a low-rank subspace.
- **Unsupervised baselines** — four clustering algorithms produce ARI scores of 0.008–0.089, i.e. the fake/real distinction is weakly but not strongly recoverable without labels. If clustering had been trivially separable, that would have been cause to suspect a leakage-adjacent signal.

---

## Headline results

All 14 supervised models, sorted by weighted F1 on the 20% held-out test set:

| Rank | Model | Accuracy | Weighted F1 |
|---:|---|---:|---:|
| 1 | **Ridge Classifier** | **0.9772** | **0.9772** |
| 2 | Linear SVC (calibrated) | 0.9750 | 0.9750 |
| 3 | Voting Ensemble (top-3 soft) | 0.9690 | 0.9690 |
| 4 | SGD (log / logistic) | 0.9678 | 0.9678 |
| 5 | SGD (hinge / SVM) | 0.9664 | 0.9664 |
| 6 | Stacking (LR + NB + SVC → LR) | 0.9658 | 0.9658 |
| 7 | Logistic Regression* | 0.9382 | 0.9382 |
| 8 | XGBoost | 0.9328 | 0.9328 |
| 9 | Hist Gradient Boosting | 0.9318 | 0.9318 |
| 10 | Extra Trees | 0.9228 | 0.9228 |
| 11 | Random Forest | 0.9164 | 0.9164 |
| 12 | Multinomial NB | 0.8962 | 0.8962 |
| 13 | Complement NB | 0.8962 | 0.8962 |
| 14 | Bernoulli NB | 0.8908 | 0.8907 |
| 15 | Decision Tree | 0.8488 | 0.8488 |

*\*Logistic Regression underperformed relative to other linear models due to LBFGS solver convergence issues on the 100k-feature combined matrix. The Ridge Classifier (least-squares closed form) and calibrated Linear SVC do not hit this limitation. Switching to the `saga` solver or increasing `max_iter` is flagged as a trivial follow-up.*

**Top-performing models are all linear.** This confirms the well-established NLP finding that high-dimensional sparse TF-IDF features are close to linearly separable — non-linear tree-based methods underperform once trained on the reduced (SVD-50) features they require for tractability.

---

## Table of contents

- [What the pipeline does](#what-the-pipeline-does)
- [Dataset](#dataset)
- [Methodology](#methodology)
- [Counter-intuitive findings](#counter-intuitive-findings)
- [Unsupervised analysis](#unsupervised-analysis)
- [Limitations](#limitations)
- [Reproduce](#reproduce)
- [Repository structure](#repository-structure)
- [References](#references)
- [Team](#team)

---

## What the pipeline does

1. **Text preprocessing** — lowercase → contraction expansion → URL/mention/number normalization → punctuation stripping → stop-word removal → WordNet lemmatization. Two variants kept: `clean` (for vectorization) and `clean_raw` (for n-gram EDA).
2. **Feature engineering** — 23 handcrafted lexical, stylistic, structural, sentiment (VADER), and readability features.
3. **Vectorization** — four schemes: TF-IDF word (1,2)-grams × 60k features, TF-IDF char (3,5)-grams × 40k, BoW (1,2)-grams × 50k, hashing vectorizer with 2^17 features.
4. **Combined feature matrix** — horizontal sparse stack of TF-IDF word + char + scaled handcrafted features. 100,019 columns, ~99.98% zeros.
5. **Supervised learning** — 14 classifiers + 2 meta-ensembles (voting + stacking), evaluated with accuracy, precision, recall, weighted F1, ROC-AUC, 5-fold CV.
6. **Unsupervised exploration** — 4 clustering algorithms (MiniBatch KMeans, Agglomerative Ward, BIRCH, Gaussian Mixture), 3 topic models (LDA, NMF, LSA), 2 embeddings (t-SNE, UMAP), and Isolation Forest anomaly detection.
7. **Interpretability** — Logistic Regression coefficient analysis on TF-IDF features; top-25 words indicative of fake vs. real classes.
8. **Interactive prediction** — given any title + body, returns classification + confidence + ensemble vote + linguistic feature breakdown.

## Dataset

**WELFake Dataset** — introduced by Verma et al. (2021) in IEEE Transactions on Computational Social Systems specifically to mitigate the single-source domain bias that plagues fake-news research. Merges four independent corpora:

| Source | Coverage | Labels |
|---|---|---|
| Kaggle | Mixed fake & real news | Fake / Real |
| McIntire | Fake news corpus | Fake |
| Reuters | Verified real-world news | Real |
| BuzzFeed Political | Political news & fact-checks | Fake / Real |

**Size:** 72,134 articles — 37,106 fake (label=0) and 35,028 real (label=1), slight class imbalance handled via **weighted F1** as primary metric.

**Working sample:** 25,000 articles stratified on label with `RANDOM_STATE=42`. Learning curves confirm this sample is sufficient — performance plateaus well before the full corpus size.

**Source:** [Kaggle — WELFake](https://www.kaggle.com/datasets/saurabhshahane/fake-news-classification). The CSV (~250 MB) is not committed to this repository; see [Reproduce](#reproduce) for download instructions.

## Methodology

### Feature engineering (23 features)

| Group | Features |
|---|---|
| Lexical (5) | `word_count`, `char_count`, `unique_words`, `avg_word_len`, `type_token_ratio` |
| Stylistic (7) | `cap_ratio`, `cap_count`, `exclamation`, `question`, `punct_ratio`, `quote_count`, `has_url` |
| Structural (4) | `sentence_count`, `avg_sent_len`, `title_word_count`, `title_cap_ratio` |
| Sentiment — VADER (4) | `sent_compound`, `sent_pos`, `sent_neg`, `sent_neu` |
| Readability (3) | `flesch_reading_ease`, `flesch_kincaid_grade`, `smog_index` |

### Vectorization

| Component | Columns | Purpose |
|---|---:|---|
| TF-IDF word (1,2)-grams | 60,000 | Word-level semantic and syntactic patterns |
| TF-IDF char (3,5)-grams | 40,000 | Sub-word style patterns (robust to misspellings; captures register markers like `!!!`, `CAPS`) |
| Handcrafted features | 19 | Linguistic, structural, sentiment signals (scaled before stacking) |
| **Combined sparse matrix** | **100,019** | Horizontally stacked via `scipy.sparse.hstack`; ~99.98% zeros |

`TF-IDF(t, d) = log(1 + count(t, d)) × log(N / df(t))` — sublinear TF suppresses high-frequency words, IDF upweights rare discriminative terms.

### Training setup

- **Split:** 80/20 stratified on label, `random_state=42`.
- **Naive Bayes** trained on TF-IDF word only (requires non-negative features).
- **All other models** trained on the combined feature matrix.
- **Tree/boosting models** (Random Forest, Extra Trees, XGBoost, HistGB) trained on SVD-50 reduced features for tractability.
- **Linear SVC** wrapped in `CalibratedClassifierCV` to produce probability outputs for soft voting.
- **Stacking** uses 3-fold CV to train the LR meta-learner on LR + NB + SVC predictions.

### Validation

- 5-fold stratified CV on top 5 models (macro F1 scoring) — confirms test-set performance is not a split artifact.
- Learning curves for best model — convergence confirmed, no overfitting.
- Grid search over LR regularization `C ∈ {0.1, 0.5, 1.0, 5.0, 10.0}` (3-fold CV, macro F1) — optimum at `C = 10`.

## Counter-intuitive findings

A few things a naive "fake news = clickbait" intuition gets wrong:

- **Real news is more negative in sentiment.** VADER compound score: real ≈ −0.13, fake ≈ −0.04. This is driven by Reuters-style political and international reporting, which is structurally negative (conflict, crisis, criticism). Fake news tends to skew neutral or positive — clickbait is not necessarily negative.
- **Real news cites sources 12× more often.** URL presence is the single most discriminative handcrafted feature.
- **Real news has richer vocabulary.** Type-token ratio: 0.609 real vs. 0.586 fake.
- **Fake articles are longer, not shorter.** ~595 words vs. ~523 words — but with more quotes (2.19 vs. 1.38 average) and lower vocabulary precision.
- **Capitalization favors real news.** `cap_ratio` 0.063 vs. 0.039, driven by proper nouns (institutions, named figures) in professional journalism.

These align with the Logistic Regression coefficient analysis: features indicative of real news are policy-oriented terms, named institutions, and source bywords like "reuters"; features indicative of fake news skew toward emotionally charged and conspiracy-adjacent language.

## Unsupervised analysis

### Clustering on SVD-100 features

| Algorithm | Configuration | ARI vs. true labels |
|---|---|---:|
| MiniBatch KMeans | k=2, n_init=15 | 0.008 |
| Agglomerative | k=2, Ward linkage, n=5000 | **0.089** |
| BIRCH | threshold=0.5, k=2 | 0.000 (collapsed to single cluster) |
| Gaussian Mixture | n=2, diagonal covariance | 0.043 |

Moderate ARI scores confirm the fake/real boundary is *partially* recoverable from unsupervised text features, but the two classes overlap substantially in SVD space — supervised models remain essential for high-accuracy classification.

### Topic modeling (12 topics each)

- **LDA** identifies distinct clusters including US politics, international conflict, healthcare/science, entertainment, economic policy, elections. Real news concentrates on governance topics; fake news disperses across emotional/sensationalist clusters.
- **NMF** yields more coherent, geographically aligned topics — empirically more interpretable than LDA on this corpus.
- **LSA** (Truncated SVD) is less interpretable (components include negative weights) but computationally efficient.

### Anomaly detection

Isolation Forest trained exclusively on real news (contamination = 0.1). Fake articles receive marginally more negative anomaly scores (0.021 vs. 0.043 for real), but ARI with true labels is only 0.028 — **too low for standalone deployment**. This is a genuine null result and is reported honestly: unsupervised anomaly detection alone cannot substitute for a labeled classifier on this problem.

## Limitations

- **Temporal drift.** Training data is 2020-era. Fake news language evolves; a model trained today will degrade without retraining. No time-based holdout test was performed.
- **Domain generalization.** WELFake is English-language and US-centric. Performance on non-English or non-Western news is unknown.
- **Adversarial robustness.** Sophisticated fake news written to mimic Reuters-style prose (higher `avg_word_len`, zero exclamations, formal register) approaches the decision boundary. See the "maritime trade volume" example in the [presentation](reports/presentation.pdf), slide 9.
- **Context blindness.** Text-only. Ignores metadata (publication domain, author credibility), propagation patterns, and image/video content.
- **Logistic Regression convergence.** LR underperformed due to LBFGS solver issues on the 100k-feature matrix. Switching to the `saga` solver or increasing `max_iter` is a trivial follow-up.
- **No external validation set.** Results are 80/20 split + 5-fold CV on WELFake. Performance on a truly held-out source (e.g., recent articles from outside the original four corpora) would be the strongest test.

## Reproduce

```bash
# 1. Clone
git clone https://github.com/MatteoMirgone/welfake-fake-news-detection.git
cd welfake-fake-news-detection

# 2. Environment (Python 3.10+)
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 3. Download WELFake (~250 MB) — see data/README.md for Kaggle instructions
#    Place the CSV at: data/WELFake_Dataset.csv

# 4. Download NLTK resources (first run only)
python -c "import nltk; nltk.download('vader_lexicon'); nltk.download('stopwords'); nltk.download('wordnet'); nltk.download('averaged_perceptron_tagger')"

# 5. Run
jupyter lab notebooks/welfake_fake_news_detection.ipynb
```

**Runtime note.** On a standard laptop, the full pipeline on the 25k stratified sample takes ~20–30 minutes. Setting `SAMPLE_SIZE = None` in the config cell runs on the full 72k at ~3–4× cost. Results plateau well before the full sample size, so 25k is the sensible default for iteration.

**Key hyperparameters** (editable in the configuration cell at the top of the notebook):

```python
DATA_PATH    = "data/WELFake_Dataset.csv"
SAMPLE_SIZE  = 25000    # None = full 72k
RANDOM_STATE = 42
N_TOPICS     = 12       # LDA / NMF / LSA
N_CLUSTERS   = 2        # KMeans / GMM
```

## Repository structure

```
welfake-fake-news-detection/
├── README.md
├── LICENSE
├── requirements.txt
├── .gitignore
├── .gitattributes
├── data/
│   └── README.md                                   # Dataset download instructions
├── notebooks/
│   └── welfake_fake_news_detection.ipynb           # Full pipeline, top-to-bottom reproducible
└── reports/
    ├── final_report.pdf                            # Academic paper (abstract → references)
    ├── presentation.pdf                            # Final class presentation (PDF)
    ├── presentation.pptx                           # Final class presentation (editable)
    └── notebook_preview.html                       # Static HTML export of the notebook
```

## References

- Verma, P., Agrawal, P., Amorim, I., & Prodan, R. (2021). WELFake: Word embedding over linguistic features for fake news detection. *IEEE Transactions on Computational Social Systems*, 8(4), 881–893.
- Hutto, C. J., & Gilbert, E. E. (2014). VADER: A parsimonious rule-based model for sentiment analysis of social media text. *ICWSM-14.*
- Blei, D. M., Ng, A. Y., & Jordan, M. I. (2003). Latent Dirichlet Allocation. *Journal of Machine Learning Research*, 3, 993–1022.
- Liu, F. T., Ting, K. M., & Zhou, Z. H. (2008). Isolation Forest. *ICDM 2008*, 413–422.
- Shu, K., Sliva, A., Wang, S., Tang, J., & Liu, H. (2017). Fake news detection on social media: A data mining perspective. *ACM SIGKDD Explorations*, 19(1), 22–36.
- van der Maaten, L., & Hinton, G. (2008). Visualizing data using t-SNE. *Journal of Machine Learning Research*, 9, 2579–2605.
- Pedregosa, F. et al. (2011). Scikit-learn: Machine learning in Python. *Journal of Machine Learning Research*, 12, 2825–2830.

## Team

- **Dhruvi Gandhi**
- **Luciano Handal Baracatt**
- **Matteo Mirgone** *(maintainer of this repository)*
- **Sameeksha Kini**

**Mentor:** Prof. Greg Green
**Course:** ADSP 31009 — Machine Learning I, MS in Applied Data Science, The University of Chicago

*This is a team project. The code, report, and analysis are the collective work of all four authors. This repository is maintained by Matteo Mirgone for portfolio purposes; please credit the full author list in any derivative work.*
