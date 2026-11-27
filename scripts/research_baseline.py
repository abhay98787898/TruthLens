
from pathlib import Path
import json

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
)

RANDOM_STATE = 42
DATA_PATH = Path("data/WELFake_Dataset.csv")
REPORT_DIR = Path("reports")
REPORT_DIR.mkdir(parents=True, exist_ok=True)


def normalize(series):
    return (
        series.fillna("")
        .astype(str)
        .str.lower()
        .str.strip()
        .str.replace(r"\s+", " ", regex=True)
    )


def main():
    print("Loading WELFake dataset...")
    df = pd.read_csv(DATA_PATH)

    required = {"title", "text", "label"}
    if not required.issubset(df.columns):
        raise ValueError(f"Missing required columns: {required - set(df.columns)}")

    df["title_norm"] = normalize(df["title"])
    df["text_norm"] = normalize(df["text"])

    # Verify duplicate article pairs do not have conflicting labels.
    label_counts = df.groupby(
        ["title_norm", "text_norm"], dropna=False
    )["label"].nunique()

    conflicts = int((label_counts > 1).sum())
    print("Duplicate groups with conflicting labels:", conflicts)

    if conflicts:
        raise ValueError(
            "Conflicting labels found for duplicate article pairs. "
            "Investigate them before training."
        )

    # Remove repeated normalized title-text pairs.
    before = len(df)
    df = df.drop_duplicates(
        subset=["title_norm", "text_norm"]
    ).copy()
    print("Rows removed as duplicate pairs:", before - len(df))

    # Keep available title and article text.
    df["content"] = (
        df["title"].fillna("").astype(str).str.strip()
        + " "
        + df["text"].fillna("").astype(str).str.strip()
    ).str.strip()

    # Remove only records with no usable content.
    df = df[df["content"].str.len() > 0].copy()

    df["label"] = pd.to_numeric(df["label"], errors="raise").astype(int)
    if not set(df["label"].unique()).issubset({0, 1}):
        raise ValueError("Unexpected labels: expected 0=fake and 1=real.")

    print("Rows available for modelling:", len(df))
    print("Label counts:")
    print(df["label"].value_counts().sort_index().to_string())

    # Fixed, stratified holdout split for both models.
    X_train, X_test, y_train, y_test = train_test_split(
        df["content"],
        df["label"],
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=df["label"],
    )

    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            C=1.0,
            random_state=RANDOM_STATE,
        ),
        "Linear SVM": LinearSVC(
            C=1.0,
            random_state=RANDOM_STATE,
        ),
    }

    all_metrics = []
    error_frames = []

    for name, classifier in models.items():
        print(f"\nTraining {name}...")

        pipeline = Pipeline([
            (
                "tfidf",
                TfidfVectorizer(
                    max_features=50000,
                    ngram_range=(1, 2),
                    min_df=2,
                    sublinear_tf=True,
                    stop_words="english",
                ),
            ),
            ("classifier", classifier),
        ])

        pipeline.fit(X_train, y_train)
        predictions = pipeline.predict(X_test)

        # Both models provide a continuous decision score for ROC-AUC.
        scores = pipeline.decision_function(X_test)

        metrics = {
            "model": name,
            "accuracy": accuracy_score(y_test, predictions),
            "precision": precision_score(
                y_test, predictions, zero_division=0
            ),
            "recall": recall_score(
                y_test, predictions, zero_division=0
            ),
            "f1": f1_score(y_test, predictions, zero_division=0),
            "roc_auc": roc_auc_score(y_test, scores),
            "train_rows": len(X_train),
            "test_rows": len(X_test),
        }
        all_metrics.append(metrics)

        print(json.dumps(metrics, indent=2))
        print("\nClassification report:")
        print(classification_report(
            y_test,
            predictions,
            target_names=["Fake (0)", "Real (1)"],
            zero_division=0,
        ))
        print("Confusion matrix [rows=true, columns=predicted]:")
        print(confusion_matrix(y_test, predictions, labels=[0, 1]))

        errors = pd.DataFrame({
            "model": name,
            "title": df.loc[X_test.index, "title"].values,
            "true_label": y_test.values,
            "predicted_label": predictions,
            "text_excerpt": (
                df.loc[X_test.index, "text"]
                .fillna("")
                .astype(str)
                .str.slice(0, 500)
                .values
            ),
        })
        errors = errors[errors["true_label"] != errors["predicted_label"]]
        error_frames.append(errors)

        # Logistic Regression coefficients indicate associations,
        # not proof that individual words cause a prediction.
        if name == "Logistic Regression":
            vectorizer = pipeline.named_steps["tfidf"]
            coefficients = pipeline.named_steps["classifier"].coef_[0]
            feature_names = vectorizer.get_feature_names_out()

            top_fake = coefficients.argsort()[:20]
            top_real = coefficients.argsort()[-20:][::-1]

            features = pd.DataFrame({
                "term": (
                    list(feature_names[top_fake])
                    + list(feature_names[top_real])
                ),
                "coefficient": (
                    list(coefficients[top_fake])
                    + list(coefficients[top_real])
                ),
                "associated_class": (
                    ["Fake (0)"] * len(top_fake)
                    + ["Real (1)"] * len(top_real)
                ),
            })
            features.to_csv(
                REPORT_DIR / "top_logistic_features.csv",
                index=False,
            )

    results = pd.DataFrame(all_metrics).sort_values("f1", ascending=False)
    results.to_csv(REPORT_DIR / "baseline_metrics.csv", index=False)

    if error_frames:
        pd.concat(error_frames, ignore_index=True).to_csv(
            REPORT_DIR / "misclassified_examples.csv",
            index=False,
        )

    print("\nBaseline comparison:")
    print(results.to_string(index=False))
    print("\nReports saved in:", REPORT_DIR.resolve())


if __name__ == "__main__":
    main()
