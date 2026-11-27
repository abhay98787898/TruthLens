
from pathlib import Path
import re

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


# These are diagnostic rules, not a claim that these sources are unreliable.
PUBLISHER_PATTERN = re.compile(
    r"\b(?:reuters|breitbart|the new york times|new york times|"
    r"washington post|associated press|fox news|cnn|bbc)\b",
    flags=re.IGNORECASE,
)

BOILERPLATE_PATTERNS = [
    re.compile(r"\bfeatured image\b", re.IGNORECASE),
    re.compile(r"\bgetty images?\b", re.IGNORECASE),
    re.compile(r"\bimage courtesy of\b", re.IGNORECASE),
    re.compile(r"\bfollow us on twitter\b", re.IGNORECASE),
    re.compile(r"\btwitter\.com\b", re.IGNORECASE),
]


def remove_boilerplate(text):
    for pattern in BOILERPLATE_PATTERNS:
        text = pattern.sub(" ", text)
    return re.sub(r"\s+", " ", text).strip()


def remove_publisher_names(text):
    text = PUBLISHER_PATTERN.sub(" ", text)
    return re.sub(r"\s+", " ", text).strip()


def build_pipeline(classifier):
    return Pipeline([
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


def main():
    print("Loading dataset...")
    df = pd.read_csv(DATA_PATH)

    required = {"title", "text", "label"}
    if not required.issubset(df.columns):
        raise ValueError(f"Missing columns: {required - set(df.columns)}")

    df["title_norm"] = normalize(df["title"])
    df["text_norm"] = normalize(df["text"])

    # Ensure identical normalized title-text pairs do not have
    # conflicting labels before removing duplicates.
    label_counts = df.groupby(
        ["title_norm", "text_norm"], dropna=False
    )["label"].nunique()

    conflicts = int((label_counts > 1).sum())
    print("Duplicate groups with conflicting labels:", conflicts)

    if conflicts:
        raise ValueError(
            "Conflicting labels found. Resolve these before evaluation."
        )

    original_rows = len(df)
    df = df.drop_duplicates(
        subset=["title_norm", "text_norm"]
    ).copy()

    print("Duplicate rows removed:", original_rows - len(df))

    df["label"] = pd.to_numeric(
        df["label"], errors="raise"
    ).astype(int)

    if not set(df["label"].unique()).issubset({0, 1}):
        raise ValueError("Expected labels 0=fake and 1=real.")

    # Keep the original text as the primary model input.
    df["original_content"] = (
        df["title"].fillna("").astype(str).str.strip()
        + " "
        + df["text"].fillna("").astype(str).str.strip()
    ).str.strip()

    # Experiment A: remove selected formatting boilerplate.
    df["no_boilerplate"] = df["original_content"].map(
        remove_boilerplate
    )

    # Experiment B: additionally remove selected publisher names.
    # This is a diagnostic intervention, not a recommended final
    # preprocessing rule.
    df["no_publisher_names"] = df["no_boilerplate"].map(
        remove_publisher_names
    )

    variants = {
        "Original": "original_content",
        "Boilerplate removed": "no_boilerplate",
        "Boilerplate and publisher names removed": "no_publisher_names",
    }

    # Use the same row indices for all variants to make comparisons fair.
    train_idx, test_idx = train_test_split(
        df.index,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=df["label"],
    )

    y_train = df.loc[train_idx, "label"]
    y_test = df.loc[test_idx, "label"]

    classifiers = {
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

    results = []

    for variant_name, text_column in variants.items():
        print(f"\n===== {variant_name} =====")

        X_train = df.loc[train_idx, text_column]
        X_test = df.loc[test_idx, text_column]

        print("Empty train texts:", int(X_train.str.len().eq(0).sum()))
        print("Empty test texts:", int(X_test.str.len().eq(0).sum()))

        for model_name, classifier in classifiers.items():
            pipeline = build_pipeline(classifier)
            pipeline.fit(X_train, y_train)

            predictions = pipeline.predict(X_test)
            scores = pipeline.decision_function(X_test)

            row = {
                "variant": variant_name,
                "model": model_name,
                "accuracy": accuracy_score(y_test, predictions),
                "precision": precision_score(
                    y_test, predictions, zero_division=0
                ),
                "recall": recall_score(
                    y_test, predictions, zero_division=0
                ),
                "f1": f1_score(
                    y_test, predictions, zero_division=0
                ),
                "roc_auc": roc_auc_score(y_test, scores),
                "train_rows": len(train_idx),
                "test_rows": len(test_idx),
            }
            results.append(row)

            print(
                f"{model_name}: "
                f"Accuracy={row['accuracy']:.4f}, "
                f"F1={row['f1']:.4f}, "
                f"ROC-AUC={row['roc_auc']:.4f}"
            )

    results_df = pd.DataFrame(results)
    output_path = REPORT_DIR / "cue_ablation_metrics.csv"
    results_df.to_csv(output_path, index=False)

    print("\n===== COMPARISON TABLE =====")
    print(
        results_df[
            ["variant", "model", "accuracy", "precision",
             "recall", "f1", "roc_auc"]
        ].to_string(index=False)
    )

    print("\nSaved:", output_path.resolve())


if __name__ == "__main__":
    main()
