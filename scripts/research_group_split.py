
from pathlib import Path

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
)

DATA_PATH = Path("data/WELFake_Dataset.csv")
REPORT_DIR = Path("reports")
REPORT_DIR.mkdir(parents=True, exist_ok=True)
RANDOM_STATE = 42


def normalize(series):
    return (
        series.fillna("")
        .astype(str)
        .str.lower()
        .str.strip()
        .str.replace(r"\s+", " ", regex=True)
    )


def main():
    print("Loading dataset...")
    df = pd.read_csv(DATA_PATH)

    required = {"title", "text", "label"}
    if not required.issubset(df.columns):
        raise ValueError(f"Missing columns: {required - set(df.columns)}")

    df["title_norm"] = normalize(df["title"])
    df["text_norm"] = normalize(df["text"])
    df["label"] = pd.to_numeric(
        df["label"], errors="raise"
    ).astype(int)

    if not set(df["label"].unique()).issubset({0, 1}):
        raise ValueError("Expected labels 0=fake and 1=real.")

    # Remove conflicting exact article pairs before deduplication.
    conflicts = df.groupby(
        ["title_norm", "text_norm"], dropna=False
    )["label"].nunique()

    if (conflicts > 1).any():
        raise ValueError("Conflicting labels found in duplicate pairs.")

    df = df.drop_duplicates(
        subset=["title_norm", "text_norm"]
    ).copy()

    df["content"] = (
        df["title"].fillna("").astype(str).str.strip()
        + " "
        + df["text"].fillna("").astype(str).str.strip()
    ).str.strip()

    df = df[df["content"].str.len() > 0].copy()

    # Use normalized headline as the grouping key.
    # If a headline is missing, use normalized article text instead.
    df["group_key"] = df["title_norm"]
    missing_title = df["group_key"].eq("")
    df.loc[missing_title, "group_key"] = (
        "missing-title:" + df.loc[missing_title, "text_norm"]
    )

    # GroupShuffleSplit keeps each group entirely in train OR test.
    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=0.20,
        random_state=RANDOM_STATE,
    )

    train_idx, test_idx = next(
        splitter.split(
            df["content"],
            df["label"],
            groups=df["group_key"],
        )
    )

    train = df.iloc[train_idx].copy()
    test = df.iloc[test_idx].copy()

    overlap = set(train["group_key"]) & set(test["group_key"])
    assert len(overlap) == 0, "Group overlap detected!"

    print("Total usable rows:", len(df))
    print("Train rows:", len(train))
    print("Test rows:", len(test))
    print("Shared headline groups:", len(overlap))

    print("\nTrain label distribution:")
    print(train["label"].value_counts(normalize=True).sort_index().round(4))

    print("\nTest label distribution:")
    print(test["label"].value_counts(normalize=True).sort_index().round(4))

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

    results = []
    errors = []

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

        pipeline.fit(train["content"], train["label"])

        predictions = pipeline.predict(test["content"])
        scores = pipeline.decision_function(test["content"])

        metrics = {
            "model": name,
            "accuracy": accuracy_score(test["label"], predictions),
            "precision": precision_score(
                test["label"], predictions, zero_division=0
            ),
            "recall": recall_score(
                test["label"], predictions, zero_division=0
            ),
            "f1": f1_score(
                test["label"], predictions, zero_division=0
            ),
            "roc_auc": roc_auc_score(test["label"], scores),
            "train_rows": len(train),
            "test_rows": len(test),
        }
        results.append(metrics)

        print(pd.Series(metrics).to_string())
        print(
            classification_report(
                test["label"],
                predictions,
                labels=[0, 1],
                target_names=["Fake (0)", "Real (1)"],
                zero_division=0,
            )
        )

        error_df = pd.DataFrame({
            "model": name,
            "title": test["title"].values,
            "true_label": test["label"].values,
            "predicted_label": predictions,
            "text_excerpt": test["text"].fillna("").astype(str)
            .str.slice(0, 500).values,
        })

        errors.append(
            error_df[
                error_df["true_label"] != error_df["predicted_label"]
            ]
        )

    results_df = pd.DataFrame(results)
    results_df.to_csv(
        REPORT_DIR / "group_split_metrics.csv", index=False
    )

    pd.concat(errors, ignore_index=True).to_csv(
        REPORT_DIR / "group_split_errors.csv", index=False
    )

    print("\n===== GROUP-SPLIT RESULTS =====")
    print(results_df.round(4).to_string(index=False))
    print("\nSaved reports in:", REPORT_DIR.resolve())


if __name__ == "__main__":
    main()
