
from pathlib import Path

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
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


def normalize(series):
    return (
        series.fillna("")
        .astype(str)
        .str.lower()
        .str.strip()
        .str.replace(r"\s+", " ", regex=True)
    )


def main():
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

    # Audit exact normalized article bodies for conflicting labels.
    body_groups = df.groupby("text_norm", dropna=False)["label"].nunique()
    conflicts = int((body_groups > 1).sum())
    print("Article-body groups with conflicting labels:", conflicts)

    # Remove exact normalized title-text duplicates.
    df = df.drop_duplicates(
        subset=["title_norm", "text_norm"]
    ).copy()

    df["content"] = (
        df["title"].fillna("").astype(str).str.strip()
        + " "
        + df["text"].fillna("").astype(str).str.strip()
    ).str.strip()

    df = df[df["content"].str.len() > 0].copy()

    # Group by normalized article body.
    # For records with no body, fall back to normalized title.
    df["group_key"] = df["text_norm"]
    no_body = df["group_key"].eq("")
    df.loc[no_body, "group_key"] = (
        "empty-body:" + df.loc[no_body, "title_norm"]
    )

    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=0.20,
        random_state=42,
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
    assert not overlap, "Article-body group overlap detected!"

    print("\nTrain rows:", len(train))
    print("Test rows:", len(test))
    print("Shared article-body groups:", len(overlap))

    print("\nTrain label proportions:")
    print(train["label"].value_counts(normalize=True).sort_index().round(4))

    print("\nTest label proportions:")
    print(test["label"].value_counts(normalize=True).sort_index().round(4))

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
        (
            "classifier",
            LinearSVC(C=1.0, random_state=42),
        ),
    ])

    print("\nTraining Linear SVM...")
    pipeline.fit(train["content"], train["label"])

    predictions = pipeline.predict(test["content"])
    scores = pipeline.decision_function(test["content"])

    metrics = {
        "split": "Normalized article-body groups",
        "model": "Linear SVM",
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

    print("\nMetrics:")
    print(pd.Series(metrics).to_string())

    print("\nClassification report:")
    print(classification_report(
        test["label"],
        predictions,
        labels=[0, 1],
        target_names=["Fake (0)", "Real (1)"],
        zero_division=0,
    ))

    pd.DataFrame([metrics]).to_csv(
        REPORT_DIR / "text_group_split_metrics.csv",
        index=False,
    )

    errors = pd.DataFrame({
        "title": test["title"].values,
        "true_label": test["label"].values,
        "predicted_label": predictions,
        "text_excerpt": test["text"].fillna("").astype(str)
        .str.slice(0, 500).values,
    })

    errors = errors[
        errors["true_label"] != errors["predicted_label"]
    ]

    errors.to_csv(
        REPORT_DIR / "text_group_split_errors.csv",
        index=False,
    )

    print("\nSaved reports in:", REPORT_DIR.resolve())


if __name__ == "__main__":
    main()
