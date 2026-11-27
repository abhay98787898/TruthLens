
from pathlib import Path
import json

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC


ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "WELFake_Dataset.csv"
ARTIFACT_DIR = ROOT / "artifacts"

def normalize(series):
    return (
        series.fillna("")
        .astype(str)
        .str.lower()
        .str.strip()
        .str.replace(r"\s+", " ", regex=True)
    )

def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)

    required = {"title", "text", "label"}
    if not required.issubset(df.columns):
        raise ValueError(f"Required columns: {sorted(required)}")

    df["title_norm"] = normalize(df["title"])
    df["text_norm"] = normalize(df["text"])
    df["label"] = pd.to_numeric(
        df["label"], errors="raise"
    ).astype(int)

    # Empty article text ko prediction ke training data se exclude karein.
    df = df[df["text_norm"].ne("")].copy()

    # Same normalized title + body ki duplicate copies remove karein.
    df = df.drop_duplicates(
        subset=["title_norm", "text_norm"]
    ).copy()

    # Headline aur article dono model ko input milenge.
    df["content"] = (
        df["title_norm"] + " " + df["text_norm"]
    ).str.strip()

    X = df["content"]
    y = df["label"]

    pipeline = Pipeline([
        (
            "tfidf",
            TfidfVectorizer(
                max_features=50000,
                ngram_range=(1, 2),
                min_df=2,
                sublinear_tf=True,
                stop_words="english",
                dtype="float32",
            ),
        ),
        ("classifier", LinearSVC()),
    ])

    print(f"Training rows: {len(df):,}")
    print("Training final Linear SVM on cleaned dataset...")

    pipeline.fit(X, y)

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    model_path = ARTIFACT_DIR / "final_svm_pipeline.joblib"
    joblib.dump(pipeline, model_path)

    metadata = {
        "model": "LinearSVC",
        "vectorizer": "TF-IDF",
        "input": "headline + article text",
        "label_mapping": {"0": "Fake", "1": "Real"},
        "training_rows": int(len(df)),
        "removed_empty_text_rows_and_exact_duplicates": True,
        "dataset": "WELFake_Dataset.csv",
    }

    metadata_path = ARTIFACT_DIR / "model_metadata.json"
    metadata_path.write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    print("\nFinal model saved:", model_path)
    print("Metadata saved:", metadata_path)
    print("Label mapping: 0 = Fake, 1 = Real")
    print("Training completed successfully.")

if __name__ == "__main__":
    main()
