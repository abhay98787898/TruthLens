
from pathlib import Path
import re
import json

import pandas as pd
from datasketch import MinHash, MinHashLSH
from sklearn.model_selection import GroupShuffleSplit
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "WELFake_Dataset.csv"
REPORT_DIR = ROOT / "reports"

NUM_PERM = 64
THRESHOLD = 0.85
MIN_TOKENS = 40
RANDOM_STATE = 42
TEST_SIZE = 0.20

# Initial, conservative list of source and boilerplate cues.
# This is an experimental list, not a claim that these terms are
# always misleading or should always be removed.
SOURCE_TERMS = [
    "reuters",
    "breitbart",
    "new york times",
    "washington post",
    "associated press",
    "cnn",
    "fox news",
    "bbc",
    "getty images",
    "featured image",
    "image via",
    "read more:",
]


def normalize(value):
    if pd.isna(value):
        return ""
    return re.sub(r"\s+", " ", str(value).lower()).strip()


def shingles(text, size=5):
    tokens = re.findall(r"\b[a-z0-9]+\b", text)
    if len(tokens) < size:
        return set()
    return {
        " ".join(tokens[i:i + size])
        for i in range(len(tokens) - size + 1)
    }


def make_minhash(shingle_set):
    mh = MinHash(num_perm=NUM_PERM)
    for item in shingle_set:
        mh.update(item.encode("utf-8"))
    return mh


class UnionFind:
    def __init__(self, size):
        self.parent = list(range(size))
        self.rank = [0] * size

    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1


def remove_source_cues(text):
    cleaned = text

    for term in SOURCE_TERMS:
        cleaned = re.sub(
            re.escape(term),
            " ",
            cleaned,
            flags=re.IGNORECASE,
        )

    # Remove common image/caption boilerplate patterns.
    cleaned = re.sub(
        r"\b(?:photo|image|picture)\s+(?:by|via|credit:)\s+[^.]{1,100}",
        " ",
        cleaned,
        flags=re.IGNORECASE,
    )

    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def build_groups(df):
    n = len(df)
    uf = UnionFind(n)

    # Exact article-body grouping.
    first_body = {}
    first_empty_title = {}

    for i, row in df.iterrows():
        body = row["text_norm"]

        if body:
            if body in first_body:
                uf.union(i, first_body[body])
            else:
                first_body[body] = i
        else:
            title = row["title_norm"]
            if title:
                if title in first_empty_title:
                    uf.union(i, first_empty_title[title])
                else:
                    first_empty_title[title] = i

    word_counts = df["text_norm"].str.findall(
        r"\b[a-z0-9]+\b"
    ).str.len()

    eligible_positions = [
        int(i) for i in df.index[
            df["text_norm"].ne("") & (word_counts >= MIN_TOKENS)
        ]
    ]

    lsh = MinHashLSH(threshold=THRESHOLD, num_perm=NUM_PERM)
    shingles_by_id = {}
    candidates = set()

    for eligible_id, df_position in enumerate(eligible_positions):
        shingle_set = shingles(df.at[df_position, "text_norm"])
        if not shingle_set:
            continue

        mh = make_minhash(shingle_set)

        for candidate in lsh.query(mh):
            a, b = sorted((int(candidate), eligible_id))
            if a != b:
                candidates.add((a, b))

        lsh.insert(str(eligible_id), mh)
        shingles_by_id[eligible_id] = shingle_set

    verified = 0

    for a, b in candidates:
        set_a = shingles_by_id.get(a)
        set_b = shingles_by_id.get(b)

        if not set_a or not set_b:
            continue

        union_size = len(set_a | set_b)
        if not union_size:
            continue

        similarity = len(set_a & set_b) / union_size

        if similarity >= THRESHOLD:
            uf.union(eligible_positions[a], eligible_positions[b])
            verified += 1

    root_to_group = {}
    groups = []

    for i in range(n):
        root = uf.find(i)
        if root not in root_to_group:
            root_to_group[root] = len(root_to_group)
        groups.append(root_to_group[root])

    print("Near-duplicate candidate pairs:", len(candidates))
    print("Verified near-duplicate pairs:", verified)
    print("Connected groups:", len(set(groups)))

    return groups


def evaluate_variant(name, train_text, test_text, y_train, y_test):
    vectorizer = TfidfVectorizer(
        max_features=50000,
        ngram_range=(1, 2),
        min_df=2,
        sublinear_tf=True,
        stop_words="english",
    )

    X_train = vectorizer.fit_transform(train_text)
    X_test = vectorizer.transform(test_text)

    model = LinearSVC(random_state=RANDOM_STATE)
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    scores = model.decision_function(X_test)

    return {
        "variant": name,
        "accuracy": accuracy_score(y_test, predictions),
        "precision": precision_score(
            y_test, predictions, zero_division=0
        ),
        "recall": recall_score(
            y_test, predictions, zero_division=0
        ),
        "f1": f1_score(y_test, predictions, zero_division=0),
        "roc_auc": roc_auc_score(y_test, scores),
        "features": len(vectorizer.vocabulary_),
    }


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(DATA_PATH)

    df = pd.read_csv(DATA_PATH)

    required = {"title", "text", "label"}
    if not required.issubset(df.columns):
        raise ValueError(f"Required columns: {sorted(required)}")

    df["title_norm"] = df["title"].map(normalize)
    df["text_norm"] = df["text"].map(normalize)
    df["label"] = pd.to_numeric(
        df["label"], errors="raise"
    ).astype(int)

    if not set(df["label"].unique()).issubset({0, 1}):
        raise ValueError("Expected binary labels 0 and 1.")

    df = df.drop_duplicates(
        subset=["title_norm", "text_norm"]
    ).reset_index(drop=True)

    print("Building near-duplicate groups...")
    df["duplicate_group"] = build_groups(df)

    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )

    train_idx, test_idx = next(
        splitter.split(
            df,
            df["label"],
            groups=df["duplicate_group"],
        )
    )

    train = df.iloc[train_idx].copy()
    test = df.iloc[test_idx].copy()

    if set(train["duplicate_group"]).intersection(
        set(test["duplicate_group"])
    ):
        raise RuntimeError("Near-duplicate groups overlap between splits.")

    y_train = train["label"]
    y_test = test["label"]

    # Variant 1: original title + body.
    train_original = (
        train["title_norm"] + " " + train["text_norm"]
    ).str.strip()
    test_original = (
        test["title_norm"] + " " + test["text_norm"]
    ).str.strip()

    # Variant 2: remove listed source and boilerplate cues.
    train_clean = train_original.map(remove_source_cues)
    test_clean = test_original.map(remove_source_cues)

    results = [
        evaluate_variant(
            "baseline_original_text",
            train_original,
            test_original,
            y_train,
            y_test,
        ),
        evaluate_variant(
            "source_and_boilerplate_cues_removed",
            train_clean,
            test_clean,
            y_train,
            y_test,
        ),
    ]

    result_df = pd.DataFrame(results)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    csv_path = REPORT_DIR / "source_cue_ablation_metrics.csv"
    json_path = REPORT_DIR / "source_cue_ablation_metrics.json"

    result_df.to_csv(csv_path, index=False)
    json_path.write_text(
        json.dumps(results, indent=2),
        encoding="utf-8",
    )

    print("\n" + "=" * 65)
    print("SOURCE-CUE ABLATION RESULTS")
    print("=" * 65)
    print(result_df.to_string(index=False))
    print("\nSaved:", csv_path)
    print("Saved:", json_path)
    print(
        "\nInterpretation caution: performance changes measure the "
        "effect of this specific cue-removal list, not proof of "
        "data leakage or publisher bias."
    )


if __name__ == "__main__":
    main()
