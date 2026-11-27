
from pathlib import Path
import json
import re
from collections import defaultdict

import pandas as pd
from datasketch import MinHash, MinHashLSH
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.svm import LinearSVC


ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "WELFake_Dataset.csv"
REPORT_DIR = ROOT / "reports"

NUM_PERM = 64
SIMILARITY_THRESHOLD = 0.85
MIN_TOKENS = 40
RANDOM_STATE = 42
TEST_SIZE = 0.20


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

    def find(self, item):
        while self.parent[item] != item:
            self.parent[item] = self.parent[self.parent[item]]
            item = self.parent[item]
        return item

    def union(self, a, b):
        root_a = self.find(a)
        root_b = self.find(b)

        if root_a == root_b:
            return

        if self.rank[root_a] < self.rank[root_b]:
            root_a, root_b = root_b, root_a

        self.parent[root_b] = root_a

        if self.rank[root_a] == self.rank[root_b]:
            self.rank[root_a] += 1


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(DATA_PATH)

    df = pd.read_csv(DATA_PATH)

    required = {"title", "text", "label"}
    if not required.issubset(df.columns):
        raise ValueError(f"Required columns: {sorted(required)}")

    # Match the existing audit's normalization and deduplication.
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

    n_rows = len(df)
    print("=" * 65)
    print("NEAR-DUPLICATE-AWARE GROUP SPLIT")
    print("=" * 65)
    print("Deduplicated rows:", f"{n_rows:,}")

    # Eligible articles are used for approximate similarity search.
    word_counts = df["text_norm"].str.findall(
        r"\b[a-z0-9]+\b"
    ).str.len()

    eligible_positions = [
        int(i)
        for i in df.index[
            df["text_norm"].ne("")
            & (word_counts >= MIN_TOKENS)
        ]
    ]

    print("Eligible articles:", f"{len(eligible_positions):,}")
    print("Building MinHash LSH index...")

    lsh = MinHashLSH(
        threshold=SIMILARITY_THRESHOLD,
        num_perm=NUM_PERM,
    )

    eligible_shingles = {}
    candidate_pairs = set()

    for eligible_id, df_position in enumerate(eligible_positions):
        shingle_set = shingles(df.at[df_position, "text_norm"])

        if not shingle_set:
            continue

        mh = make_minhash(shingle_set)

        for candidate in lsh.query(mh):
            a, b = sorted((int(candidate), eligible_id))
            if a != b:
                candidate_pairs.add((a, b))

        lsh.insert(str(eligible_id), mh)
        eligible_shingles[eligible_id] = shingle_set

        if (eligible_id + 1) % 5000 == 0:
            print(
                f"Indexed {eligible_id + 1:,} / "
                f"{len(eligible_positions):,}"
            )

    print("Candidate pairs:", f"{len(candidate_pairs):,}")
    print("Verifying exact Jaccard similarities...")

    uf = UnionFind(n_rows)

    # First, group identical non-empty normalized article bodies.
    # Empty bodies are grouped by title instead.
    first_body_position = {}
    first_empty_title_position = {}

    for i, row in df.iterrows():
        body = row["text_norm"]

        if body:
            if body in first_body_position:
                uf.union(i, first_body_position[body])
            else:
                first_body_position[body] = i
        else:
            title = row["title_norm"]

            if title:
                if title in first_empty_title_position:
                    uf.union(i, first_empty_title_position[title])
                else:
                    first_empty_title_position[title] = i

    # Add edges for verified near-duplicate pairs.
    verified_pairs = []
    cross_split_not_applicable = 0

    for a, b in sorted(candidate_pairs):
        set_a = eligible_shingles.get(a)
        set_b = eligible_shingles.get(b)

        if not set_a or not set_b:
            continue

        union_size = len(set_a | set_b)
        if not union_size:
            continue

        similarity = len(set_a & set_b) / union_size

        if similarity < SIMILARITY_THRESHOLD:
            continue

        df_a = eligible_positions[a]
        df_b = eligible_positions[b]

        uf.union(df_a, df_b)

        verified_pairs.append({
            "row_a": df_a,
            "row_b": df_b,
            "similarity": round(similarity, 4),
            "label_a": int(df.at[df_a, "label"]),
            "label_b": int(df.at[df_b, "label"]),
            "title_a": df.at[df_a, "title"],
            "title_b": df.at[df_b, "title"],
        })

    print("Verified near-duplicate pairs:", f"{len(verified_pairs):,}")

    # Assign connected-component IDs as group labels.
    root_to_group = {}
    group_ids = []

    for i in range(n_rows):
        root = uf.find(i)

        if root not in root_to_group:
            root_to_group[root] = len(root_to_group)

        group_ids.append(root_to_group[root])

    df["duplicate_group"] = group_ids

    group_sizes = df["duplicate_group"].value_counts()
    largest_group_size = int(group_sizes.max())

    print("Connected groups:", f"{len(group_sizes):,}")
    print("Largest group size:", largest_group_size)

    # Split only by group, never by individual article.
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

    train_groups = set(train["duplicate_group"])
    test_groups = set(test["duplicate_group"])
    shared_groups = train_groups.intersection(test_groups)

    if shared_groups:
        raise RuntimeError(
            f"Group leakage detected: {len(shared_groups)} shared groups."
        )

    # Same basic TF-IDF + Linear SVM family as the earlier experiment.
    # Fit the vectorizer on training data only.
    train_text = (
        train["title_norm"] + " " + train["text_norm"]
    ).str.strip()

    test_text = (
        test["title_norm"] + " " + test["text_norm"]
    ).str.strip()

    vectorizer = TfidfVectorizer(
        max_features=50000,
        ngram_range=(1, 2),
        min_df=2,
        sublinear_tf=True,
        stop_words="english",
        dtype="float32",
    )

    X_train = vectorizer.fit_transform(train_text)
    X_test = vectorizer.transform(test_text)

    y_train = train["label"]
    y_test = test["label"]

    model = LinearSVC(random_state=RANDOM_STATE)
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    decision_scores = model.decision_function(X_test)

    metrics = {
        "experiment": "near_duplicate_connected_component_split",
        "dataset": "WELFake_Dataset.csv",
        "deduplicated_rows": int(n_rows),
        "train_rows": int(len(train)),
        "test_rows": int(len(test)),
        "train_groups": int(len(train_groups)),
        "test_groups": int(len(test_groups)),
        "shared_groups": int(len(shared_groups)),
        "connected_groups_total": int(len(group_sizes)),
        "largest_group_size": largest_group_size,
        "eligible_rows_min_40_words": int(len(eligible_positions)),
        "minhash_permutations": NUM_PERM,
        "candidate_pairs": int(len(candidate_pairs)),
        "verified_near_duplicate_pairs": int(len(verified_pairs)),
        "similarity_threshold": SIMILARITY_THRESHOLD,
        "similarity_method": "Word 5-gram Jaccard, approximate LSH retrieval",
        "split_method": "GroupShuffleSplit over connected components",
        "test_size_requested": TEST_SIZE,
        "random_state": RANDOM_STATE,
        "model": "TF-IDF + LinearSVC",
        "accuracy": float(accuracy_score(y_test, predictions)),
        "precision": float(
            precision_score(y_test, predictions, zero_division=0)
        ),
        "recall": float(
            recall_score(y_test, predictions, zero_division=0)
        ),
        "f1": float(f1_score(y_test, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, decision_scores)),
        "confusion_matrix": confusion_matrix(
            y_test, predictions, labels=[0, 1]
        ).tolist(),
        "label_mapping": {"0": "Fake", "1": "Real"},
        "note": (
            "Connected components include verified LSH candidate pairs "
            "with exact Jaccard >= threshold and identical non-empty "
            "normalized bodies. LSH is approximate and can miss pairs. "
            "This split does not guarantee source, topic, or temporal "
            "independence. Labels were not manually changed."
        ),
    }

    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    summary_path = REPORT_DIR / "near_duplicate_group_split_metrics.json"
    summary_path.write_text(
        json.dumps(metrics, indent=2),
        encoding="utf-8",
    )

    # Save row assignments for reproducibility and inspection.
    assignments = df[
        ["title", "label", "duplicate_group"]
    ].copy()
    assignments["split"] = "train"
    assignments.loc[test_idx, "split"] = "test"
    assignments.to_csv(
        REPORT_DIR / "near_duplicate_group_split_assignments.csv",
        index=False,
    )

    pair_df = pd.DataFrame(verified_pairs)
    pair_df.to_csv(
        REPORT_DIR / "near_duplicate_group_split_pairs.csv",
        index=False,
    )

    errors = test.copy()
    errors["actual_label"] = y_test.to_numpy()
    errors["predicted_label"] = predictions
    errors["decision_score"] = decision_scores
    errors = errors[
        errors["actual_label"] != errors["predicted_label"]
    ]
    errors[
        ["title", "actual_label", "predicted_label", "decision_score"]
    ].to_csv(
        REPORT_DIR / "near_duplicate_group_split_errors.csv",
        index=False,
    )

    report = classification_report(
        y_test,
        predictions,
        labels=[0, 1],
        target_names=["Fake (0)", "Real (1)"],
        zero_division=0,
    )

    print("\n" + "=" * 65)
    print("EVALUATION RESULTS")
    print("=" * 65)
    print(json.dumps(metrics, indent=2))
    print("\nClassification report:\n")
    print(report)
    print("\nSaved:", summary_path)
    print("Saved: reports/near_duplicate_group_split_assignments.csv")
    print("Saved: reports/near_duplicate_group_split_pairs.csv")
    print("Saved: reports/near_duplicate_group_split_errors.csv")


if __name__ == "__main__":
    main()
