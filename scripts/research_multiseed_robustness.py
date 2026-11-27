
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from datasketch import MinHash, MinHashLSH
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GroupShuffleSplit
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)


# --------------------------------------------------
# Configuration
# --------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "WELFake_Dataset.csv"
REPORT_DIR = ROOT / "reports"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

SEEDS = [42, 21, 7]
TEST_SIZE = 0.20

NUM_PERM = 64
JACCARD_THRESHOLD = 0.85
MIN_TOKENS = 40

MAX_FEATURES = 50_000
NGRAM_RANGE = (1, 2)
MIN_DF = 2

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


# --------------------------------------------------
# Text normalization and source-cue removal
# --------------------------------------------------

def normalize_text(value) -> str:
    if pd.isna(value):
        return ""
    value = str(value).lower()
    value = re.sub(r"\s+", " ", value)
    return value.strip()


def remove_source_cues(text: str) -> str:
    """Remove selected publisher/credit cues for an ablation experiment."""
    cleaned = text

    for term in SOURCE_TERMS:
        cleaned = re.sub(
            rf"(?<!\w){re.escape(term)}(?!\w)",
            " ",
            cleaned,
            flags=re.IGNORECASE,
        )

    # Remove common image/photo credit phrases.
    cleaned = re.sub(
        r"\b(?:featured\s+)?(?:image|photo|picture)\s*:"
        r"|\bimage\s+via\b"
        r"|\bphoto\s+by\b"
        r"|\bimage\s+credit\b",
        " ",
        cleaned,
        flags=re.IGNORECASE,
    )

    return re.sub(r"\s+", " ", cleaned).strip()


# --------------------------------------------------
# Near-duplicate grouping helpers
# --------------------------------------------------

class UnionFind:
    def __init__(self, n: int):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        root_a = self.find(a)
        root_b = self.find(b)

        if root_a == root_b:
            return

        if self.rank[root_a] < self.rank[root_b]:
            root_a, root_b = root_b, root_a

        self.parent[root_b] = root_a

        if self.rank[root_a] == self.rank[root_b]:
            self.rank[root_a] += 1


def word_shingles(text: str, n: int = 5) -> set[str]:
    # Match the original group's regex-based tokenization.
    words = re.findall(r"\b[a-z0-9]+\b", text)

    if len(words) < n:
        return set()

    return {
        " ".join(words[i:i + n])
        for i in range(len(words) - n + 1)
    }


def make_minhash(shingles: set[str]) -> MinHash:
    mh = MinHash(num_perm=NUM_PERM)
    for shingle in shingles:
        mh.update(shingle.encode("utf-8"))
    return mh


def build_near_duplicate_groups(
    df: pd.DataFrame,
) -> tuple[np.ndarray, dict]:
    """
    Match the original near-duplicate grouping methodology:
    - Regex-based tokenization
    - Minimum 40 regex tokens for LSH
    - Exact normalized body grouping
    - Empty bodies grouped by normalized title
    - MinHash LSH candidate search
    - Exact Jaccard verification on 5-word shingles

    LSH is approximate and may miss some near-duplicate pairs.
    """
    n = len(df)
    union_find = UnionFind(n)

    # 1. Group identical non-empty bodies.
    # Empty bodies are grouped by title, matching the original script.
    body_to_first_row: dict[str, int] = {}
    empty_title_to_first_row: dict[str, int] = {}

    for idx, row in enumerate(df.itertuples(index=False)):
        body = row.body_norm
        title = row.title_norm

        if body:
            if body in body_to_first_row:
                union_find.union(idx, body_to_first_row[body])
            else:
                body_to_first_row[body] = idx
        elif title:
            if title in empty_title_to_first_row:
                union_find.union(idx, empty_title_to_first_row[title])
            else:
                empty_title_to_first_row[title] = idx

    # 2. Build the LSH index from eligible articles only.
    lsh = MinHashLSH(
        threshold=JACCARD_THRESHOLD,
        num_perm=NUM_PERM,
    )

    eligible_shingles: dict[int, set[str]] = {}
    candidate_pairs: set[tuple[int, int]] = set()

    for idx, body in enumerate(df["body_norm"]):
        tokens = re.findall(r"\b[a-z0-9]+\b", body)

        if not body or len(tokens) < MIN_TOKENS:
            continue

        shingles = word_shingles(body, n=5)
        if not shingles:
            continue

        mh = make_minhash(shingles)

        # Query before inserting to avoid matching the row to itself.
        for matched_key in lsh.query(mh):
            other_idx = int(matched_key)
            candidate_pairs.add(
                (min(idx, other_idx), max(idx, other_idx))
            )

        lsh.insert(str(idx), mh)
        eligible_shingles[idx] = shingles

    # 3. Verify candidates using exact Jaccard similarity.
    verified_pairs = 0

    for left, right in sorted(candidate_pairs):
        left_shingles = eligible_shingles.get(left)
        right_shingles = eligible_shingles.get(right)

        if not left_shingles or not right_shingles:
            continue

        union_size = len(left_shingles | right_shingles)
        if union_size == 0:
            continue

        similarity = (
            len(left_shingles & right_shingles) / union_size
        )

        if similarity >= JACCARD_THRESHOLD:
            union_find.union(left, right)
            verified_pairs += 1

    # 4. Convert connected components into group IDs.
    roots = [union_find.find(i) for i in range(n)]
    _, group_ids = np.unique(roots, return_inverse=True)

    group_sizes = pd.Series(group_ids).value_counts()

    stats = {
        "rows": int(n),
        "eligible_rows": int(len(eligible_shingles)),
        "candidate_pairs": int(len(candidate_pairs)),
        "verified_near_duplicate_pairs": int(verified_pairs),
        "connected_groups": int(len(group_sizes)),
        "largest_group_size": int(group_sizes.max()),
    }

    return group_ids, stats


# --------------------------------------------------
# Model training and evaluation
# --------------------------------------------------

def evaluate_variant(
    train_texts: pd.Series,
    test_texts: pd.Series,
    y_train: pd.Series,
    y_test: pd.Series,
    seed: int,
) -> dict:
    vectorizer = TfidfVectorizer(
        max_features=MAX_FEATURES,
        ngram_range=NGRAM_RANGE,
        min_df=MIN_DF,
        stop_words="english",
        sublinear_tf=True,
        dtype=np.float32,
    )

    X_train = vectorizer.fit_transform(train_texts)
    X_test = vectorizer.transform(test_texts)

    model = LinearSVC(random_state=seed)
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    decision_scores = model.decision_function(X_test)

    return {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "precision": float(
            precision_score(y_test, predictions, zero_division=0)
        ),
        "recall": float(
            recall_score(y_test, predictions, zero_division=0)
        ),
        "f1": float(f1_score(y_test, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, decision_scores)),
        "features": int(X_train.shape[1]),
    }


def main() -> None:
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}\n"
            "Check that data/WELFake_Dataset.csv exists."
        )

    print("=" * 65)
    print("MULTI-SEED ROBUSTNESS ANALYSIS")
    print("=" * 65)

    df = pd.read_csv(DATA_PATH)

    required_columns = {"title", "text", "label"}
    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(f"Dataset is missing columns: {missing_columns}")

    df = df[["title", "text", "label"]].copy()
    df["title_norm"] = df["title"].map(normalize_text)
    df["body_norm"] = df["text"].map(normalize_text)

    # Preserve the earlier experiment's exact title/body deduplication.
    df = df.drop_duplicates(
        subset=["title_norm", "body_norm"],
        keep="first",
    ).reset_index(drop=True)

    df["label"] = pd.to_numeric(df["label"], errors="raise").astype(int)

    if not set(df["label"].unique()).issubset({0, 1}):
        raise ValueError("Expected binary labels: 0=fake, 1=real.")

    df["original_text"] = (
        df["title_norm"] + " " + df["body_norm"]
    ).str.strip()

    df["cleaned_text"] = df["original_text"].map(remove_source_cues)

    print(f"Rows after exact title/body deduplication: {len(df):,}")
    print("Building near-duplicate connected groups once...")

    groups, grouping_stats = build_near_duplicate_groups(df)

    print(f"Eligible rows for near-duplicate search: {grouping_stats['eligible_rows']:,}")
    print(f"LSH candidate pairs: {grouping_stats['candidate_pairs']:,}")
    print(
        "Verified near-duplicate pairs: "
        f"{grouping_stats['verified_near_duplicate_pairs']:,}"
    )
    print(f"Connected groups: {grouping_stats['connected_groups']:,}")
    print(f"Largest group: {grouping_stats['largest_group_size']:,}")
    print()

    all_results = []

    for seed in SEEDS:
        splitter = GroupShuffleSplit(
            n_splits=1,
            test_size=TEST_SIZE,
            random_state=seed,
        )

        train_idx, test_idx = next(
            splitter.split(df, df["label"], groups=groups)
        )

        train_groups = set(groups[train_idx])
        test_groups = set(groups[test_idx])
        shared_groups = train_groups.intersection(test_groups)

        if shared_groups:
            raise RuntimeError(
                f"Group leakage detected for seed {seed}: "
                f"{len(shared_groups)} shared groups."
            )

        train_df = df.iloc[train_idx]
        test_df = df.iloc[test_idx]

        y_train = train_df["label"]
        y_test = test_df["label"]

        print("-" * 65)
        print(
            f"Seed {seed}: train={len(train_idx):,}, "
            f"test={len(test_idx):,}, shared groups=0"
        )

        variants = {
            "baseline_original_text": (
                train_df["original_text"],
                test_df["original_text"],
            ),
            "source_and_boilerplate_cues_removed": (
                train_df["cleaned_text"],
                test_df["cleaned_text"],
            ),
        }

        for variant_name, (train_texts, test_texts) in variants.items():
            metrics = evaluate_variant(
                train_texts=train_texts,
                test_texts=test_texts,
                y_train=y_train,
                y_test=y_test,
                seed=seed,
            )

            result = {
                "seed": seed,
                "variant": variant_name,
                "train_rows": int(len(train_idx)),
                "test_rows": int(len(test_idx)),
                "train_groups": int(len(train_groups)),
                "test_groups": int(len(test_groups)),
                "shared_groups": int(len(shared_groups)),
                **metrics,
            }
            all_results.append(result)

            print(
                f"{variant_name}: "
                f"Accuracy={metrics['accuracy']:.4f}, "
                f"F1={metrics['f1']:.4f}, "
                f"ROC-AUC={metrics['roc_auc']:.4f}"
            )

    per_seed = pd.DataFrame(all_results)

    # Calculate paired F1 difference for each seed.
    f1_wide = per_seed.pivot(
        index="seed",
        columns="variant",
        values="f1",
    )

    baseline_name = "baseline_original_text"
    cleaned_name = "source_and_boilerplate_cues_removed"

    f1_difference = (
        f1_wide[baseline_name] - f1_wide[cleaned_name]
    ).rename("baseline_minus_cue_removed_f1")

    per_seed_path = REPORT_DIR / "multiseed_robustness_corrected_per_seed.csv"
    per_seed.to_csv(per_seed_path, index=False)

    summary = (
        per_seed.groupby("variant")[
            ["accuracy", "precision", "recall", "f1", "roc_auc"]
        ]
        .agg(["mean", "std"])
    )

    summary.columns = [
        f"{metric}_{stat}"
        for metric, stat in summary.columns
    ]
    summary = summary.reset_index()

    summary_path = REPORT_DIR / "multiseed_robustness_corrected_summary.csv"
    summary.to_csv(summary_path, index=False)

    differences_path = REPORT_DIR / "multiseed_f1_paired_differences_corrected.csv"
    f1_difference.reset_index().to_csv(differences_path, index=False)

    summary_json = {
        "seeds": SEEDS,
        "test_size": TEST_SIZE,
        "grouping_method": (
            "Exact normalized article-body grouping plus approximate "
            "MinHash LSH candidates verified with word 5-gram Jaccard."
        ),
        "near_duplicate_threshold": JACCARD_THRESHOLD,
        "min_tokens_for_near_duplicate_search": MIN_TOKENS,
        "grouping_stats": grouping_stats,
        "per_seed_results": all_results,
        "paired_f1_difference": {
            "definition": "baseline_original_text F1 minus cue-removed F1",
            "per_seed": {
                str(seed): float(value)
                for seed, value in f1_difference.items()
            },
            "mean": float(f1_difference.mean()),
            "std_sample": (
                float(f1_difference.std(ddof=1))
                if len(f1_difference) > 1
                else 0.0
            ),
        },
        "limitations": [
            "Different seeds are different splits of the same dataset, "
            "not independent datasets.",
            "MinHash LSH is approximate and may miss near duplicates.",
            "This experiment does not establish source, topic, or temporal "
            "independence.",
            "Source-cue removal uses a manually selected list and is not "
            "proof that the removed cues caused leakage.",
        ],
    }

    json_path = REPORT_DIR / "multiseed_robustness_corrected_summary.json"
    json_path.write_text(
        json.dumps(summary_json, indent=2),
        encoding="utf-8",
    )

    print()
    print("=" * 65)
    print("MULTI-SEED SUMMARY (mean and sample standard deviation)")
    print("=" * 65)

    for _, row in summary.iterrows():
        print(f"\nVariant: {row['variant']}")
        for metric in ["accuracy", "precision", "recall", "f1", "roc_auc"]:
            print(
                f"  {metric}: "
                f"{row[f'{metric}_mean']:.4f} "
                f"+/- {row[f'{metric}_std']:.4f}"
            )

    print("\nPaired F1 difference (baseline - cue-removed):")
    for seed, value in f1_difference.items():
        print(f"  Seed {seed}: {value:.4f}")
    print(f"  Mean: {f1_difference.mean():.4f}")
    print(f"  Sample std: {f1_difference.std(ddof=1):.4f}")

    print("\nReports saved:")
    print(f"  {per_seed_path}")
    print(f"  {summary_path}")
    print(f"  {differences_path}")
    print(f"  {json_path}")


if __name__ == "__main__":
    main()
