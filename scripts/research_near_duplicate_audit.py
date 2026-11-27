
from pathlib import Path
import json
import re

import pandas as pd
from datasketch import MinHash, MinHashLSH
from sklearn.model_selection import GroupShuffleSplit


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


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}"
        )

    df = pd.read_csv(DATA_PATH)

    required = {"title", "text", "label"}
    if not required.issubset(df.columns):
        raise ValueError(
            f"Required columns: {sorted(required)}"
        )

    # Normalize text and labels.
    df["title_norm"] = df["title"].map(normalize)
    df["text_norm"] = df["text"].map(normalize)
    df["label"] = pd.to_numeric(
        df["label"], errors="raise"
    ).astype(int)

    # Remove exact normalized title-body duplicates.
    # Keep empty-body rows for split assignment.
    df = df.drop_duplicates(
        subset=["title_norm", "text_norm"]
    ).reset_index(drop=True)

    # Group by article body. Empty bodies use normalized title
    # as a fallback group key, matching the previous experiment.
    df["group_key"] = df["text_norm"].where(
        df["text_norm"].ne(""),
        "empty-body:" + df["title_norm"],
    )

    # Recreate the article-body group split.
    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )

    train_idx, test_idx = next(
        splitter.split(
            df,
            df["label"],
            groups=df["group_key"],
        )
    )

    df["split"] = "train"
    df.loc[test_idx, "split"] = "test"

    # Empty bodies remain in the split but are excluded from
    # the near-duplicate similarity audit.
    word_counts = df["text_norm"].str.findall(
        r"\b[a-z0-9]+\b"
    ).str.len()

    eligible = df.loc[
        df["text_norm"].ne("")
        & (word_counts >= MIN_TOKENS)
    ].copy().reset_index(drop=True)

    print("=" * 60)
    print("NEAR-DUPLICATE ARTICLE AUDIT")
    print("=" * 60)
    print(f"Deduplicated dataset rows: {len(df):,}")
    print(
        "Empty article-body rows:",
        int(df["text_norm"].eq("").sum()),
    )
    print(
        f"Rows audited (at least {MIN_TOKENS} words): "
        f"{len(eligible):,}"
    )
    print(
        "Train rows:",
        int((df["split"] == "train").sum()),
    )
    print(
        "Test rows:",
        int((df["split"] == "test").sum()),
    )
    print("Building MinHash LSH index...")

    lsh = MinHashLSH(
        threshold=SIMILARITY_THRESHOLD,
        num_perm=NUM_PERM,
    )

    shingles_by_id = {}
    candidate_pairs = set()

    for i, text in enumerate(eligible["text_norm"]):
        shingle_set = shingles(text)

        if not shingle_set:
            continue

        mh = make_minhash(shingle_set)

        # Query before insertion to avoid self-matches.
        for candidate in lsh.query(mh):
            a, b = sorted((int(candidate), i))

            if a != b:
                candidate_pairs.add((a, b))

        lsh.insert(str(i), mh)
        shingles_by_id[i] = shingle_set

        if (i + 1) % 5000 == 0:
            print(
                f"Indexed {i + 1:,} / {len(eligible):,}"
            )

    print(f"Candidate pairs found: {len(candidate_pairs):,}")
    print("Verifying exact shingle similarities...")

    records = []

    for a, b in sorted(candidate_pairs):
        set_a = shingles_by_id[a]
        set_b = shingles_by_id[b]

        union_size = len(set_a | set_b)

        if union_size == 0:
            continue

        # Exact Jaccard similarity for each retrieved candidate.
        similarity = len(set_a & set_b) / union_size

        if similarity < SIMILARITY_THRESHOLD:
            continue

        row_a = eligible.iloc[a]
        row_b = eligible.iloc[b]

        records.append({
            "similarity": round(similarity, 4),
            "split_a": row_a["split"],
            "split_b": row_b["split"],
            "cross_split": row_a["split"] != row_b["split"],
            "label_a": int(row_a["label"]),
            "label_b": int(row_b["label"]),
            "labels_match": int(row_a["label"])
            == int(row_b["label"]),
            "title_a": row_a["title"],
            "title_b": row_b["title"],
            "body_length_a": len(row_a["text_norm"]),
            "body_length_b": len(row_b["text_norm"]),
            "row_a": int(a),
            "row_b": int(b),
        })

    columns = [
        "similarity",
        "split_a",
        "split_b",
        "cross_split",
        "label_a",
        "label_b",
        "labels_match",
        "title_a",
        "title_b",
        "body_length_a",
        "body_length_b",
        "row_a",
        "row_b",
    ]

    result = pd.DataFrame(records, columns=columns)

    if not result.empty:
        result = result.sort_values(
            ["cross_split", "similarity"],
            ascending=[False, False],
        ).reset_index(drop=True)

    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    pairs_path = REPORT_DIR / "near_duplicate_pairs.csv"
    result.to_csv(pairs_path, index=False)

    cross_split_pairs = (
        result[result["cross_split"]]
        if not result.empty
        else result.copy()
    )

    summary = {
        "dataset": "WELFake_Dataset.csv",
        "deduplicated_rows": int(len(df)),
        "empty_body_rows_retained_for_split": int(
            df["text_norm"].eq("").sum()
        ),
        "eligible_rows_min_40_words": int(len(eligible)),
        "train_rows_total": int((df["split"] == "train").sum()),
        "test_rows_total": int((df["split"] == "test").sum()),
        "minhash_permutations": NUM_PERM,
        "candidate_pairs": int(len(candidate_pairs)),
        "verified_pairs_similarity_ge_threshold": int(len(result)),
        "verified_cross_split_pairs": int(
            cross_split_pairs.shape[0]
        ),
        "cross_split_pairs_with_different_labels": int(
            (
                cross_split_pairs["labels_match"] == False
            ).sum()
        ) if not cross_split_pairs.empty else 0,
        "similarity_method": (
            "Jaccard similarity of word 5-gram shingles"
        ),
        "similarity_threshold": SIMILARITY_THRESHOLD,
        "minimum_article_words": MIN_TOKENS,
        "split_method": "GroupShuffleSplit",
        "test_size": TEST_SIZE,
        "random_state": RANDOM_STATE,
        "note": (
            "Empty-body rows are retained for split assignment but "
            "excluded from similarity comparisons. LSH candidate "
            "retrieval is approximate and can miss similar pairs. "
            "Cross-split pairs indicate potential overlap, not "
            "confirmed evaluation leakage. Inspect pairs manually."
        ),
    }

    summary_path = REPORT_DIR / "near_duplicate_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    print("\n" + "=" * 60)
    print("AUDIT SUMMARY")
    print("=" * 60)
    print(json.dumps(summary, indent=2))
    print("\nSaved pairs:", pairs_path)
    print("Saved summary:", summary_path)


if __name__ == "__main__":
    main()
