
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
INPUT_PATH = ROOT / "reports" / "near_duplicate_pairs.csv"
OUTPUT_PATH = ROOT / "reports" / "cross_split_pairs_for_review.csv"


def main():
    if not INPUT_PATH.exists():
        raise FileNotFoundError(INPUT_PATH)

    df = pd.read_csv(INPUT_PATH)

    if df.empty:
        print("No verified near-duplicate pairs were found.")
        return

    cross = df[df["cross_split"] == True].copy()

    if cross.empty:
        print("No cross-split pairs found.")
        return

    cross = cross.sort_values(
        "similarity", ascending=False
    )

    cross["label_status"] = cross.apply(
        lambda row: (
            "DIFFERENT LABELS"
            if row["label_a"] != row["label_b"]
            else "Same label"
        ),
        axis=1,
    )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    cross.to_csv(OUTPUT_PATH, index=False)

    print("=" * 65)
    print("CROSS-SPLIT NEAR-DUPLICATE REVIEW")
    print("=" * 65)
    print("Total cross-split pairs:", len(cross))
    print(
        "Pairs with different labels:",
        int((cross["label_a"] != cross["label_b"]).sum()),
    )

    print("\nTOP 20 PAIRS BY SIMILARITY")
    for number, (_, row) in enumerate(
        cross.head(20).iterrows(), start=1
    ):
        print(f"\n--- Pair {number} ---")
        print(f"Similarity: {row['similarity']}")
        print(f"Split: {row['split_a']} / {row['split_b']}")
        print(
            f"Labels: {row['label_a']} / {row['label_b']} "
            f"({row['label_status']})"
        )
        print("Title A:", str(row["title_a"])[:300])
        print("Title B:", str(row["title_b"])[:300])

    different = cross[
        cross["label_a"] != cross["label_b"]
    ]

    if not different.empty:
        print("\nALL DIFFERENT-LABEL PAIRS")
        for _, row in different.iterrows():
            print("\nSimilarity:", row["similarity"])
            print("Title A:", str(row["title_a"])[:500])
            print("Title B:", str(row["title_b"])[:500])
            print("Labels:", row["label_a"], "/", row["label_b"])

    print("\nFull review report saved to:")
    print(OUTPUT_PATH)


if __name__ == "__main__":
    main()
