
from pathlib import Path
import re
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "WELFake_Dataset.csv"
PAIRS_PATH = ROOT / "reports" / "cross_split_pairs_for_review.csv"
OUTPUT_PATH = ROOT / "reports" / "label_conflict_body_review.csv"


def normalize(value):
    if pd.isna(value):
        return ""
    return re.sub(r"\s+", " ", str(value).lower()).strip()


def main():
    pairs = pd.read_csv(PAIRS_PATH)
    df = pd.read_csv(DATA_PATH)

    df["title_norm"] = df["title"].map(normalize)
    df["text_norm"] = df["text"].map(normalize)
    df["label"] = pd.to_numeric(
        df["label"], errors="raise"
    ).astype(int)

    conflicts = pairs[
        pairs["label_a"] != pairs["label_b"]
    ].copy()

    if conflicts.empty:
        print("No different-label pairs found.")
        return

    reviewed = []

    for number, (_, pair) in enumerate(
        conflicts.iterrows(), start=1
    ):
        title_a = normalize(pair["title_a"])
        title_b = normalize(pair["title_b"])

        matched = df[
            df["title_norm"].isin([title_a, title_b])
        ]

        print("\n" + "=" * 70)
        print(f"CONFLICT {number}")
        print("Similarity:", pair["similarity"])
        print("Title A:", pair["title_a"])
        print("Title B:", pair["title_b"])
        print("Labels in pair:", pair["label_a"], "/", pair["label_b"])
        print("Original rows with matching titles:", len(matched))

        for index, row in matched.iterrows():
            body = row["text_norm"]

            print("\nOriginal dataset row:", index)
            print("Label:", row["label"])
            print("Body length:", len(body))
            print("Body preview:", body[:700] or "[EMPTY BODY]")

            reviewed.append({
                "conflict_number": number,
                "similarity": pair["similarity"],
                "title": row["title"],
                "label": int(row["label"]),
                "body_length": len(body),
                "body_preview": body[:1000],
                "body_normalized": body,
            })

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(reviewed).to_csv(
        OUTPUT_PATH, index=False
    )

    print("\nReview file saved to:", OUTPUT_PATH)


if __name__ == "__main__":
    main()
