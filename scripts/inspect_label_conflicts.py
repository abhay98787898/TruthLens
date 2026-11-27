
from pathlib import Path

import pandas as pd

DATA_PATH = Path("data/WELFake_Dataset.csv")
OUTPUT_PATH = Path("reports/label_conflicts.csv")


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
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH.resolve()}"
        )

    df = pd.read_csv(DATA_PATH)

    required_columns = {"title", "text", "label"}
    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(
            f"Dataset is missing required columns: "
            f"{sorted(missing_columns)}"
        )

    df["text_norm"] = normalize(df["text"])
    df["label"] = pd.to_numeric(
        df["label"], errors="raise"
    ).astype(int)

    # Empty article bodies ko conflict analysis se exclude karein.
    empty_body_mask = df["text_norm"].eq("")
    empty_body_rows = int(empty_body_mask.sum())

    non_empty_df = df.loc[~empty_body_mask].copy()

    # Check karein ki same normalized, non-empty body ke
    # records mein multiple labels hain ya nahi.
    label_counts = non_empty_df.groupby(
        "text_norm"
    )["label"].nunique()

    conflicting_bodies = label_counts[
        label_counts > 1
    ].index

    conflicts = non_empty_df[
        non_empty_df["text_norm"].isin(
            conflicting_bodies
        )
    ].copy()

    print("=" * 60)
    print("ARTICLE-BODY LABEL CONFLICT INSPECTION")
    print("=" * 60)

    print("Total dataset rows:", len(df))
    print("Rows with empty article bodies:", empty_body_rows)
    print(
        "Conflicting non-empty article-body groups:",
        len(conflicting_bodies),
    )
    print("Rows involved in non-empty conflicts:", len(conflicts))

    if conflicts.empty:
        print(
            "\nNo conflicting non-empty normalized "
            "article-body groups found."
        )

        # Purani report ko galti se current result na samjhein.
        OUTPUT_PATH.parent.mkdir(
            parents=True, exist_ok=True
        )

        conflicts.drop(
            columns=["text_norm"]
        ).to_csv(OUTPUT_PATH, index=False)

        print("Empty conflict report saved to:", OUTPUT_PATH.resolve())
        return

    conflicts["body_length"] = (
        conflicts["text_norm"].str.len()
    )

    for group_number, (body, group) in enumerate(
        conflicts.groupby("text_norm", sort=False),
        start=1,
    ):
        print(f"\n===== CONFLICT GROUP {group_number} =====")
        print("Normalized body length:", len(body))
        print(
            "Distinct labels:",
            sorted(group["label"].unique()),
        )

        print(
            group[
                ["title", "label", "body_length"]
            ].to_string(index=False)
        )

        print("\nArticle body excerpt:")
        print(body[:800])

    OUTPUT_PATH.parent.mkdir(
        parents=True, exist_ok=True
    )

    conflicts.drop(
        columns=["text_norm"]
    ).to_csv(OUTPUT_PATH, index=False)

    print(
        "\nSaved full conflicting rows to:",
        OUTPUT_PATH.resolve(),
    )


if __name__ == "__main__":
    main()
