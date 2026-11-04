
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
from sklearn.metrics import classification_report


INPUT_PATH = "outputs/evaluation_results.csv"
LABELLED_PATH = "data/processed/labelled_tickets.csv"
OUTPUT_DIR = Path("outputs")

LABELS = [
    "api",
    "billing",
    "cancellation",
    "complaint",
    "technical",
    "upgrade"
]


def main():

    if not Path(INPUT_PATH).exists():
        raise FileNotFoundError(
            "Run src.evaluation before bias analysis."
        )

    df = pd.read_csv(INPUT_PATH)

    required = [
        "human_label",
        "groq_label",
        "distilbert_label"
    ]

    for column in required:
        if column not in df.columns:
            raise ValueError(f"Missing column: {column}")

    df = df.dropna(subset=required)

    df = df[
        df["human_label"].isin(LABELS)
        & df["groq_label"].isin(LABELS)
        & df["distilbert_label"].isin(LABELS)
    ].copy()

    if df.empty:
        raise ValueError("No valid evaluation records found.")

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    print("\n========== LABEL DISTRIBUTION ==========")

    distribution = df["human_label"].value_counts().reindex(
        LABELS,
        fill_value=0
    )

    print(distribution)

    distribution.plot(
        kind="bar",
        title="Human Label Distribution",
        xlabel="Category",
        ylabel="Number of Tickets"
    )

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "label_distribution.png")
    plt.close()

    print("\n========== MODEL PERFORMANCE ==========")

    for model in ["groq_label", "distilbert_label"]:

        print(f"\n{model.upper()}")

        report = classification_report(
            df["human_label"],
            df[model],
            labels=LABELS,
            zero_division=0,
            output_dict=True
        )

        report_df = pd.DataFrame(report).transpose()

        print(report_df.round(3))

        report_df.to_csv(
            OUTPUT_DIR / f"{model}_classification_report.csv"
        )

    df["groq_error"] = (
        df["human_label"] != df["groq_label"]
    )

    df["distilbert_error"] = (
        df["human_label"] != df["distilbert_label"]
    )

    print("\n========== LABEL NOISE ==========")

    print(
        "Groq disagreements:",
        df["groq_error"].sum()
    )

    print(
        "DistilBERT disagreements:",
        df["distilbert_error"].sum()
    )

    disagreements = df[
        df["groq_error"] | df["distilbert_error"]
    ].copy()

    disagreements.to_csv(
        OUTPUT_DIR / "label_disagreements.csv",
        index=False
    )

    print("\n========== CATEGORY-LEVEL ERRORS ==========")

    error_summary = df.groupby("human_label").agg(
        total_tickets=("human_label", "count"),
        groq_errors=("groq_error", "sum"),
        distilbert_errors=("distilbert_error", "sum")
    )

    print(error_summary)

    error_summary.to_csv(
        OUTPUT_DIR / "category_error_analysis.csv"
    )

    print("\n========== CONFIDENCE ANALYSIS ==========")

    confidence_columns = [
        column for column in [
            "groq_confidence",
            "distilbert_confidence"
        ]
        if column in df.columns
    ]

    for column in confidence_columns:

        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        )

        print(f"\n{column}")

        print(
            df.groupby("human_label")[column]
            .mean()
            .round(2)
        )

    df.to_csv(
        OUTPUT_DIR / "bias_analysis_results.csv",
        index=False
    )

    print("\nBias and noise analysis completed.")
    print("Results saved in:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
