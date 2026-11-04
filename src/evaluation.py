
import pandas as pd
import matplotlib.pyplot as plt
import torch

from pathlib import Path
from groq import Groq
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay
)

from src.few_shot_labeler import classify_ticket, API_KEY, MODEL_NAME
from src.predict import predict_ticket


HUMAN_PATH = "data/human/human_labels.csv"
OUTPUT_PATH = "outputs/evaluation_results.csv"

LABELS = [
    "api",
    "billing",
    "cancellation",
    "complaint",
    "technical",
    "upgrade"
]


def main():

    if not API_KEY:
        raise ValueError("GROQ_API_KEY missing in .env")

    human_df = pd.read_csv(HUMAN_PATH)

    required_columns = [
        "ticket_id",
        "ticket_text",
        "human_label"
    ]

    for column in required_columns:
        if column not in human_df.columns:
            raise ValueError(f"Missing column: {column}")

    human_df["human_label"] = (
        human_df["human_label"]
        .fillna("")
        .astype(str)
        .str.strip()
        .str.lower()
    )

    human_df = human_df[
        human_df["human_label"].isin(LABELS)
    ].dropna(subset=["ticket_text"])

    if human_df.empty:
        raise ValueError(
            "No completed human labels found. Fill human_label first."
        )

    Path("outputs").mkdir(
        parents=True,
        exist_ok=True
    )

    if Path(OUTPUT_PATH).exists():
        existing_df = pd.read_csv(OUTPUT_PATH)
        records = {
            str(row["ticket_id"]): row.to_dict()
            for _, row in existing_df.iterrows()
        }
    else:
        records = {}

    client = Groq(api_key=API_KEY)

    print("Human-labelled tickets:", len(human_df))
    print("Groq model:", MODEL_NAME)

    for row in human_df.itertuples(index=False):

        ticket_id = str(row.ticket_id)
        ticket_text = str(row.ticket_text)

        existing = records.get(ticket_id, {})

        groq_label = existing.get("groq_label", "")

        if groq_label not in LABELS:

            print("\nGetting Groq prediction:", ticket_id)

            result = classify_ticket(
                client,
                ticket_text
            )

            groq_label = result["predicted_label"]
            groq_confidence = result["confidence_estimate"]

        else:
            groq_confidence = existing.get(
                "groq_confidence",
                None
            )

        distilbert_result = predict_ticket(ticket_text)

        records[ticket_id] = {
            "ticket_id": ticket_id,
            "ticket_text": ticket_text,
            "human_label": row.human_label,
            "groq_label": groq_label,
            "groq_confidence": groq_confidence,
            "distilbert_label": distilbert_result["predicted_label"],
            "distilbert_confidence": distilbert_result["confidence"]
        }

        pd.DataFrame(records.values()).to_csv(
            OUTPUT_PATH,
            index=False
        )

        print("Human:", row.human_label)
        print("Groq:", groq_label)
        print("DistilBERT:", distilbert_result["predicted_label"])

    results = pd.DataFrame(records.values())

    results = results[
        results["ticket_id"].astype(str).isin(
            human_df["ticket_id"].astype(str)
        )
    ].copy()

    results = results.dropna(
        subset=[
            "human_label",
            "groq_label",
            "distilbert_label"
        ]
    )

    results = results[
        results["human_label"].isin(LABELS)
        & results["groq_label"].isin(LABELS)
        & results["distilbert_label"].isin(LABELS)
    ]

    if results.empty:
        raise ValueError("No valid evaluation predictions found.")

    models = {
        "Groq": "groq_label",
        "DistilBERT": "distilbert_label"
    }

    print("\n========== EVALUATION RESULTS ==========")

    for model_name, column in models.items():

        accuracy = accuracy_score(
            results["human_label"],
            results[column]
        )

        macro_f1 = f1_score(
            results["human_label"],
            results[column],
            labels=LABELS,
            average="macro",
            zero_division=0
        )

        print(f"\n{model_name}")
        print("Agreement:", round(accuracy * 100, 2), "%")
        print("Accuracy:", round(accuracy, 4))
        print("Macro F1:", round(macro_f1, 4))

        matrix = confusion_matrix(
            results["human_label"],
            results[column],
            labels=LABELS
        )

        display = ConfusionMatrixDisplay(
            confusion_matrix=matrix,
            display_labels=LABELS
        )

        display.plot(
            xticks_rotation=45,
            cmap="Blues"
        )

        plt.title(f"{model_name} Confusion Matrix")
        plt.tight_layout()

        filename = (
            "outputs/groq_confusion_matrix.png"
            if model_name == "Groq"
            else "outputs/distilbert_confusion_matrix.png"
        )

        plt.savefig(filename)
        plt.close()

        print("Confusion matrix saved:", filename)

    results.to_csv(
        OUTPUT_PATH,
        index=False
    )

    print("\nEvaluation completed.")
    print("Evaluated tickets:", len(results))
    print("Results saved:", OUTPUT_PATH)


if __name__ == "__main__":
    main()
