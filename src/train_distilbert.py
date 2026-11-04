
import json
import pandas as pd
import torch

from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.optim import AdamW
from sklearn.model_selection import train_test_split


DATA_PATH = "data/processed/labelled_tickets.csv"
MODEL_PATH = "models/distilbert"
BASE_MODEL = "distilbert-base-uncased"

LABELS = [
    "api",
    "billing",
    "cancellation",
    "complaint",
    "technical",
    "upgrade"
]

LABEL2ID = {label: i for i, label in enumerate(LABELS)}
ID2LABEL = {i: label for label, i in LABEL2ID.items()}

EPOCHS = 1
BATCH_SIZE = 4
MAX_LENGTH = 128
LEARNING_RATE = 2e-5


class TicketDataset(Dataset):

    def __init__(self, texts, labels, tokenizer):

        self.encodings = tokenizer(
            texts,
            truncation=True,
            padding="max_length",
            max_length=MAX_LENGTH,
            return_tensors="pt"
        )

        self.labels = torch.tensor(
            labels,
            dtype=torch.long
        )

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):

        return {
            "input_ids": self.encodings["input_ids"][index],
            "attention_mask": self.encodings["attention_mask"][index],
            "labels": self.labels[index]
        }


def main():

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Device:", device)

    df = pd.read_csv(DATA_PATH)

    df = df[
        (df["labelling_status"] == "success")
        & (df["predicted_label"].isin(LABELS))
    ]

    df = df.dropna(subset=["ticket_text"])
    df = df.drop_duplicates(subset=["ticket_text"])

    df = df.head(30).copy()

    if len(df) < 10:
        raise ValueError("At least 10 labelled tickets are required.")

    df["label_id"] = df["predicted_label"].map(LABEL2ID)

    counts = df["label_id"].value_counts()

    if len(counts) < 2 or counts.min() < 2:
        raise ValueError(
            "Need at least two categories with two examples each. "
            "Add more Groq-labelled tickets."
        )

    print("Training samples:", len(df))
    print("\nCategory distribution:")
    print(df["predicted_label"].value_counts())

    train_texts, val_texts, train_labels, val_labels = train_test_split(
        df["ticket_text"].tolist(),
        df["label_id"].tolist(),
        test_size=0.2,
        random_state=42,
        stratify=df["label_id"]
    )

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

    train_data = TicketDataset(
        train_texts,
        train_labels,
        tokenizer
    )

    val_data = TicketDataset(
        val_texts,
        val_labels,
        tokenizer
    )

    train_loader = DataLoader(
        train_data,
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    val_loader = DataLoader(
        val_data,
        batch_size=BATCH_SIZE
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL,
        num_labels=len(LABELS),
        id2label=ID2LABEL,
        label2id=LABEL2ID
    )

    model.to(device)

    optimizer = AdamW(
        model.parameters(),
        lr=LEARNING_RATE
    )

    model.train()

    for epoch in range(EPOCHS):

        total_loss = 0

        for batch in train_loader:

            batch = {
                key: value.to(device)
                for key, value in batch.items()
            }

            optimizer.zero_grad()

            outputs = model(**batch)

            loss = outputs.loss

            loss.backward()

            optimizer.step()

            total_loss += loss.item()

        print(
            f"Epoch {epoch + 1}/{EPOCHS}, "
            f"Loss: {total_loss / len(train_loader):.4f}"
        )

    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():

        for batch in val_loader:

            labels = batch["labels"].to(device)

            inputs = {
                key: value.to(device)
                for key, value in batch.items()
                if key != "labels"
            }

            outputs = model(**inputs)

            predictions = torch.argmax(
                outputs.logits,
                dim=1
            )

            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    accuracy = correct / total

    print("\nValidation accuracy:", round(accuracy * 100, 2), "%")

    Path(MODEL_PATH).mkdir(
        parents=True,
        exist_ok=True
    )

    model.save_pretrained(MODEL_PATH)
    tokenizer.save_pretrained(MODEL_PATH)

    with open(
        Path(MODEL_PATH) / "labels.json",
        "w"
    ) as file:
        json.dump(LABEL2ID, file, indent=4)

    print("\nDistilBERT training completed!")
    print("Model saved at:", MODEL_PATH)


if __name__ == "__main__":
    main()
