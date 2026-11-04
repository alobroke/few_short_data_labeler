
import torch

from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_PATH = "models/distilbert"

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_PATH
)

model.to(device)
model.eval()


def predict_ticket(ticket):

    inputs = tokenizer(
        ticket,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128
    )

    inputs = {
        key: value.to(device)
        for key, value in inputs.items()
    }

    with torch.no_grad():

        outputs = model(**inputs)

        probabilities = torch.softmax(
            outputs.logits,
            dim=1
        )

    confidence, prediction = probabilities.max(dim=1)

    label = model.config.id2label[prediction.item()]

    return {
        "ticket": ticket,
        "predicted_label": label,
        "confidence": round(confidence.item() * 100, 2)
    }


if __name__ == "__main__":

    test_tickets = [
        "My payment was deducted twice from my account.",
        "I cannot access my API endpoint.",
        "I want to cancel my subscription."
    ]

    for ticket in test_tickets:

        result = predict_ticket(ticket)

        print("\nTicket:", result["ticket"])
        print("Prediction:", result["predicted_label"])
        print("Confidence:", result["confidence"], "%")
