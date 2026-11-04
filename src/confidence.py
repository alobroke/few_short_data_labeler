
import json

LABELS = [
    "api",
    "billing",
    "cancellation",
    "complaint",
    "technical",
    "upgrade"
]

EXAMPLES = """
Ticket: Our API endpoint returns a 500 error.
Label: api

Ticket: I was charged twice for my monthly invoice.
Label: billing

Ticket: Please cancel my account subscription.
Label: cancellation

Ticket: Your customer service has been terrible.
Label: complaint

Ticket: The application crashes whenever I open it.
Label: technical
"""


def create_prompt(ticket):

    return f"""
You are a customer support ticket classifier.

Choose exactly one category:
{", ".join(LABELS)}

Learn from these five examples:

{EXAMPLES}

Classify this ticket:

{ticket}

Return valid JSON with these fields:
{{
    "label": "one allowed category",
    "reason": "short explanation",
    "confidence": 90
}}

Confidence must be an integer from 0 to 100.
It is your estimated confidence, not token probability.
"""


def parse_prediction(response_text):

    result = json.loads(response_text)

    label = result["label"].strip().lower()
    confidence = int(result["confidence"])

    if label not in LABELS:
        raise ValueError(f"Invalid label: {label}")

    if not 0 <= confidence <= 100:
        raise ValueError("Confidence must be between 0 and 100")

    return {
        "predicted_label": label,
        "label_reason": result["reason"],
        "confidence_estimate": confidence
    }
