
import os
import json
import pandas as pd

from pathlib import Path
from dotenv import load_dotenv
from groq import Groq


load_dotenv()

API_KEY = os.getenv("GROQ_API_KEY")
MODEL_NAME = "openai/gpt-oss-20b"

INPUT_PATH = "data/processed/labelled_tickets.csv"
OUTPUT_PATH = "outputs/augmented_examples.csv"

LABELS = [
    "api",
    "billing",
    "cancellation",
    "complaint",
    "technical",
    "upgrade"
]

NUM_EXAMPLES = 5


def generate_examples(client, category):

    prompt = f"""
Generate exactly {NUM_EXAMPLES} realistic customer support tickets
belonging to the category: {category}.

Categories:
{", ".join(LABELS)}

Return valid JSON in this format:

{{
    "examples": [
        {{
            "ticket_text": "Example customer support ticket"
        }}
    ]
}}

All examples must belong to the category {category}.
Make every example different.
Do not include labels or explanations.
"""

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0.7,
        max_completion_tokens=1024,
        response_format={"type": "json_object"}
    )

    result = json.loads(
        response.choices[0].message.content
    )

    examples = result["examples"]

    if len(examples) != NUM_EXAMPLES:
        raise ValueError("Incorrect number of generated examples.")

    return [
        item["ticket_text"].strip()
        for item in examples
        if item.get("ticket_text", "").strip()
    ]


def main():

    if not API_KEY:
        raise ValueError("GROQ_API_KEY missing in .env")

    df = pd.read_csv(INPUT_PATH)

    df = df[
        (df["labelling_status"] == "success")
        & (df["predicted_label"].isin(LABELS))
    ]

    if df.empty:
        raise ValueError("No labelled tickets found.")

    distribution = (
        df["predicted_label"]
        .value_counts()
        .reindex(LABELS, fill_value=0)
    )

    print("Current category distribution:")
    print(distribution)

    rare_category = distribution.idxmin()

    print("\nRare category:", rare_category)

    client = Groq(api_key=API_KEY)

    examples = generate_examples(
        client,
        rare_category
    )

    output = pd.DataFrame({
        "ticket_text": examples,
        "predicted_label": rare_category,
        "source": "synthetic"
    })

    Path("outputs").mkdir(
        parents=True,
        exist_ok=True
    )

    output.to_csv(
        OUTPUT_PATH,
        index=False
    )

    print("\nGenerated examples:")

    for index, example in enumerate(examples, 1):
        print(f"{index}. {example}")

    print("\nSaved:", OUTPUT_PATH)


if __name__ == "__main__":
    main()
