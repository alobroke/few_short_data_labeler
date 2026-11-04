
import os
import time
import pandas as pd

from pathlib import Path
from dotenv import load_dotenv
from groq import Groq

from src.confidence import create_prompt, parse_prediction

load_dotenv()

API_KEY = os.getenv("GROQ_API_KEY")
MODEL_NAME = "openai/gpt-oss-20b"

INPUT_PATH = Path("data/processed/cleaned_tickets.csv")
OUTPUT_PATH = Path("data/processed/labelled_tickets.csv")

REQUEST_INTERVAL = 3
MAX_RETRIES = 5

last_request_time = 0


def classify_ticket(client, ticket):

    global last_request_time

    for attempt in range(MAX_RETRIES):

        elapsed = time.monotonic() - last_request_time

        if elapsed < REQUEST_INTERVAL:
            time.sleep(REQUEST_INTERVAL - elapsed)

        try:

            last_request_time = time.monotonic()

            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {
                        "role": "user",
                        "content": create_prompt(ticket)
                    }
                ],
                temperature=0,
                max_completion_tokens=512,
                response_format={"type": "json_object"}
            )

            content = response.choices[0].message.content

            if not content:
                raise ValueError("Empty Groq response")

            return parse_prediction(content)

        except Exception as error:

            status = getattr(error, "status_code", None)

            if status not in (429, 500, 502, 503, 504):
                raise

            if attempt == MAX_RETRIES - 1:
                raise

            delay = min(5 * (2 ** attempt), 60)

            response_obj = getattr(error, "response", None)

            if response_obj is not None:
                retry_after = response_obj.headers.get("retry-after")

                if retry_after:
                    try:
                        delay = max(delay, float(retry_after))
                    except ValueError:
                        pass

            print(
                f"Groq error {status}. "
                f"Retrying in {delay} seconds..."
            )

            time.sleep(delay)

    raise RuntimeError("Maximum retries exceeded")


def label_dataset(limit=30):

    if not API_KEY:
        raise ValueError("GROQ_API_KEY missing in .env")

    if not INPUT_PATH.exists():
        raise FileNotFoundError(INPUT_PATH)

    client = Groq(api_key=API_KEY)

    df = pd.read_csv(INPUT_PATH)

    if OUTPUT_PATH.exists():

        existing = pd.read_csv(OUTPUT_PATH)

        records = {
            str(row["ticket_id"]): row.to_dict()
            for _, row in existing.iterrows()
        }

    else:
        records = {}

    completed = {
        ticket_id
        for ticket_id, record in records.items()
        if record.get("labelling_status") == "success"
    }

    pending = df[
        ~df["ticket_id"].astype(str).isin(completed)
    ]

    if limit is not None:
        pending = pending.head(limit)

    print("Total tickets:", len(df))
    print("Already labelled:", len(completed))
    print("Remaining:", len(pending))
    print("Provider: Groq")
    print("Model:", MODEL_NAME)

    for index, row in enumerate(pending.itertuples(index=False), 1):

        ticket_id = str(row.ticket_id)

        print(f"\n[{index}/{len(pending)}] {ticket_id}")

        try:

            prediction = classify_ticket(
                client,
                str(row.ticket_text)
            )

            records[ticket_id] = {
                "ticket_id": ticket_id,
                "ticket_text": row.ticket_text,
                **prediction,
                "model": MODEL_NAME,
                "provider": "Groq",
                "labelling_status": "success"
            }

            print("Label:", prediction["predicted_label"])
            print(
                "Confidence estimate:",
                prediction["confidence_estimate"],
                "%"
            )

        except Exception as error:

            records[ticket_id] = {
                "ticket_id": ticket_id,
                "ticket_text": row.ticket_text,
                "predicted_label": "",
                "label_reason": "",
                "confidence_estimate": None,
                "model": MODEL_NAME,
                "provider": "Groq",
                "labelling_status": "failed",
                "error": str(error)
            }

            print("Error:", error)

        OUTPUT_PATH.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        pd.DataFrame(records.values()).to_csv(
            OUTPUT_PATH,
            index=False
        )

    print("\nLabelling completed.")

    print(
        "Successful:",
        sum(
            r.get("labelling_status") == "success"
            for r in records.values()
        )
    )

    print("Saved:", OUTPUT_PATH)


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument("--limit", type=int, default=30)

    args = parser.parse_args()

    label_dataset(args.limit)
