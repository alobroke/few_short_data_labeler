
import pandas as pd
from datasets import load_dataset
from pathlib import Path



DATASET_ID = "cngchis/Support-Ticket-Router-12K-Cleaned"

RAW_PATH = Path("data/raw/unlabelled_tickets.csv")
HUMAN_PATH = Path("data/human/human_labels.csv")

RANDOM_STATE = 42
HUMAN_SAMPLE_SIZE = 50




def download_data():

    print("Downloading support ticket dataset...")

    dataset = load_dataset(DATASET_ID)

    frames = []

    for split_name, split_data in dataset.items():

        df = split_data.to_pandas()

        print(f"Loaded {split_name}: {len(df)} records")

        frames.append(df)

    data = pd.concat(frames, ignore_index=True)

    print("\nOriginal columns:", data.columns.tolist())

    return data




def prepare_unlabelled_data(data):

    # Support different possible text column names
    possible_columns = [
        "ticket_text",
        "text",
        "ticket",
        "description",
        "body"
    ]

    text_column = next(
        (col for col in possible_columns if col in data.columns),
        None
    )

    if text_column is None:
        raise ValueError(
            "Ticket text column not found. "
            f"Available columns: {data.columns.tolist()}"
        )

   
    data = data[[text_column]].copy()

    data.rename(
        columns={text_column: "ticket_text"},
        inplace=True
    )

    
    data["ticket_text"] = (
        data["ticket_text"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    data = data[data["ticket_text"] != ""]

   
    data.drop_duplicates(
        subset=["ticket_text"],
        inplace=True
    )

    data.reset_index(drop=True, inplace=True)

   
    data.insert(
        0,
        "ticket_id",
        [f"T{i:05d}" for i in range(1, len(data) + 1)]
    )

    return data



def create_human_template(data):

    sample_size = min(HUMAN_SAMPLE_SIZE, len(data))

    human_data = data.sample(
        n=sample_size,
        random_state=RANDOM_STATE
    ).copy()

    human_data["human_label"] = ""
    human_data["reviewer_notes"] = ""

    return human_data


# -----------------------------------
# 5. Main execution
# -----------------------------------

if __name__ == "__main__":

    try:

        RAW_PATH.parent.mkdir(parents=True, exist_ok=True)
        HUMAN_PATH.parent.mkdir(parents=True, exist_ok=True)

        # Download
        original_data = download_data()

        # Prepare unlabelled tickets
        tickets = prepare_unlabelled_data(original_data)

        # Save unlabelled CSV
        tickets.to_csv(RAW_PATH, index=False)

        # Create human review template
        human_data = create_human_template(tickets)

        human_data.to_csv(HUMAN_PATH, index=False)

        # Display results
        print("\n========== SUCCESS ==========")

        print("Total tickets:", len(tickets))
        print("Human review tickets:", len(human_data))

        print("\nSaved files:")
        print(RAW_PATH)
        print(HUMAN_PATH)

        print("\nSample tickets:")
        print(tickets.head())

        print("\nDataset preparation completed!")

    except Exception as e:

        print("\nERROR:", str(e))
        raise
