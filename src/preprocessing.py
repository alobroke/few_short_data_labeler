
import pandas as pd
import re
from pathlib import Path

# File paths
INPUT_PATH = Path("data/raw/unlabelled_tickets.csv")
OUTPUT_PATH = Path("data/processed/cleaned_tickets.csv")


def clean_text(text):
    """
    Clean an individual support ticket.
    """

    if pd.isna(text):
        return ""

    text = str(text)

    # Remove HTML tags
    text = re.sub(r"<[^>]+>", " ", text)

    # Remove URLs
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)

    # Normalize whitespace
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def preprocess_data(df):
    """
    Preprocess the complete ticket dataset.
    """

    print("Starting preprocessing...")

    # Check required columns
    if "ticket_text" not in df.columns:
        raise ValueError("Missing ticket_text column")

    # Remove missing tickets
    df = df.dropna(subset=["ticket_text"]).copy()

    # Clean ticket text
    df["ticket_text"] = df["ticket_text"].apply(clean_text)

    # Remove empty tickets
    df = df[df["ticket_text"].str.len() > 0]

    # Remove duplicate tickets
    df = df.drop_duplicates(
        subset=["ticket_text"]
    )

    # Remove extremely short tickets
    df = df[df["ticket_text"].str.len() >= 5]

    # Calculate text length
    df["text_length"] = df["ticket_text"].str.len()

    # Reset index
    df = df.reset_index(drop=True)

    # Generate IDs if missing
    if "ticket_id" not in df.columns:
        df.insert(
            0,
            "ticket_id",
            [f"T{i:05d}" for i in range(1, len(df) + 1)]
        )

    print("Preprocessing completed!")

    return df


def main():

    print("Loading dataset...")

    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {INPUT_PATH}"
        )

    df = pd.read_csv(INPUT_PATH)

    original_count = len(df)

    print("Original tickets:", original_count)

    # Preprocess
    cleaned_df = preprocess_data(df)

    # Create output directory
    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    # Save processed dataset
    cleaned_df.to_csv(
        OUTPUT_PATH,
        index=False
    )

    # Display statistics
    print("\n========== RESULTS ==========")

    print("Original tickets:", original_count)
    print("Cleaned tickets:", len(cleaned_df))
    print("Removed tickets:", original_count - len(cleaned_df))

    print("\nText length statistics:")
    print(cleaned_df["text_length"].describe())

    print("\nSample cleaned tickets:")
    print(
        cleaned_df[
            ["ticket_id", "ticket_text", "text_length"]
        ].head()
    )

    print("\nSaved:", OUTPUT_PATH)


if __name__ == "__main__":
    main()
