
import os
import streamlit as st

from dotenv import load_dotenv
from groq import Groq

from src.few_shot_labeler import classify_ticket, MODEL_NAME
from src.predict import predict_ticket


load_dotenv()

st.set_page_config(
    page_title="Few-Shot Data Labeler",
    page_icon="",
    layout="wide"
)


@st.cache_resource
def load_groq_client():

    api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        try:
            api_key = st.secrets["GROQ_API_KEY"]
        except Exception:
            raise ValueError("GROQ_API_KEY is missing.")

    return Groq(api_key=api_key)


st.title("Few-Shot Data Labeler")

st.write(
    "Compare LLM-based few-shot classification "
    "with a fine-tuned DistilBERT model."
)

st.divider()

ticket = st.text_area(
    "Enter customer support ticket",
    placeholder="Example: My payment was deducted twice.",
    height=150
)

if st.button("Classify Ticket", type="primary"):

    if not ticket.strip():

        st.warning("Please enter a support ticket.")

    else:

        try:

            with st.spinner("Classifying ticket..."):

                client = load_groq_client()

                groq_result = classify_ticket(
                    client,
                    ticket.strip()
                )

                distilbert_result = predict_ticket(
                    ticket.strip()
                )

            st.success("Classification completed!")

            st.subheader("Classification Results")

            col1, col2 = st.columns(2)

            with col1:

                st.markdown("### Groq Few-Shot")

                st.metric(
                    "Predicted Category",
                    groq_result["predicted_label"].title()
                )

                st.metric(
                    "Estimated Confidence",
                    f'{groq_result["confidence_estimate"]}%'
                )

                st.caption("Model: " + MODEL_NAME)

                st.write("Reason:")
                st.write(groq_result["label_reason"])

            with col2:

                st.markdown("### DistilBERT")

                st.metric(
                    "Predicted Category",
                    distilbert_result["predicted_label"].title()
                )

                st.metric(
                    "Model Confidence",
                    f'{distilbert_result["confidence"]}%'
                )

                st.caption("Model: distilbert-base-uncased (fine-tuned)")

            st.divider()

            if (
                groq_result["predicted_label"]
                == distilbert_result["predicted_label"]
            ):
                st.success("Both models predicted the same category.")
            else:
                st.warning("The two models predicted different categories.")

        except Exception as error:

            st.error(f"Classification failed: {error}")


st.divider()

st.caption(
    "Project: P9 - Few-Shot Data Labeler | "
    "Groq API + DistilBERT + Streamlit"
)
