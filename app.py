import streamlit as st
import pandas as pd
from transformers import pipeline

# --- Model logic ---
# Uses a pretrained biomedical NER model from Hugging Face to find drug
# mentions ("Medication") and symptom/reaction mentions ("Sign_symptom").
# Later, this will be swapped to own model.


@st.cache_resource
def load_ner_pipeline():
    return pipeline(
        "token-classification",
        model="d4data/biomedical-ner-all",
        aggregation_strategy="simple",
    )


def extract_entities(text: str):
    """Find drug mentions and reaction mentions in the text using the NER model."""
    ner = load_ner_pipeline()
    results = ner(text)
    drugs_found = [r["word"] for r in results if r["entity_group"] == "Medication"]
    reactions_found = [r["word"] for r in results if r["entity_group"] == "Sign_symptom"]
    return drugs_found, reactions_found


def detect_adr(text: str):
    """Decide whether this text describes an adverse drug reaction."""
    drugs, reactions = extract_entities(text)
    is_adr = len(drugs) > 0 and len(reactions) > 0
    return is_adr, drugs, reactions


# --- Frontend (the webpage itself) ---

st.set_page_config(page_title="ADR Detector", page_icon="💊")

st.title("💊 Adverse Drug Reaction Detector")
st.write(
    "Paste a social-media-style post below. This uses a pretrained "
    "biomedical NER model to find drug and symptom mentions — later "
    "this will be replaced with a model fine-tuned on your own data."
)

user_text = st.text_area(
    "Post text",
    placeholder="e.g. started metformin last week and I can't stop feeling nauseous",
    height=120,
)

if st.button("Analyze"):
    if not user_text.strip():
        st.warning("Please enter some text first.")
    else:
        with st.spinner("Loading model and analyzing..."):
            is_adr, drugs, reactions = detect_adr(user_text)

        if is_adr:
            st.success("Possible adverse drug reaction detected!")
        else:
            st.info("No adverse drug reaction detected.")

        st.subheader("Extracted entities")
        results = pd.DataFrame(
            {
                "Type": ["Drug"] * len(drugs) + ["Reaction"] * len(reactions),
                "Text": drugs + reactions,
            }
        )

        if results.empty:
            st.write("Nothing recognized in this text yet.")
        else:
            st.table(results)
