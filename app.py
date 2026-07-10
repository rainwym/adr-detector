import streamlit as st
import pandas as pd

# --- Placeholder "model logic" ---
# This is a stand-in for your real trained NER/classification model.
# Later, you'll swap the inside of these two functions to call your
# fine-tuned BERT model instead. The rest of the app stays the same.

KNOWN_DRUGS = [
    "metformin", "glucophage", "ibuprofen", "advil", "lisinopril",
    "atorvastatin", "lipitor", "sertraline", "zoloft", "amoxicillin",
]

REACTION_KEYWORDS = [
    "nauseous", "nausea", "dizzy", "dizziness", "headache", "rash",
    "vomiting", "fatigue", "tired", "swelling", "pain", "cramps",
    "insomnia", "can't sleep", "diarrhea",
]


def extract_entities(text: str):
    """Find drug mentions and reaction mentions in the text (placeholder: simple keyword matching)."""
    text_lower = text.lower()
    drugs_found = [d for d in KNOWN_DRUGS if d in text_lower]
    reactions_found = [r for r in REACTION_KEYWORDS if r in text_lower]
    return drugs_found, reactions_found


def detect_adr(text: str):
    """Decide whether this text describes an adverse drug reaction (placeholder logic)."""
    drugs, reactions = extract_entities(text)
    is_adr = len(drugs) > 0 and len(reactions) > 0
    return is_adr, drugs, reactions


# --- Frontend (the webpage itself) ---

st.set_page_config(page_title="ADR Detector", page_icon="💊")

st.title("💊 Adverse Drug Reaction Detector")
st.write(
    "Paste a social-media-style post below. This v1 uses simple keyword "
    "matching as a placeholder — later this will be replaced with a "
    "fine-tuned NLP model."
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
