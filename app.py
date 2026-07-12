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
        aggregation_strategy="none",
    )


def merge_tokens(raw_tokens, text):
    """Stitch the model's raw sub-word tokens (e.g. 'met' + '##formin')
    back into whole words, using each token's position in the original
    text rather than the raw token strings themselves."""
    entities = []
    current = None
    for tok in raw_tokens:
        tag = tok["entity"]
        is_subword = tok["word"].startswith("##")

        if tag == "O" and not is_subword:
            current = None
            continue

        label = tag.split("-", 1)[-1]
        continues_prev = is_subword or (
            current is not None and not tag.startswith("B-") and current["label"] == label
        )

        if continues_prev and current is not None:
            current["end"] = tok["end"]
        else:
            if current is not None:
                entities.append(current)
            current = {"label": label, "start": tok["start"], "end": tok["end"]}

    if current is not None:
        entities.append(current)

    return [(e["label"], text[e["start"]:e["end"]]) for e in entities]


def extract_entities(text: str):
    """Find drug mentions and reaction mentions in the text using the NER model."""
    ner = load_ner_pipeline()
    raw_tokens = ner(text)
    merged = merge_tokens(raw_tokens, text)
    drugs_found = [word for label, word in merged if label == "Medication"]
    reactions_found = [word for label, word in merged if label == "Sign_symptom"]
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
