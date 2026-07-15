import streamlit as st
import pandas as pd
from transformers import pipeline
from pypdf import PdfReader
import docx

# --- Model logic ---
# Uses a BERT model fine-tuned on the CADEC dataset (real, informal forum
# posts) to find drug mentions ("Drug") and adverse reaction mentions ("ADR").


def read_uploaded_file(uploaded_file):
    """Extract raw text from an uploaded .pdf, .docx, or .txt file."""
    name = uploaded_file.name.lower()

    if name.endswith(".pdf"):
        reader = PdfReader(uploaded_file)
        return "\n".join(page.extract_text() or "" for page in reader.pages)

    if name.endswith(".docx"):
        document = docx.Document(uploaded_file)
        return "\n".join(paragraph.text for paragraph in document.paragraphs)

    return uploaded_file.read().decode("utf-8")


@st.cache_resource
def load_ner_pipeline():
    return pipeline(
        "token-classification",
        model="rainwym/adr-ner-cadec",
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


def chunk_text(text: str, max_chars: int = 800):
    """Split long text into pieces that safely fit under the model's token
    limit, breaking on whole words only."""
    words = text.split()
    chunks = []
    current_words = []
    current_len = 0

    for word in words:
        if current_len + len(word) + 1 > max_chars and current_words:
            chunks.append(" ".join(current_words))
            current_words = []
            current_len = 0
        current_words.append(word)
        current_len += len(word) + 1

    if current_words:
        chunks.append(" ".join(current_words))

    return chunks


def extract_entities(text: str):
    """Find drug mentions and reaction mentions in the text using the NER model."""
    ner = load_ner_pipeline()
    raw_tokens = ner(text)
    merged = merge_tokens(raw_tokens, text)
    drugs_found = [word for label, word in merged if label == "Drug"]
    reactions_found = [word for label, word in merged if label == "ADR"]
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
    "Paste a social-media-style post below. This uses a BERT model "
    "fine-tuned on real forum posts (the CADEC dataset) to find drug "
    "and adverse reaction mentions."
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
