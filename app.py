import re
import streamlit as st
import pandas as pd
from transformers import pipeline
from pypdf import PdfReader
import docx

# --- Model logic ---
# Uses BERT model fine-tuned on the CADEC dataset to find drug mentions ("Drug") and adverse reaction mentions ("ADR").


def clean_text(text: str) -> str:
    """Undo common PDF text-extraction artifacts: rejoin words that were
    split by a line-wrap hyphen (e.g. 'dys- tonia' -> 'dystonia'), and
    collapse repeated whitespace down to single spaces."""
    text = re.sub(r"(?<=[a-z])-\s+(?=[a-z])", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def read_uploaded_file(uploaded_file):
    """Extract raw text from an uploaded .pdf, .docx, or .txt file."""
    name = uploaded_file.name.lower()

    if name.endswith(".pdf"):
        reader = PdfReader(uploaded_file)
        raw_text = "\n".join(page.extract_text() or "" for page in reader.pages)
    elif name.endswith(".docx"):
        document = docx.Document(uploaded_file)
        raw_text = "\n".join(paragraph.text for paragraph in document.paragraphs)
    else:
        raw_text = uploaded_file.read().decode("utf-8")

    return clean_text(raw_text)


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
    """Find drug mentions and reaction mentions in the text using the NER model.
    Long text is processed in chunks so the whole document gets analyzed,
    not just the part that fits in one model call."""
    ner = load_ner_pipeline()
    drugs_found, reactions_found = [], []

    for chunk in chunk_text(text):
        raw_tokens = ner(chunk)
        merged = merge_tokens(raw_tokens, chunk)
        drugs_found += [word for label, word in merged if label == "Drug"]
        reactions_found += [word for label, word in merged if label == "ADR"]

    return drugs_found, reactions_found


def detect_adr(text: str):
    """Decide whether this text describes an adverse drug reaction."""
    drugs, reactions = extract_entities(text)
    is_adr = len(drugs) > 0 and len(reactions) > 0
    return is_adr, drugs, reactions


# --- Frontend ---

st.set_page_config(page_title="ADR Detector", page_icon="💊", layout="wide")

st.title("💊 Adverse Drug Reaction Detector")
st.write(
    "Paste a social-media-style post below. This uses a BERT model "
    "fine-tuned on real forum posts (the CADEC dataset) to find drug "
    "and adverse reaction mentions."
)

input_col, results_col = st.columns(2)

with input_col:
    st.subheader("Input")

    # Reserve the text box's spot now, so it renders above the uploader
    # below -- we fill it in further down, once we know whether a file
    # was uploaded.
    post_text_slot = st.empty()

    uploaded_file = st.file_uploader(
        "Or upload a document (.txt, .pdf, .docx)", type=["txt", "pdf", "docx"]
    )
    extracted_text = read_uploaded_file(uploaded_file) if uploaded_file is not None else ""

    user_text = post_text_slot.text_area(
        "Post text",
        value=extracted_text,
        placeholder="e.g. started metformin last week and I can't stop feeling nauseous",
        height=120,
    )

    analyze_clicked = st.button("Analyze")

with results_col:
    st.subheader("Results")

    if analyze_clicked:
        if not user_text.strip():
            st.warning("Please enter some text first.")
        else:
            with st.spinner("Loading model and analyzing..."):
                is_adr, drugs, reactions = detect_adr(user_text)

            if is_adr:
                st.success("Possible adverse drug reaction detected!")
            else:
                st.info("No adverse drug reaction detected.")

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
    else:
        st.write("Results will appear here once you click Analyze.")
