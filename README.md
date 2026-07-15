# 💊 Adverse Drug Reaction Detector

A web app that reads informal, social-media-style text and flags mentions of drugs and their adverse reactions, using a BERT model fine-tuned on real patient forum posts.

**Live demo:** https://adr-detector-dkgnpfbkkxa43pkicwypta.streamlit.app

Try it with something like: *"started metformin last week and I can't stop feeling nauseous"*

## What it does

Paste a post, and the app extracts:
- **Drug** mentions (e.g. "metformin")
- **ADR** (adverse drug reaction) mentions (e.g. "nauseous")

and flags whether the post likely describes an adverse drug reaction.

## How it works

1. **Data**: [CADEC](https://data.csiro.au/collection/csiro:10948) — 1,250 real patient forum posts, hand-annotated with drug/reaction/symptom spans.
2. **Model**: [`Bio_ClinicalBERT`](https://huggingface.co/emilyalsentzer/Bio_ClinicalBERT) fine-tuned as a token-classification (NER) model on CADEC's `Drug` and `ADR` labels. Fine-tuning notebook: [`fine_tune_cadec.ipynb`](fine_tune_cadec.ipynb) (run on Google Colab).
3. **App**: a single-file [Streamlit](https://streamlit.io) app (`app.py`) that loads the fine-tuned model from the [Hugging Face Hub](https://huggingface.co/rainwym/adr-ner-cadec) and runs inference on user input.

Starting from a general-purpose biomedical NER model (trained on formal clinical notes) initially misclassified casual phrases like "can't stop feeling" as symptoms. Fine-tuning on CADEC's informal, patient-written text fixed this — a concrete example of why domain-specific fine-tuning matters over an off-the-shelf model.

## Tech stack

| Layer | Tool |
|---|---|
| Data | [CADEC](https://data.csiro.au/collection/csiro:10948) |
| Model | [Bio_ClinicalBERT](https://huggingface.co/emilyalsentzer/Bio_ClinicalBERT), fine-tuned with 🤗 Transformers |
| Training | Google Colab (free GPU) |
| Model hosting | [Hugging Face Hub](https://huggingface.co/rainwym/adr-ner-cadec) |
| App | Streamlit |
| Hosting | [Streamlit Community Cloud](https://streamlit.io/cloud) |

## Run it locally

```bash
git clone https://github.com/rainwym/adr-detector.git
cd adr-detector
python -m venv venv
venv\Scripts\Activate.ps1   # Windows PowerShell
pip install -r requirements.txt
streamlit run app.py
```

## Project structure

```
app.py                  # Streamlit app (frontend + inference)
explore_data.py         # Parses CADEC's raw text + annotations into a DataFrame
fine_tune_cadec.ipynb   # Fine-tuning notebook (Colab)
requirements.txt
```

`data/` (the raw CADEC dataset) is not included in this repo — download it from [data.csiro.au](https://data.csiro.au/collection/csiro:10948) if you want to re-run `explore_data.py` or the fine-tuning notebook yourself.
