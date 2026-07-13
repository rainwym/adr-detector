import pandas as pd
from pathlib import Path

TEXT_DIR = Path("data/cadec/text")
ANN_DIR = Path("data/cadec/original")


def parse_ann_file(ann_path: Path):
    """Parse a single brat-format .ann file into a list of entity dicts."""
    entities = []
    for line in ann_path.read_text(encoding="utf-8").splitlines():
        if not line.startswith("T"):
            continue  # skip AnnotatorNotes ("#...") and relation lines
        parts = line.split("\t")
        if len(parts) < 3:
            continue
        entity_id, label_span, text = parts[0], parts[1], parts[2]
        label, *span_tokens = label_span.split(" ")
        span_str = " ".join(span_tokens)

        # Some entities are "discontinuous" (e.g. "9 19;25 30" when the
        # annotated phrase isn't one unbroken run of characters). We keep
        # it simple and just take the outer start/end of the whole span.
        spans = [s.split() for s in span_str.split(";")]
        start = int(spans[0][0])
        end = int(spans[-1][1])

        entities.append(
            {"entity_id": entity_id, "label": label, "start": start, "end": end, "text": text}
        )
    return entities


def load_cadec():
    rows = []
    for text_path in sorted(TEXT_DIR.glob("*.txt")):
        ann_path = ANN_DIR / (text_path.stem + ".ann")
        if not ann_path.exists():
            continue
        post_text = text_path.read_text(encoding="utf-8")
        for entity in parse_ann_file(ann_path):
            rows.append(
                {
                    "post_id": text_path.stem,
                    "label": entity["label"],
                    "entity_text": entity["text"],
                    "start": entity["start"],
                    "end": entity["end"],
                    "post_text": post_text,
                }
            )
    return pd.DataFrame(rows)


if __name__ == "__main__":
    df = load_cadec()
    print(f"Loaded {len(df)} entity annotations from {df['post_id'].nunique()} posts\n")
    print("Entity type counts:")
    print(df["label"].value_counts())
    print("\nSample rows:")
    print(df[["post_id", "label", "entity_text"]].head(10))
