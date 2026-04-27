"""
preprocess.py
-------------
Stage 1: Fetch Wikidata labels for Subject/Relation/Object IDs and
         detect subject entity spans in questions using Flair GloVe embeddings.

Input  (TSV, no header): Subject_ID  Relation_ID  Object_ID  Question
Output (TSV, no header): Subject  Subject_ID  Relation  Relation_ID
                         Object   Object_ID   Question  Start_Subject  End_Subject
"""

import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
import pandas as pd
from tqdm import tqdm
from flair.data import Sentence
from flair.embeddings import WordEmbeddings, DocumentPoolEmbeddings
from scipy.spatial.distance import cosine


# ── Config ───────────────────────────────────────────────────────────────────

MAX_WORKERS       = 4
WIKIDATA_API_URL  = "https://www.wikidata.org/w/api.php"
WIKIDATA_BATCH_SIZE = 50
MAX_RETRIES       = 6
BACKOFF_FACTOR    = 0.5
BATCH_SLEEP       = 0.10
USER_AGENT        = "GeoQA-Labeler/1.0 (local-run) contact: your_email@example.com"

# ── Embeddings (loaded once) ─────────────────────────────────────────────────

glove_embedding    = WordEmbeddings("glove")
document_embeddings = DocumentPoolEmbeddings([glove_embedding])

# ── Wikidata session ─────────────────────────────────────────────────────────

SESSION = requests.Session()
SESSION.headers.update({"User-Agent": USER_AGENT})


def normalize_entity_id(entity_id: str) -> str | None:
    """Convert R-prefixed relation IDs to P-prefixed; pass Q/P IDs through."""
    if entity_id is None:
        return None
    s = str(entity_id).strip()
    if not s:
        return None
    if s.startswith("R") and s[1:].isdigit():
        return f"P{s[1:]}"
    return s


def fetch_labels_batch(norm_ids: list[str]) -> dict[str, str]:
    """Fetch Wikidata English labels for a batch of normalized entity IDs."""
    out = {nid: "N/A" for nid in norm_ids}
    if not norm_ids:
        return out

    params = {
        "action": "wbgetentities",
        "ids": "|".join(norm_ids),
        "format": "json",
        "languages": "en",
        "props": "labels",
    }
    for attempt in range(MAX_RETRIES):
        try:
            r = SESSION.get(WIKIDATA_API_URL, params=params, timeout=30)
            if r.status_code in (429, 500, 502, 503, 504):
                raise requests.RequestException(f"HTTP {r.status_code}")
            r.raise_for_status()
            entities = r.json().get("entities", {})
            for nid in norm_ids:
                ent = entities.get(nid, {})
                labels = ent.get("labels", {}) if isinstance(ent, dict) else {}
                out[nid] = labels.get("en", {}).get("value", "N/A")
            return out
        except requests.exceptions.RequestException:
            time.sleep(BACKOFF_FACTOR * (2 ** attempt))
    return out


def fetch_all_labels_parallel(all_raw_ids) -> dict[str, str]:
    """
    Fetch labels for all raw IDs in parallel batches.
    Returns mapping: raw_id_str -> label.
    """
    raw_to_norm: dict[str, str] = {}
    for rid in all_raw_ids:
        if rid is None:
            continue
        rid_str = str(rid).strip()
        if not rid_str:
            continue
        norm = normalize_entity_id(rid_str)
        if norm:
            raw_to_norm[rid_str] = norm

    unique_norm = sorted(set(raw_to_norm.values()))
    if not unique_norm:
        return {}

    batches = [
        unique_norm[i : i + WIKIDATA_BATCH_SIZE]
        for i in range(0, len(unique_norm), WIKIDATA_BATCH_SIZE)
    ]
    norm_to_label: dict[str, str] = {}

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futures = [ex.submit(fetch_labels_batch, b) for b in batches]
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Wikidata labels"):
            try:
                norm_to_label.update(fut.result())
            except Exception as e:
                print(f"Label batch failed: {e}")
            time.sleep(BATCH_SLEEP)

    return {raw: norm_to_label.get(norm, "N/A") for raw, norm in raw_to_norm.items()}


def preprocess_text(text) -> str:
    if text is None or (isinstance(text, float) and pd.isna(text)):
        return ""
    return str(text).lower().strip()


def find_subject_span_flair(subject: str, question: str, max_span_len: int = 12) -> tuple[int, int]:
    """
    Find the token span in `question` that best matches `subject`
    using cosine similarity of GloVe document embeddings.

    Returns (start_token, end_token) or (-1, -1) on failure.
    """
    question = preprocess_text(question)
    subject  = preprocess_text(subject)

    if not question or not subject or subject == "n/a":
        return -1, -1

    q_sent = Sentence(question)
    s_sent = Sentence(subject)

    try:
        document_embeddings.embed(q_sent)
        document_embeddings.embed(s_sent)
    except Exception as e:
        print(f"Embedding error: {e}")
        return -1, -1

    subj_emb = s_sent.embedding.cpu()
    tokens   = [t.text for t in q_sent]
    if not tokens:
        return -1, -1

    best_span: tuple[int, int] | None = None
    best_dist = float("inf")

    n = len(tokens)
    for start in range(n):
        for end in range(start + 1, min(n, start + max_span_len) + 1):
            span_sent = Sentence(" ".join(tokens[start:end]))
            try:
                document_embeddings.embed(span_sent)
            except Exception:
                continue
            dist = cosine(subj_emb, span_sent.embedding.cpu())
            if dist < best_dist:
                best_dist = dist
                best_span = (start, end)

    if best_span and best_dist < 0.2:
        return best_span
    return -1, -1


def process_file(input_path: str | Path, output_path: str | Path) -> None:
    """
    Process one data split (test/train/valid).
    Reads input TSV, fetches labels, detects spans, writes output TSV.
    """
    input_path  = Path(input_path)
    output_path = Path(output_path)

    if not input_path.exists():
        raise FileNotFoundError(f"Missing: {input_path}")

    df = pd.read_csv(
        input_path,
        sep="\t",
        names=["Subject_ID", "Relation_ID", "Object_ID", "Question"],
        dtype=str,
        keep_default_na=False,
    )

    for col in ["Subject_ID", "Relation_ID", "Object_ID"]:
        df[col] = df[col].apply(
            lambda x: str(x).strip() if x and str(x).strip() else None
        )

    all_ids = pd.unique(df[["Subject_ID", "Relation_ID", "Object_ID"]].values.ravel("K"))
    raw_to_label = fetch_all_labels_parallel(all_ids)

    for col, label_col in [
        ("Subject_ID", "Subject_Label"),
        ("Relation_ID", "Relation_Label"),
        ("Object_ID", "Object_Label"),
    ]:
        df[label_col] = df[col].apply(
            lambda x: raw_to_label.get(str(x), "N/A") if x else "N/A"
        )

    starts, ends = [], []
    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Spans ({input_path.name})"):
        s, e = find_subject_span_flair(row["Subject_Label"], row["Question"])
        starts.append(s)
        ends.append(e)

    out = df[
        [
            "Subject_Label", "Subject_ID",
            "Relation_Label", "Relation_ID",
            "Object_Label", "Object_ID",
            "Question",
        ]
    ].copy()
    out["Start_Subject"] = starts
    out["End_Subject"]   = ends
    out.columns = [
        "Subject", "Subject_ID",
        "Relation", "Relation_ID",
        "Object", "Object_ID",
        "Question", "Start_Subject", "End_Subject",
    ]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_path, sep="\t", index=False, header=False)
    print(f"Wrote: {output_path}")


def main(base_dir: str = ".") -> None:
    """Run preprocessing for all three splits."""
    base = Path(base_dir)

    splits = {
        "test":  ("not_ans/test/annotated_wd_data_test.txt",  "not_ans/test/flair_test_model_4.txt"),
        "train": ("not_ans/train/annotated_wd_data_train.txt", "not_ans/train/flair_train_model_4.txt"),
        "valid": ("not_ans/valid/annotated_wd_data_valid.txt", "not_ans/valid/flair_valid_model_4.txt"),
    }

    for split, (inp, out) in splits.items():
        print(f"\n--- {split.upper()} ---")
        process_file(base / inp, base / out)

    print("\nAll splits processed.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="NeuralGeoQA preprocessing")
    parser.add_argument("--base_dir", default=".", help="Project root with not_ans/ subdirs")
    args = parser.parse_args()
    main(args.base_dir)
