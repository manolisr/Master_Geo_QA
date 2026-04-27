"""
train.py
--------
Two-phase training for NeuralGeoQA.

Phase 1 — Pretrain on Wikidata SimpleQuestions
  Model : bert-base-uncased
  Heads : Span (start/end token positions) + Relation (125 Wikidata properties)
  Data  : 19,481 preprocessed SimpleQuestions examples
  Output: saved encoder + span/relation heads → used as starting point for Phase 2

Phase 2 — Fine-tune QType head on GeoQuestions1089
  Model : GeoQABERT (same encoder loaded from Phase 1)
  Heads : Span + Relation → FROZEN. New QType head trained.
  Data  : 1,087 GeoQuestions1089 examples (geo_train.tsv with QType column)
  Output: best_model.pt + config.json → loaded by evaluate.py

Usage:
  python train.py phase1 --base_dir /data/geoqa
  python train.py phase2 --base_dir /data/geoqa --pretrained_dir /data/geoqa/phase1_output
"""

import os
import re
import gc
import json
import time
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from collections import Counter
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from transformers import (
    AutoTokenizer,
    AutoModel,
    AutoConfig,
    AutoModelForQuestionAnswering,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup,
)
from sklearn.metrics import (
    precision_score, recall_score, f1_score, classification_report,
)
from sklearn.model_selection import train_test_split

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


# =============================================================================
# SHARED: CLASS MERGING
# =============================================================================
# 12 fine-grained QTypes → 8 merged classes for training.
# Types with <10 examples are absorbed into their nearest parent.
# At inference, merged predictions are split back via keyword rules
# (see refine_qtype() in evaluate.py).

CLASS_MERGE_MAP = {
    "C_class_spatial_rel": "C_class",       # 4 → merge into C_class (33+4=37)
    "C_class_in":          "C_class",
    "B_directional":       "B_spatial",     # 2 → merge into B_spatial (156+2=158)
    "E_class_distance":    "E_class_near",  # 7 → merge into E_class_near (307+7=314)
    "G_comparative":       "G_superlative", # 3 → merge into G_superlative (179+3=182)
    # Unchanged
    "A_attribute":         "A_attribute",
    "B_boolean":           "B_boolean",
    "B_spatial":           "B_spatial",
    "E_class_near":        "E_class_near",
    "F_thematic_spatial":  "F_thematic_spatial",
    "G_count":             "G_count",
    "G_superlative":       "G_superlative",
}


def merge_qtype(qtype: str) -> str:
    return CLASS_MERGE_MAP.get(qtype, qtype)


# =============================================================================
# SHARED: GeoQABERT MODEL
# =============================================================================

class GeoQABERT(nn.Module):
    """
    bert-base-uncased encoder with three prediction heads:
      span_start / span_end  : subject entity span (token-level)
      rel_classifier         : Wikidata relation prediction
      qtype_classifier       : question type (8 merged classes)

    Phase 1 trains span + relation heads only.
    Phase 2 freezes those and trains qtype_classifier.
    """

    def __init__(
        self,
        model_name: str,
        num_relations: int,
        num_qtypes: int,
        dropout: float = 0.1,
        class_weights: torch.Tensor = None,
    ):
        super().__init__()
        self.config           = AutoConfig.from_pretrained(model_name)
        self.encoder          = AutoModel.from_pretrained(model_name)
        hidden                = self.config.hidden_size

        self.span_start       = nn.Linear(hidden, 1)
        self.span_end         = nn.Linear(hidden, 1)
        self.rel_dropout      = nn.Dropout(dropout)
        self.rel_classifier   = nn.Linear(hidden, num_relations)
        self.qtype_dropout    = nn.Dropout(dropout)
        self.qtype_classifier = nn.Linear(hidden, num_qtypes)

        self.qtype_loss_fn = (
            nn.CrossEntropyLoss(weight=class_weights)
            if class_weights is not None
            else nn.CrossEntropyLoss()
        )
        self.span_loss_fn = nn.CrossEntropyLoss()
        self.rel_loss_fn  = nn.CrossEntropyLoss()

    def forward(
        self,
        input_ids,
        attention_mask,
        start_positions=None,
        end_positions=None,
        relation_labels=None,
        qtype_labels=None,
    ):
        out     = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        seq_out = out.last_hidden_state
        cls_out = seq_out[:, 0, :]

        start_logits  = self.span_start(seq_out).squeeze(-1)
        end_logits    = self.span_end(seq_out).squeeze(-1)
        rel_logits    = self.rel_classifier(self.rel_dropout(cls_out))
        qtype_logits  = self.qtype_classifier(self.qtype_dropout(cls_out))

        loss = None
        if start_positions is not None and end_positions is not None and relation_labels is not None:
            # Phase 1 loss: span + relation
            loss = (
                self.span_loss_fn(start_logits, start_positions)
                + self.span_loss_fn(end_logits, end_positions)
                + self.rel_loss_fn(rel_logits, relation_labels)
            )
        elif qtype_labels is not None:
            # Phase 2 loss: qtype only
            loss = self.qtype_loss_fn(qtype_logits, qtype_labels)

        return {
            "loss":         loss,
            "start_logits": start_logits,
            "end_logits":   end_logits,
            "rel_logits":   rel_logits,
            "qtype_logits": qtype_logits,
        }


# =============================================================================
# PHASE 1: SIMPLEQUESTIONS DATASET + SPAN HELPERS
# =============================================================================

def whitespace_token_spans(text: str) -> list:
    return [(m.start(), m.end()) for m in re.finditer(r"\S+", text)]


def word_span_to_char_span(question, start_word, end_word, subject=None):
    if question is None:
        return None, None
    q     = str(question)
    spans = whitespace_token_spans(q)
    if isinstance(start_word, int) and isinstance(end_word, int):
        if 0 <= start_word < end_word <= len(spans):
            return spans[start_word][0], spans[end_word - 1][1]
    if subject is not None:
        s   = str(subject).strip()
        pos = q.lower().find(s.lower())
        if pos != -1:
            return pos, pos + len(s)
    return None, None


def char_span_to_token_span(offset_mapping, char_start, char_end):
    if char_start is None or char_end is None:
        return 0, 0
    start_tok = end_tok = None
    for i, (s, e) in enumerate(offset_mapping):
        if s == 0 and e == 0: continue
        if start_tok is None and s <= char_start < e: start_tok = i
        if s < char_end <= e: end_tok = i
    if start_tok is None:
        for i, (s, e) in enumerate(offset_mapping):
            if s == 0 and e == 0: continue
            if s >= char_start: start_tok = i; break
    if end_tok is None:
        for i in range(len(offset_mapping) - 1, -1, -1):
            s, e = offset_mapping[i]
            if s == 0 and e == 0: continue
            if e <= char_end: end_tok = i; break
    if start_tok is None or end_tok is None or start_tok > end_tok:
        return 0, 0
    return int(start_tok), int(end_tok)


class SimpleQuestionsDataset(Dataset):
    """
    Preprocessed SimpleQuestions TSV (no header):
    0 Subject | 1 Subject_ID | 2 Relation | 3 Relation_ID |
    4 Object  | 5 Object_ID  | 6 Question | 7 Start_Subject | 8 End_Subject
    """

    def __init__(self, data, tokenizer, relation_to_id, max_length=96):
        self.data           = data
        self.tokenizer      = tokenizer
        self.relation_to_id = relation_to_id
        self.max_length     = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        row           = self.data.iloc[index]
        subject_label = row[0]
        question      = row[6]
        start_word    = int(row[7])
        end_word      = int(row[8])
        rel_id        = row[3]
        relation      = self.relation_to_id[rel_id]

        inputs = self.tokenizer(
            str(question),
            add_special_tokens=True,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
            return_offsets_mapping=True,
        )

        input_ids      = inputs["input_ids"].squeeze(0)
        attention_mask = inputs["attention_mask"].squeeze(0)
        offset_mapping = inputs["offset_mapping"].squeeze(0).tolist()

        char_start, char_end = word_span_to_char_span(
            str(question), start_word, end_word, subject=str(subject_label)
        )
        token_start, token_end = char_span_to_token_span(
            offset_mapping, char_start, char_end
        )

        seq_len     = int(input_ids.shape[0])
        token_start = max(0, min(token_start, seq_len - 1))
        token_end   = max(0, min(token_end, seq_len - 1))
        if token_start > token_end:
            token_start = token_end = 0

        return {
            "input_ids":       input_ids,
            "attention_mask":  attention_mask,
            "start_positions": torch.tensor(token_start, dtype=torch.long),
            "end_positions":   torch.tensor(token_end,   dtype=torch.long),
            "relation_label":  torch.tensor(relation,    dtype=torch.long),
        }


# =============================================================================
# PHASE 2: GEOQUESTIONS DATASET
# =============================================================================

class QTypeDataset(Dataset):
    """
    GeoQuestions1089 TSV with QType column.
    Uses merged classes (QType_Merged) for training.
    """

    def __init__(self, df, tokenizer, qtype_to_id, max_length=96):
        self.df         = df.reset_index(drop=True)
        self.tokenizer  = tokenizer
        self.qtype_to_id = qtype_to_id
        self.max_length = max_length

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row      = self.df.iloc[idx]
        question = str(row["Question"])
        qtype    = str(row["QType_Merged"])

        enc = self.tokenizer(
            question,
            add_special_tokens=True,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        return {
            "input_ids":    enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "qtype_label":  torch.tensor(self.qtype_to_id[qtype], dtype=torch.long),
        }


# =============================================================================
# PHASE 1 TRAINING
# =============================================================================

def p1_train_epoch(model, loader, optimizer, scheduler, device,
                   scaler, use_amp, accumulation_steps=8):
    model.train()
    total = span_total = rel_total = 0.0
    optimizer.zero_grad(set_to_none=True)

    for step, batch in enumerate(tqdm(loader, desc="P1 Train")):
        ids    = batch["input_ids"].to(device, non_blocking=True)
        mask   = batch["attention_mask"].to(device, non_blocking=True)
        starts = batch["start_positions"].to(device, non_blocking=True)
        ends   = batch["end_positions"].to(device, non_blocking=True)
        rels   = batch["relation_label"].to(device, non_blocking=True)

        with torch.amp.autocast("cuda", enabled=use_amp):
            out  = model(input_ids=ids, attention_mask=mask,
                         start_positions=starts, end_positions=ends,
                         relation_labels=rels)
            loss = out["loss"] / accumulation_steps

        scaler.scale(loss).backward()

        if (step + 1) % accumulation_steps == 0:
            scaler.step(optimizer); scaler.update()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)

        total += float(loss.item()) * accumulation_steps

    if len(loader) % accumulation_steps != 0:
        scaler.step(optimizer); scaler.update()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)

    return total / len(loader)


@torch.no_grad()
def p1_eval_epoch(model, loader, device, use_amp, id_to_relation):
    model.eval()
    cls_preds, cls_true = [], []
    start_correct = end_correct = n = 0
    total = 0.0

    for batch in tqdm(loader, desc="P1 Eval"):
        ids    = batch["input_ids"].to(device, non_blocking=True)
        mask   = batch["attention_mask"].to(device, non_blocking=True)
        starts = batch["start_positions"].to(device, non_blocking=True)
        ends   = batch["end_positions"].to(device, non_blocking=True)
        rels   = batch["relation_label"].to(device, non_blocking=True)

        with torch.amp.autocast("cuda", enabled=use_amp):
            out = model(input_ids=ids, attention_mask=mask,
                        start_positions=starts, end_positions=ends,
                        relation_labels=rels)

        total += float(out["loss"].item())
        s_pred = torch.argmax(out["start_logits"], dim=1)
        e_pred = torch.argmax(out["end_logits"],   dim=1)
        c_pred = torch.argmax(out["rel_logits"],   dim=1)

        bs = s_pred.size(0); n += bs
        start_correct += int((s_pred == starts).sum())
        end_correct   += int((e_pred == ends).sum())
        cls_preds.extend(c_pred.cpu().tolist())
        cls_true.extend(rels.cpu().tolist())

    return {
        "loss":      total / len(loader),
        "start_acc": start_correct / max(1, n),
        "end_acc":   end_correct   / max(1, n),
        "span_acc":  sum(p == t and q == u for (p, q), (t, u)
                         in zip(zip(cls_preds, cls_preds), zip(cls_true, cls_true))) / max(1, n),
        "rel_acc":   sum(p == t for p, t in zip(cls_preds, cls_true)) / max(1, n),
        "rel_f1_w":  f1_score(cls_true, cls_preds, average="weighted", zero_division=0),
    }


def train_phase1(
    base_dir:           str   = "/content/drive/MyDrive/ANS/triple_head/",
    epochs:             int   = 3,
    batch_size:         int   = 16,
    accumulation_steps: int   = 1,
    learning_rate:      float = 2e-5,
    max_length:         int   = 96,
    model_name:         str   = "bert-base-uncased",
) -> None:
    """
    Phase 1: train span + relation heads on Wikidata SimpleQuestions.
    Saves encoder weights and config for Phase 2 to load.
    """
    gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()

    out_dir = os.path.join(base_dir, "phase1_output")
    os.makedirs(out_dir, exist_ok=True)

    train_file = os.path.join(base_dir, "train/flair_train_model_4.txt")
    valid_file = os.path.join(base_dir, "valid/flair_valid_model_4.txt")
    test_file  = os.path.join(base_dir, "test/flair_test_model_4.txt")

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

    all_data = pd.concat([
        pd.read_csv(f, sep="\t", header=None)
        for f in [train_file, valid_file, test_file]
    ])
    unique_relations = all_data[3].unique()
    relation_to_id   = {rel: idx for idx, rel in enumerate(unique_relations)}
    id_to_relation   = {idx: rel for rel, idx in relation_to_id.items()}
    num_relations    = len(relation_to_id)
    print(f"Relations: {num_relations}")

    train_data = pd.read_csv(train_file, sep="\t", header=None)
    valid_data = pd.read_csv(valid_file, sep="\t", header=None)
    test_data  = pd.read_csv(test_file,  sep="\t", header=None)

    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"

    # num_qtypes=1 placeholder — qtype head not used in Phase 1
    model = GeoQABERT(model_name=model_name, num_relations=num_relations, num_qtypes=1)
    model.to(device)

    def make_loader(data, shuffle):
        ds = SimpleQuestionsDataset(data, tokenizer, relation_to_id, max_length)
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                          pin_memory=True, num_workers=0)

    train_loader = make_loader(train_data, True)
    valid_loader = make_loader(valid_data, False)
    test_loader  = make_loader(test_data,  False)

    optimizer   = AdamW(model.parameters(), lr=learning_rate)
    total_steps = len(train_loader) * epochs
    scheduler   = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=int(0.06 * total_steps),
        num_training_steps=total_steps,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    best_rel_acc = 0.0
    t0 = time.time()

    for epoch in range(1, epochs + 1):
        print(f"\nEpoch {epoch}/{epochs}")
        tr_loss = p1_train_epoch(model, train_loader, optimizer, scheduler,
                                  device, scaler, use_amp, accumulation_steps)
        print(f"  Train loss={tr_loss:.4f}")

        vm = p1_eval_epoch(model, valid_loader, device, use_amp, id_to_relation)
        print(f"  Valid loss={vm['loss']:.4f}  "
              f"StartAcc={vm['start_acc']:.4f}  EndAcc={vm['end_acc']:.4f}  "
              f"RelAcc={vm['rel_acc']:.4f}  RelF1-w={vm['rel_f1_w']:.4f}")

        if vm["rel_acc"] > best_rel_acc:
            best_rel_acc = vm["rel_acc"]
            torch.save(model.state_dict(),
                       os.path.join(out_dir, "best_model.pt"))
            print(f"  New best RelAcc={best_rel_acc:.4f}")

        if device.type == "cuda": torch.cuda.empty_cache()

    print(f"\nPhase 1 complete in {time.time()-t0:.1f}s")

    # Load best and run test
    model.load_state_dict(torch.load(
        os.path.join(out_dir, "best_model.pt"),
        map_location=device, weights_only=True,
    ))
    tm = p1_eval_epoch(model, test_loader, device, use_amp, id_to_relation)
    print(f"Test  StartAcc={tm['start_acc']:.4f}  EndAcc={tm['end_acc']:.4f}  "
          f"RelAcc={tm['rel_acc']:.4f}  RelF1-w={tm['rel_f1_w']:.4f}")

    tokenizer.save_pretrained(os.path.join(out_dir, "tokenizer"))
    config = {
        "model_name":      model_name,
        "num_relations":   num_relations,
        "num_qtypes":      1,
        "max_length":      max_length,
        "relation_to_id":  relation_to_id,
        "id_to_relation":  {str(k): v for k, v in id_to_relation.items()},
        "best_rel_acc":    best_rel_acc,
        "test_rel_acc":    tm["rel_acc"],
        "test_span_acc":   tm["start_acc"],
    }
    with open(os.path.join(out_dir, "final_model", "config.json"), "w") as f:
        os.makedirs(os.path.join(out_dir, "final_model"), exist_ok=True)
        json.dump(config, f, indent=2)
    # copy best_model.pt to final_model/ so Phase 2 can find it
    import shutil
    shutil.copy(os.path.join(out_dir, "best_model.pt"),
                os.path.join(out_dir, "final_model", "best_model.pt"))

    print(f"Phase 1 saved to {out_dir}")


# =============================================================================
# PHASE 2 TRAINING
# =============================================================================

def p2_train_epoch(model, loader, optimizer, scheduler, device, scaler, use_amp):
    model.train()
    total = 0.0
    optimizer.zero_grad(set_to_none=True)

    for batch in tqdm(loader, desc="P2 Train"):
        ids    = batch["input_ids"].to(device, non_blocking=True)
        mask   = batch["attention_mask"].to(device, non_blocking=True)
        labels = batch["qtype_label"].to(device, non_blocking=True)

        with torch.amp.autocast("cuda", enabled=use_amp):
            out = model(input_ids=ids, attention_mask=mask, qtype_labels=labels)

        scaler.scale(out["loss"]).backward()
        scaler.step(optimizer); scaler.update()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        total += out["loss"].item()

    return total / len(loader)


@torch.no_grad()
def p2_eval_epoch(model, loader, device, use_amp, id_to_qtype):
    model.eval()
    all_pred, all_true = [], []
    total = 0.0

    for batch in tqdm(loader, desc="P2 Eval"):
        ids    = batch["input_ids"].to(device, non_blocking=True)
        mask   = batch["attention_mask"].to(device, non_blocking=True)
        labels = batch["qtype_label"].to(device, non_blocking=True)

        with torch.amp.autocast("cuda", enabled=use_amp):
            out = model(input_ids=ids, attention_mask=mask, qtype_labels=labels)

        total += out["loss"].item()
        preds  = torch.argmax(out["qtype_logits"], dim=1)
        all_pred.extend(preds.cpu().tolist())
        all_true.extend(labels.cpu().tolist())

    target_names = [id_to_qtype[i] for i in sorted(id_to_qtype)]
    return {
        "loss":     total / len(loader),
        "acc":      sum(p == t for p, t in zip(all_pred, all_true)) / len(all_pred),
        "f1_w":     f1_score(all_true, all_pred, average="weighted", zero_division=0),
        "f1_m":     f1_score(all_true, all_pred, average="macro",    zero_division=0),
        "report":   classification_report(all_true, all_pred,
                                          target_names=target_names, zero_division=0),
        "preds":    all_pred,
        "true":     all_true,
    }


def train_phase2(
    base_dir:        str   = "/content/drive/MyDrive/triple_head/",
    pretrained_dir:  str   = None,
    train_file:      str   = None,
    epochs:          int   = 15,
    batch_size:      int   = 16,
    lr_encoder:      float = 2e-5,
    lr_head:         float = 3e-4,
    warmup_ratio:    float = 0.1,
    val_split:       float = 0.15,
    max_length:      int   = 96,
) -> None:
    """
    Phase 2: freeze span + relation heads, train QType head on GeoQuestions1089.
    Produces best_model.pt + config.json for evaluate.py.
    """
    gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()

    if pretrained_dir is None:
        pretrained_dir = os.path.join(base_dir, "ANS/phase1_output")
    if train_file is None:
        train_file = os.path.join(base_dir, "GEO/geo_train.tsv")

    out_dir = os.path.join(base_dir, "GEO/qtype_model_v2")
    os.makedirs(out_dir, exist_ok=True)

    # Load Phase 1 config
    cfg_path = os.path.join(pretrained_dir, "final_model", "config.json")
    with open(cfg_path) as f:
        p1_cfg = json.load(f)

    MODEL_NAME    = p1_cfg["model_name"]
    NUM_RELATIONS = p1_cfg["num_relations"]
    relation_to_id = p1_cfg["relation_to_id"]
    id_to_relation = {int(k): v for k, v in p1_cfg["id_to_relation"].items()}

    # ── Load + merge QType labels ────────────────────────────────────────────
    df = pd.read_csv(train_file, sep="\t")
    print(f"Loaded {len(df)} training examples")

    df["QType_Merged"] = df["QType"].map(merge_qtype)

    print("\n=== Class distribution BEFORE merging ===")
    for k, v in df["QType"].value_counts().items():
        print(f"  {k:<25} {v:>4}")
    print("\n=== Class distribution AFTER merging ===")
    for k, v in df["QType_Merged"].value_counts().items():
        print(f"  {k:<25} {v:>4}")

    qtypes     = sorted(df["QType_Merged"].unique())
    qtype_to_id = {q: i for i, q in enumerate(qtypes)}
    id_to_qtype = {i: q for q, i in qtype_to_id.items()}
    NUM_QTYPES  = len(qtype_to_id)
    print(f"\nFinal merged classes: {NUM_QTYPES}")

    # ── Compute class weights ────────────────────────────────────────────────
    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"

    counts  = Counter(df["QType_Merged"])
    raw_w   = [1.0 / counts[id_to_qtype[i]] for i in range(NUM_QTYPES)]
    total_w = sum(raw_w)
    weights = [w * NUM_QTYPES / total_w for w in raw_w]
    class_weights = torch.tensor(weights, dtype=torch.float32).to(device)

    print("\n=== Class weights ===")
    for i in range(NUM_QTYPES):
        print(f"  {id_to_qtype[i]:<25} count={counts[id_to_qtype[i]]:>4}  weight={weights[i]:.3f}")

    # ── Train/val split ──────────────────────────────────────────────────────
    train_df, val_df = train_test_split(
        df, test_size=val_split, stratify=df["QType_Merged"], random_state=42
    )
    print(f"\nTrain: {len(train_df)}  Val: {len(val_df)}")

    tokenizer    = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
    train_ds     = QTypeDataset(train_df, tokenizer, qtype_to_id, max_length)
    val_ds       = QTypeDataset(val_df,   tokenizer, qtype_to_id, max_length)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              pin_memory=True, num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              pin_memory=True, num_workers=0)

    # ── Build model + load Phase 1 weights ──────────────────────────────────
    model = GeoQABERT(
        model_name=MODEL_NAME,
        num_relations=NUM_RELATIONS,
        num_qtypes=NUM_QTYPES,
        class_weights=class_weights,
    )

    ckpt = os.path.join(pretrained_dir, "final_model", "best_model.pt")
    state = torch.load(ckpt, map_location=device, weights_only=True)
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"\nLoaded Phase 1 weights from {ckpt}")
    print(f"  Missing (expected — new QType head): {len(missing)}")
    print(f"  Unexpected: {len(unexpected)}")

    # Freeze span + relation heads
    for head in [model.span_start, model.span_end,
                 model.rel_dropout, model.rel_classifier]:
        for p in head.parameters():
            p.requires_grad = False

    model.to(device)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {trainable:,}")

    # Differential LR: encoder lower, qtype head higher
    optimizer = AdamW([
        {"params": model.encoder.parameters(),          "lr": lr_encoder},
        {"params": model.qtype_classifier.parameters(), "lr": lr_head},
        {"params": model.qtype_dropout.parameters(),    "lr": lr_head},
    ])
    total_steps = len(train_loader) * epochs
    scheduler   = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(warmup_ratio * total_steps),
        num_training_steps=total_steps,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    best_f1_macro = 0.0
    best_ckpt     = os.path.join(out_dir, "best_model.pt")
    t0            = time.time()

    for epoch in range(1, epochs + 1):
        print(f"\n{'='*60}\nEpoch {epoch}/{epochs}\n{'='*60}")

        tr_loss = p2_train_epoch(model, train_loader, optimizer, scheduler,
                                  device, scaler, use_amp)
        print(f"  Train loss: {tr_loss:.4f}")

        vm = p2_eval_epoch(model, val_loader, device, use_amp, id_to_qtype)
        print(f"  Val   loss={vm['loss']:.4f}  acc={vm['acc']:.4f}  "
              f"F1-w={vm['f1_w']:.4f}  F1-m={vm['f1_m']:.4f}  ← primary metric")

        if vm["f1_m"] > best_f1_macro:
            best_f1_macro = vm["f1_m"]
            torch.save(model.state_dict(), best_ckpt)
            print(f"  New best F1-macro={best_f1_macro:.4f}  saved.")

        if device.type == "cuda": torch.cuda.empty_cache()

    print(f"\nPhase 2 complete in {time.time()-t0:.1f}s")

    # Final report on best checkpoint
    model.load_state_dict(torch.load(best_ckpt, map_location=device, weights_only=True))
    vm = p2_eval_epoch(model, val_loader, device, use_amp, id_to_qtype)
    print(f"\n{'='*60}\nBest model — Validation Report\n{'='*60}")
    print(vm["report"])

    # Save tokenizer + config
    tokenizer.save_pretrained(os.path.join(out_dir, "tokenizer"))
    config = {
        "model_name":      MODEL_NAME,
        "num_relations":   NUM_RELATIONS,
        "num_qtypes":      NUM_QTYPES,
        "max_length":      max_length,
        "qtype_to_id":     qtype_to_id,
        "id_to_qtype":     {str(k): v for k, v in id_to_qtype.items()},
        "relation_to_id":  relation_to_id,
        "id_to_relation":  {str(k): v for k, v in id_to_relation.items()},
        "class_merge_map": CLASS_MERGE_MAP,
        "class_weights":   weights,
        "best_f1_macro":   best_f1_macro,
    }
    cfg_out = os.path.join(out_dir, "config.json")
    with open(cfg_out, "w") as f:
        json.dump(config, f, indent=2)

    print(f"\nSaved to {out_dir}")
    print(f"  best_model.pt      ← load with evaluate.py")
    print(f"  config.json        ← contains id_to_qtype, id_to_relation")
    print(f"  tokenizer/         ← bert-base-uncased tokenizer")
    print(f"Best val F1-macro: {best_f1_macro:.4f}")


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="NeuralGeoQA training")
    sub    = parser.add_subparsers(dest="phase", required=True)

    # Phase 1
    p1 = sub.add_parser("phase1", help="Pretrain span + relation heads on SimpleQuestions")
    p1.add_argument("--base_dir",           default="/content/drive/MyDrive/ANS/triple_head/")
    p1.add_argument("--epochs",             type=int,   default=3)
    p1.add_argument("--batch_size",         type=int,   default=16)
    p1.add_argument("--accumulation_steps", type=int,   default=1)
    p1.add_argument("--lr",                 type=float, default=2e-5)
    p1.add_argument("--max_length",         type=int,   default=96)
    p1.add_argument("--model_name",         default="bert-base-uncased")

    # Phase 2
    p2 = sub.add_parser("phase2", help="Fine-tune QType head on GeoQuestions1089")
    p2.add_argument("--base_dir",        default="/content/drive/MyDrive/triple_head/")
    p2.add_argument("--pretrained_dir",  default=None,
                    help="Path to phase1_output/ (default: base_dir/ANS/phase1_output)")
    p2.add_argument("--train_file",      default=None,
                    help="Path to geo_train.tsv (default: base_dir/GEO/geo_train.tsv)")
    p2.add_argument("--epochs",          type=int,   default=15)
    p2.add_argument("--batch_size",      type=int,   default=16)
    p2.add_argument("--lr_encoder",      type=float, default=2e-5)
    p2.add_argument("--lr_head",         type=float, default=3e-4)
    p2.add_argument("--warmup_ratio",    type=float, default=0.1)
    p2.add_argument("--val_split",       type=float, default=0.15)
    p2.add_argument("--max_length",      type=int,   default=96)

    args = parser.parse_args()

    if args.phase == "phase1":
        train_phase1(
            base_dir=args.base_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
            accumulation_steps=args.accumulation_steps,
            learning_rate=args.lr,
            max_length=args.max_length,
            model_name=args.model_name,
        )
    else:
        train_phase2(
            base_dir=args.base_dir,
            pretrained_dir=args.pretrained_dir,
            train_file=args.train_file,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr_encoder=args.lr_encoder,
            lr_head=args.lr_head,
            warmup_ratio=args.warmup_ratio,
            val_split=args.val_split,
            max_length=args.max_length,
        )
