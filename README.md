# NeuralGeoQA

Geographic question answering over Wikidata using fine-tuned BERT and SPARQL templates.
MSc Thesis — Department of Informatics and Telecommunications, NKUA
Supervisor: Prof. Manolis Koubarakis

---

## What this system does

Given a natural language question like *"Which rivers are in Wales?"*, NeuralGeoQA:
1. Identifies the subject entity span ("wales") and question type (C_class_in) using a fine-tuned BERT model
2. Links "wales" to Wikidata entity Q25 via fuzzy search + SPARQL enrichment
3. Fills a SPARQL template with Q25 and the detected answer type (river = Q4022)
4. Executes against the Wikidata endpoint and returns "River Towy; River Usk; River Dee; ..."

It replaces the rule-based NLP of the prior GeoQA2 system (which targeted YAGO2geo) with neural question understanding, while keeping the template-based query generation approach. It is the first neural KGQA system targeting Wikidata for geospatial questions.

---

## Architecture

```
Question
   │
   ▼
[Stage 1 — Preprocess]
  Wikidata label fetch (parallel batches)
  + Flair/GloVe subject span detection
  → Produces training labels for the model, not used at inference
   │
   ▼  preprocessed TSV
[Stage 2 — Train: Phase 1]
  bert-base-uncased — Span Head (start/end token positions)
                    — Relation Head (125 Wikidata properties)
  Trained on 19,481 Wikidata SimpleQuestions examples
   │
   ▼  pretrained encoder
[Stage 2 — Train: Phase 2]
  Same encoder (fine-tuned at LR=2e-5)
  Span + Relation heads → FROZEN
  New QType Head (8 merged classes, weighted CE)
  Trained on 1,087 GeoQuestions1089 examples
  → Saved as best_model.pt (GeoQABERT)
   │
   ▼
[Stage 3 — Evaluate]
  GeoQABERT runs all 3 heads simultaneously
    │
    ├─ Entity Linking (Wikidata API + SPARQL + fuzzy scoring + disk cache)
    │
    ├─ SPARQL Template Selection (12 question types)
    │
    └─ Wikidata SPARQL execution → formatted answer
```

---

## Two-phase training

The neural model is `bert-base-uncased` with three prediction heads, trained in two phases.

**Phase 1 — Pretrain on Wikidata SimpleQuestions**

The span head and relation head are trained jointly on 19,481 examples. The span head predicts start and end token positions of the subject entity. The relation head classifies the Wikidata property from the [CLS] representation.

| Metric | Value |
|--------|-------|
| Span Accuracy | 79.4% |
| Span F1 | 89.7% |
| Relation Accuracy | 95.3% |

**Phase 2 — Fine-tune QType head on GeoQuestions1089**

The span and relation heads are frozen. A QType classification head is added and trained on 1,087 GeoQuestions with weighted cross-entropy. Class imbalance is handled two ways:

1. **Class merging**: 12 fine-grained types → 8 coarser classes for training. Types with fewer than 10 examples merge into their nearest parent (e.g. `C_class_spatial_rel → C_class`, `B_directional → B_spatial`).
2. **Weighted cross-entropy**: each class weighted inversely to its training frequency.

At inference, merged predictions are split back into 12 types via keyword rules (e.g. `B_spatial + "north of" → B_directional`). Two rule-based overrides additionally correct systematic BERT errors on `B_spatial` and `E_class_near`.

---

## Datasets

| Dataset | Size | Role |
|---------|------|------|
| Wikidata SimpleQuestions (ANS subset) | 19,481 train / 2,821 valid / 5,622 test | Phase 1: span + relation heads |
| GeoQuestions1089 | 1,087 (2 removed, overlap with test set) | Phase 2: QType head |
| GeoQuestions201 | 201 (196 with gold answers) | Out-of-distribution evaluation |

GeoQuestions1089 is heavily imbalanced — `E_class_near` has 307 examples, `B_directional` has 2. GeoQuestions201 has a very different distribution (e.g. `B_boolean` is 15% of the test set but only 1% of training), making it a genuine out-of-distribution test.

---

## Evaluation results (GeoQuestions201)

| Metric | Value |
|--------|-------|
| Question Type Accuracy | 75.6% |
| Entity Linking Success | 94.0% |
| Pipeline Execution Success | 77.6% |
| Answer Accuracy (strict) | 16.1% |
| Answer Accuracy (lenient) | 21.6% |

QType accuracy on in-distribution held-out data: 67.1% (merged classes), 58.5% (fine-grained).
Structured types (G_count, G_superlative, A_attribute) reach 91–100%. Spatial types are hardest — E_class_near reaches 2% in-distribution.

**Root causes of errors:**

| Cause | Share |
|-------|-------|
| Wikidata lacks required triples | 38% |
| No template matched pattern | 18% |
| Count over-inclusive (type hierarchy) | 9% |
| Entity linking failed | 8% |
| QType mismatch → wrong template | 8% |
| Wrong entity returned | 8% |
| Boolean answer inverted | 7% |
| List with zero recall | 4% |

The primary failure is Wikidata data quality, not model errors. SPARQL structure is generally correct when the data exists.

---

## Files

| File | Role |
|------|------|
| `preprocess.py` | Generates training labels: Wikidata label fetch + Flair/GloVe span detection |
| `train.py` | Full two-phase training: Phase 1 (span + relation on SimpleQuestions) and Phase 2 (QType head on GeoQuestions1089) |
| `evaluate.py` | Full pipeline: loads GeoQABERT, entity linking, SPARQL templates, evaluation |
| `main.py` | CLI entry point for all stages |
| `notebooks/NeuralGeoQA_Train.ipynb` | Colab — preprocessing + Phase 1 + Phase 2 training (12 cells) |
| `notebooks/NeuralGeoQA_Eval.ipynb` | Colab — full evaluation pipeline |

**Note on preprocess.py:** The Flair/GloVe span detection runs once to annotate training data. It is not used at inference — BERT predicts spans directly from the question at runtime.

---

## Setup

```bash
pip install transformers torch flair scipy pandas tqdm requests scikit-learn fuzzywuzzy python-Levenshtein
```

---

## Usage

### Preprocessing

```bash
python main.py preprocess --base_dir /data/geoqa
```

Input layout under `base_dir`:
```
not_ans/
  test/annotated_wd_data_test.txt
  train/annotated_wd_data_train.txt
  valid/annotated_wd_data_valid.txt
```

Input TSV (no header): `Subject_ID   Relation_ID   Object_ID   Question`
Output adds Wikidata labels + subject span word indices (Start_Subject, End_Subject).

---

### Training

**Phase 1** — pretrain span + relation heads on Wikidata SimpleQuestions:

```bash
python train.py phase1 \
  --base_dir /data/geoqa \
  --epochs 3 \
  --batch_size 16 \
  --lr 2e-5 \
  --model_name bert-base-uncased
```

Saves encoder + span/relation weights to `phase1_output/final_model/`.

**Phase 2** — freeze span/relation heads, train QType head on GeoQuestions1089:

```bash
python train.py phase2 \
  --base_dir /data/geoqa \
  --pretrained_dir /data/geoqa/phase1_output \
  --train_file /data/geoqa/GEO/geo_train.tsv \
  --epochs 15 \
  --lr_encoder 2e-5 \
  --lr_head 3e-4
```

Saves `qtype_model_v2/best_model.pt` + `config.json` + `tokenizer/` — the files `evaluate.py` loads directly.

---

### Evaluation

```bash
python main.py evaluate \
  --model_dir /data/geoqa/qtype_model_v2 \
  --test_file /data/geoqa/geo_test_201.tsv \
  --output_dir /data/geoqa/evaluation \
  --cache_dir /data/geoqa/cache
```

`qtype_model_v2/` must contain:
```
config.json      # model_name, num_relations, num_qtypes, id_to_qtype, id_to_relation
best_model.pt    # GeoQABERT state dict (Phase 2 output)
tokenizer/
```

---

### Single question

```bash
python main.py ask \
  --model_dir /data/geoqa/qtype_model_v2 \
  --question "Which rivers are in Wales?"
```

---

## Question types

| Type | Example |
|------|---------|
| `A_attribute` | Where is Loch Goil located? |
| `B_boolean` | Is Liverpool part of Scotland? |
| `B_spatial` | Is there a mountain within 20km of Cheshire? |
| `B_directional` | Is Hampshire north of Berkshire? |
| `C_class_in` | Which rivers are in Wales? |
| `C_class_spatial_rel` | Which counties border Lincolnshire? |
| `E_class_near` | Which restaurants are near Edinburgh Castle? |
| `E_class_distance` | Which mountains within 50km of Edinburgh? |
| `F_thematic_spatial` | Which mountains in Scotland are over 1000m? |
| `G_count` | How many counties does England have? |
| `G_superlative` | Which is the longest river in Scotland? |
| `G_comparative` | Is Scotland larger than Wales? |

---

## Comparison with GeoQA2

| | GeoQA2 | NeuralGeoQA |
|--|--------|-------------|
| Question understanding | Rule-based NLP | Fine-tuned BERT |
| Entity recognition | Stanford CoreNLP | BERT span + Wikidata API |
| Target KG | YAGO2geo | Wikidata |
| Query generation | Pattern → templates | QType → templates |
| Training data | None | 19,481 + 1,087 examples |


---


## Future directions

- Learned text-to-SPARQL (T5/BART) to replace hardcoded templates
- Wikidata geospatial normalization layer
- Spatial-aware embeddings
- Hybrid KG + LLM retrieval
