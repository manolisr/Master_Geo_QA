"""
evaluate.py
-----------
Stage 3: Full pipeline evaluation.

Components:
  - GeoQABERT        : custom 3-head model (span, relation, qtype)
  - bert_predict     : runs all three heads with rule-based qtype override
  - link_all_entities: Wikidata entity linking with fuzzy scoring + SPARQL fallback
  - build_query      : maps qtype + entities to SPARQL templates
  - execute_sparql   : executes against Wikidata with retry + disk cache
  - answer_question  : full 4-stage pipeline
  - run_evaluation   : batch evaluation on test TSV
"""

import os, re, gc, json, time, hashlib
import requests
import torch
import torch.nn as nn
import pandas as pd
from tqdm import tqdm
from fuzzywuzzy import fuzz
from transformers import AutoTokenizer, AutoModel, AutoConfig
from sklearn.metrics import f1_score
from typing import Optional

WIKIDATA_API   = "https://www.wikidata.org/w/api.php"
WIKIDATA_SPARQL = "https://query.wikidata.org/sparql"
WD_HEADERS     = {"User-Agent": "NeuralGeoQA/1.0 (MSc thesis)"}
LS             = 'SERVICE wikibase:label { bd:serviceParam wikibase:language "en". }'


# ── Model definition ──────────────────────────────────────────────────────────

class GeoQABERT(nn.Module):
    """
    Single encoder with three prediction heads:
      span_start / span_end  : subject entity span (token-level)
      rel_classifier         : Wikidata relation prediction
      qtype_classifier       : question type (A/B/C/E/F/G classes)
    """

    def __init__(self, model_name: str, num_relations: int, num_qtypes: int, dropout: float = 0.1):
        super().__init__()
        self.config          = AutoConfig.from_pretrained(model_name)
        self.encoder         = AutoModel.from_pretrained(model_name)
        hidden               = self.config.hidden_size
        self.span_start      = nn.Linear(hidden, 1)
        self.span_end        = nn.Linear(hidden, 1)
        self.rel_dropout     = nn.Dropout(dropout)
        self.rel_classifier  = nn.Linear(hidden, num_relations)
        self.qtype_dropout   = nn.Dropout(dropout)
        self.qtype_classifier = nn.Linear(hidden, num_qtypes)
        self.loss_fn         = nn.CrossEntropyLoss()

    def forward(self, input_ids, attention_mask, qtype_labels=None):
        out      = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        seq_out  = out.last_hidden_state
        cls_out  = seq_out[:, 0, :]
        start_logits  = self.span_start(seq_out).squeeze(-1)
        end_logits    = self.span_end(seq_out).squeeze(-1)
        rel_logits    = self.rel_classifier(self.rel_dropout(cls_out))
        qtype_logits  = self.qtype_classifier(self.qtype_dropout(cls_out))
        loss = self.loss_fn(qtype_logits, qtype_labels) if qtype_labels is not None else None
        return {
            "loss": loss,
            "start_logits": start_logits,
            "end_logits":   end_logits,
            "rel_logits":   rel_logits,
            "qtype_logits": qtype_logits,
        }


# ── Model loading ─────────────────────────────────────────────────────────────

_model = _tokenizer = _device = None
_id_to_qtype = _id_to_relation = {}
_MAX_LEN = 96


def load_model(model_dir: str) -> None:
    global _model, _tokenizer, _device, _id_to_qtype, _id_to_relation, _MAX_LEN

    _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open(os.path.join(model_dir, "config.json")) as f:
        cfg = json.load(f)

    _model = GeoQABERT(
        model_name=cfg["model_name"],
        num_relations=cfg["num_relations"],
        num_qtypes=cfg["num_qtypes"],
    )
    state = torch.load(
        os.path.join(model_dir, "best_model.pt"),
        map_location=_device, weights_only=True,
    )
    _model.load_state_dict(state, strict=False)
    _model.to(_device)
    _model.eval()

    _tokenizer      = AutoTokenizer.from_pretrained(
        os.path.join(model_dir, "tokenizer"), use_fast=True
    )
    _id_to_qtype    = {int(k): v for k, v in cfg["id_to_qtype"].items()}
    _id_to_relation = {int(k): v for k, v in cfg.get("id_to_relation", {}).items()}
    _MAX_LEN        = cfg.get("max_length", 96)

    print(f"Model loaded on {_device}  qtypes={len(_id_to_qtype)}  relations={len(_id_to_relation)}")


# ── BERT inference + QType override ──────────────────────────────────────────

def refine_qtype(merged: str, question: str) -> str:
    """Split merged BERT classes into fine-grained template types."""
    ql = question.lower()
    if merged == "B_spatial":
        if re.search(r"\b(north|south|east|west|northeast|northwest|southeast|southwest)\s+of\b", ql):
            return "B_directional"
    if merged == "C_class":
        if any(kw in ql for kw in ["border", "cross", "flow", "discharge", "run through", "flows through"]):
            return "C_class_spatial_rel"
        return "C_class_in"
    if merged == "E_class_near":
        if re.search(r"\d+\s*(km|kilometer|meter|m\b|mile)", ql):
            return "E_class_distance"
        if any(kw in ql for kw in ["at most", "within", "range of", "radius"]):
            return "E_class_distance"
    if merged == "G_superlative":
        if any(kw in ql for kw in ["bigger than", "larger than", "longer than", "taller than",
                                    "more counties than", "more districts than"]):
            return "G_comparative"
    return merged


def override_qtype(bert_qtype: str, question: str) -> str:
    """Rule-based override for classes where BERT systematically fails."""
    ql = question.lower()
    if bert_qtype == "B_spatial":
        if re.search(r"\b(north|south|east|west|northeast|northwest|southeast|southwest)\s+of\b", ql):
            return "B_directional"
        if not re.search(r"\b(near|km|close|within|range|away|radius|there|any)", ql):
            return "B_boolean"
    if bert_qtype == "E_class_near":
        if not re.search(r"\b(near|close|km|within|away|radius|at most|nearest|closest)", ql):
            if re.search(r"\b(border|cross|flow|discharge|run through|flows through)", ql):
                return "C_class_spatial_rel"
            return "C_class_in"
    return refine_qtype(bert_qtype, question)


@torch.no_grad()
def bert_predict(question: str) -> dict:
    """Run GeoQABERT, return span text, top-3 relations, and final qtype."""
    enc = _tokenizer(
        str(question), add_special_tokens=True, max_length=_MAX_LEN,
        padding="max_length", truncation=True, return_tensors="pt",
    )
    ids  = enc["input_ids"].to(_device)
    mask = enc["attention_mask"].to(_device)
    out  = _model(input_ids=ids, attention_mask=mask)

    si = int(torch.argmax(out["start_logits"], dim=1).item())
    ei = int(torch.argmax(out["end_logits"],   dim=1).item())
    if ei < si: ei = si
    span_text = _tokenizer.decode(ids[0, si:ei + 1], skip_special_tokens=True)
    span_text = re.sub(r"[?.,!;:]+$", "", span_text).strip()

    rel_probs = torch.softmax(out["rel_logits"], dim=1)[0]
    topk      = torch.topk(rel_probs, min(3, rel_probs.size(0)))
    relations = []
    for i, p in zip(topk.indices, topk.values):
        rid     = _id_to_relation.get(int(i), str(int(i)))
        inverse = False
        m = re.match(r"^R(\d+)$", str(rid))
        if m:
            rid, inverse = f"P{m.group(1)}", True
        relations.append({"id": rid, "prob": float(p), "inverse": inverse})

    qtype_probs = torch.softmax(out["qtype_logits"], dim=1)[0]
    qtype_idx   = int(torch.argmax(qtype_probs).item())
    bert_qtype  = _id_to_qtype[qtype_idx]

    return {
        "span_text":         span_text,
        "relations":         relations,
        "qtype":             override_qtype(bert_qtype, question),
        "qtype_bert_raw":    bert_qtype,
        "qtype_confidence":  float(qtype_probs[qtype_idx]),
    }


# ── Wikidata caching + rate limiting ─────────────────────────────────────────

_memory_cache: dict = {}
_cache_file:   str  = ""
_last_api_t    = _last_sparql_t = 0.0


def init_cache(cache_dir: str) -> None:
    global _memory_cache, _cache_file
    os.makedirs(cache_dir, exist_ok=True)
    _cache_file = os.path.join(cache_dir, "wikidata_cache.json")
    if os.path.exists(_cache_file):
        try:
            with open(_cache_file) as f:
                _memory_cache = json.load(f)
            print(f"Cache loaded: {len(_memory_cache)} entries")
        except Exception:
            _memory_cache = {}


def save_cache() -> None:
    if _cache_file:
        with open(_cache_file, "w") as f:
            json.dump(_memory_cache, f)


def _ckey(prefix: str, val: str) -> str:
    return f"{prefix}:{hashlib.md5(val.encode()).hexdigest()[:12]}"


def _cget(prefix, val):
    return _memory_cache.get(_ckey(prefix, val))


def _cset(prefix, val, result) -> None:
    _memory_cache[_ckey(prefix, val)] = result


def _rl(is_sparql: bool = False) -> None:
    global _last_api_t, _last_sparql_t
    limit = 1.5 if is_sparql else 0.8
    ref   = _last_sparql_t if is_sparql else _last_api_t
    wait  = limit - (time.time() - ref)
    if wait > 0:
        time.sleep(wait)
    if is_sparql:
        _last_sparql_t = time.time()
    else:
        _last_api_t = time.time()


def _get(url, params, is_sparql=False, retries=3, timeout=20):
    for attempt in range(retries):
        _rl(is_sparql)
        try:
            r = requests.get(url, params=params, headers=WD_HEADERS, timeout=timeout)
            if r.status_code == 429:
                time.sleep(5 * (attempt + 1)); continue
            r.raise_for_status()
            return r
        except requests.exceptions.Timeout:
            if attempt < retries - 1: time.sleep(3)
    return None


# ── Entity linking ────────────────────────────────────────────────────────────

GEO_KEYWORDS = {
    "city", "town", "village", "county", "river", "lake", "mountain",
    "castle", "bridge", "park", "church", "museum", "station", "airport",
    "district", "region", "country", "island", "forest", "monument",
    "england", "scotland", "wales", "ireland", "london", "united kingdom",
    "greece", "greek", "athens", "republic",
}
STOP_WORDS = {
    "which", "what", "where", "how", "is", "does", "are", "do", "the",
    "a", "an", "there", "that", "through", "in", "many", "has", "have",
    "near", "from", "more", "less", "than", "most", "any", "can", "find",
    "at", "between", "and", "not", "no", "its", "it", "much", "some",
    "all", "every", "with", "on", "to", "by", "for", "or", "as", "be",
    "was", "were", "been", "being", "into", "over", "under", "about",
}


def wd_search(query: str, limit: int = 10) -> list:
    cached = _cget("search", query)
    if cached is not None: return cached
    r = _get(WIKIDATA_API, {"action": "wbsearchentities", "search": query,
                             "language": "en", "limit": limit, "format": "json"})
    if r is None: return wd_search_sparql(query, limit)
    try:
        results = r.json().get("search", [])
        _cset("search", query, results)
        return results
    except Exception:
        return wd_search_sparql(query, limit)


def wd_search_sparql(query: str, limit: int = 10) -> list:
    cached = _cget("sparql_search", query)
    if cached is not None: return cached
    sparql = f"""SELECT ?item ?itemLabel ?itemDescription WHERE {{
      SERVICE wikibase:mwapi {{
        bd:serviceParam wikibase:api "EntitySearch" .
        bd:serviceParam wikibase:endpoint "www.wikidata.org" .
        bd:serviceParam mwapi:search "{query}" .
        bd:serviceParam mwapi:language "en" .
        ?item wikibase:apiOutputItem mwapi:item .
      }}
      SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }}
    }} LIMIT {limit}"""
    r = _get(WIKIDATA_SPARQL, {"query": sparql, "format": "json"}, is_sparql=True)
    if r is None: return []
    try:
        bindings = r.json().get("results", {}).get("bindings", [])
        results  = [{"id": b["item"]["value"].split("/")[-1],
                     "label": b.get("itemLabel", {}).get("value", query),
                     "description": b.get("itemDescription", {}).get("value", "")}
                    for b in bindings]
        _cset("sparql_search", query, results)
        return results
    except Exception:
        return []


def wd_details_batch(qids: list[str]) -> dict[str, dict]:
    results, uncached = {}, []
    for qid in qids:
        c = _cget("details", qid)
        if c: results[qid] = c
        else: uncached.append(qid)
    if not uncached: return results

    values = " ".join(f"wd:{q}" for q in uncached[:5])
    sparql = f"""SELECT ?entity ?typeLabel ?coord ?adminLabel ?countryLabel WHERE {{
      VALUES ?entity {{ {values} }}
      OPTIONAL {{ ?entity wdt:P31 ?type }}
      OPTIONAL {{ ?entity wdt:P625 ?coord }}
      OPTIONAL {{ ?entity wdt:P131 ?admin }}
      OPTIONAL {{ ?entity wdt:P17 ?country }}
      SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en" }}
    }}"""
    r = _get(WIKIDATA_SPARQL, {"query": sparql, "format": "json"}, is_sparql=True, timeout=25)
    for qid in uncached:
        results.setdefault(qid, {"types": [], "coordinates": None, "admin": None, "country": None})
    if r:
        try:
            for b in r.json().get("results", {}).get("bindings", []):
                qid = b["entity"]["value"].split("/")[-1]
                d   = results.setdefault(qid, {"types": [], "coordinates": None, "admin": None, "country": None})
                if "typeLabel" in b and b["typeLabel"]["value"] not in d["types"]:
                    d["types"].append(b["typeLabel"]["value"])
                if "coord" in b and not d["coordinates"]:
                    d["coordinates"] = b["coord"]["value"]
                if "adminLabel" in b and not d["admin"]:
                    d["admin"] = b["adminLabel"]["value"]
                if "countryLabel" in b and not d["country"]:
                    d["country"] = b["countryLabel"]["value"]
        except Exception:
            pass
    for qid, d in results.items():
        _cset("details", qid, d)
    return results


def clean_span(s: str) -> str:
    s = re.sub(r"[?.,!;:]+$", "", s.strip()).strip()
    return s.title() if s and s[0].islower() else s


def make_variants(span: str) -> list[str]:
    variants = [span]
    for pat in [r"^(?:county|county of|the county of)\s+",
                r"^(?:river|the river)\s+",
                r"^(?:city of|the city of)\s+",
                r"^(?:the|a|an)\s+",
                r"^(?:mount|lake|loch)\s+"]:
        stripped = re.sub(pat, "", span, flags=re.I).strip()
        if stripped and stripped != span and len(stripped) > 1:
            variants.append(stripped)
    if not span.lower().startswith("river") and len(span.split()) == 1:
        variants.append(f"River {span}")
    seen = set()
    return [v for v in variants if v.lower() not in seen and not seen.add(v.lower())]


def score_candidates(candidates: list, span: str, question: str = "") -> list:
    sl = span.lower()
    for c in candidates:
        label = c.get("label", "")
        desc  = c.get("description", "").lower()
        sim   = max(fuzz.ratio(sl, label.lower()),
                    fuzz.partial_ratio(sl, label.lower()),
                    fuzz.token_sort_ratio(sl, label.lower())) / 100.0
        geo   = 0.10 if any(kw in desc for kw in GEO_KEYWORDS) else 0.0
        exact = 0.15 if sl == label.lower() else 0.0
        ctx   = min(sum(
            0.05 for cw in re.findall(r"\b[A-Z][a-z]{2,}\b", question)
            if cw.lower() != sl and cw.lower() in desc
        ), 0.10)
        c["score"] = round(sim + geo + exact + ctx, 4)
    candidates.sort(key=lambda c: c["score"], reverse=True)
    return candidates


def enrich(candidates: list, top_k: int = 5) -> list:
    top     = candidates[:top_k]
    details = wd_details_batch([c["id"] for c in top])
    for c in top:
        d = details.get(c["id"], {"types": [], "coordinates": None, "admin": None, "country": None})
        c["details"]     = d
        c["has_coords"]  = bool(d["coordinates"])
        c["types"]       = d["types"]
        c["country"]     = d["country"]
        if d["coordinates"]:
            c["score"] += 0.15
    candidates.sort(key=lambda c: c["score"], reverse=True)
    return candidates


def truncate_at_boundary(s: str) -> str:
    for bw in [" in ", " of ", " border ", " near ", " within ", " have ", " has ",
               " north ", " south ", " east ", " west ", " less ", " more ",
               " at ", " with ", " from ", " to ", " is ", " are ", " located",
               " situated", " cross", " flow", " run", " part"]:
        idx = s.lower().find(bw)
        if idx != -1:
            return s[:idx].strip()
    return re.sub(r"\d+\s*(km|kilometer|meter|m|mile|miles)\b.*$", "", s, flags=re.I).strip()


def link_entity(span: str, question: str = "", top_k: int = 3) -> list:
    if not span or not span.strip(): return []
    cs = clean_span(span)
    strategies = [("original", span), ("cleaned", cs),
                  ("proper", re.split(r"\b(?:in|of|near|from|within)\b", cs, flags=re.I)[0].strip()),
                  ("truncated", truncate_at_boundary(span))]
    seen, unique = set(), []
    for name, val in strategies:
        if val and val.lower() not in seen:
            seen.add(val.lower()); unique.append((name, val))

    all_cands = {}
    for name, query in unique:
        for c in (lambda q: [r for v in make_variants(q) for r in wd_search(v)])(query):
            all_cands.setdefault(c["id"], c)
        if any(c.get("score", 0) > 0.8 for c in all_cands.values()): break

    if not all_cands: return []
    cands = score_candidates(list(all_cands.values()), cs, question)
    cands = enrich(cands, top_k=min(top_k + 2, 5))
    return cands[:top_k]


def extract_secondary(question: str, primary_span: str) -> list[str]:
    pl = primary_span.lower() if primary_span else ""
    words, entities, i = question.split(), [], 1
    while i < len(words):
        w = re.sub(r"[?.,!;:]+$", "", words[i])
        if w and w[0].isupper() and w.lower() not in STOP_WORDS:
            ent_words, j = [w], i + 1
            while j < len(words):
                nw = re.sub(r"[?.,!;:]+$", "", words[j])
                if nw and nw[0].isupper() and nw.lower() not in STOP_WORDS:
                    ent_words.append(nw); j += 1
                elif nw.lower() in ("of", "the", "and", "on", "in", "de") and j+1<len(words) and words[j+1][0].isupper():
                    ent_words.append(nw); j += 1
                else: break
            name = " ".join(ent_words)
            if name.lower() != pl and pl not in name.lower() and name.lower() not in pl and len(name) > 1:
                entities.append(name)
            i = j
        else: i += 1
    return entities


def link_all_entities(question: str, span: str, qtype: str = "") -> dict:
    result = {"primary": None, "secondary": [], "distance_km": None, "numeric_constraint": None}
    if span and span.strip():
        cands = link_entity(span, question, top_k=3)
        if cands:
            b = cands[0]
            result["primary"] = {
                "qid": b["id"], "label": b["label"],
                "description": b.get("description", ""),
                "score": b["score"],
                "coordinates": b.get("details", {}).get("coordinates"),
                "country": b.get("country"),
                "types": b.get("types", []),
                "all_candidates": cands,
            }
    for sp in extract_secondary(question, span)[:2]:
        cands = link_entity(sp, question, top_k=1)
        if cands:
            result["secondary"].append({"qid": cands[0]["id"], "label": cands[0]["label"],
                                        "span": sp, "score": cands[0]["score"]})
    dm = re.search(r"(\d+(?:\.\d+)?)\s*(km|kilometer|meter|m\b|mile)", question, re.I)
    if dm:
        val, unit = float(dm.group(1)), dm.group(2).lower()
        if unit in ("m", "meter"):  val /= 1000
        elif unit == "mile": val *= 1.609
        result["distance_km"] = val
    nm = re.search(r"(more than|over|less than|at least|at most|taller than|bigger than)\s+(\d+)", question, re.I)
    if nm:
        result["numeric_constraint"] = f"{nm.group(1)} {nm.group(2)}"
    return result


# ── SPARQL execution ──────────────────────────────────────────────────────────

def execute_sparql(sparql: str) -> dict:
    if not sparql or not sparql.strip():
        return {"success": False, "results": [], "ask_result": None, "error": "empty query"}
    cached = _cget("exec", sparql)
    if cached is not None: return cached
    r = _get(WIKIDATA_SPARQL, {"query": sparql, "format": "json"},
             is_sparql=True, retries=3, timeout=45)
    if r is None:
        return {"success": False, "results": [], "ask_result": None, "error": "max retries"}
    try:
        data = r.json()
        if "boolean" in data:
            result = {"success": True, "results": [], "ask_result": data["boolean"], "error": None}
        else:
            rows   = [{k: v["value"] for k, v in b.items()} for b in data.get("results", {}).get("bindings", [])]
            result = {"success": True, "results": rows, "ask_result": None, "error": None}
        _cset("exec", sparql, result)
        return result
    except Exception as e:
        return {"success": False, "results": [], "ask_result": None, "error": str(e)[:120]}


# ── SPARQL query builder ──────────────────────────────────────────────────────

def detect_answer_type(question: str) -> str:
    ql = question.lower()
    if re.match(r"^(is |does |are )", ql): return "BOOLEAN"
    if re.match(r"^how many\b", ql): return "NUMERIC|count"
    m = re.match(r"^which\s+(\w+(?:\s+\w+)?)\s", ql)
    if m:
        asked = m.group(1).strip().lower()
        type_map = {
            "river": "Q4022|river", "rivers": "Q4022|river",
            "county": "Q28575|county", "counties": "Q28575|county",
            "mountain": "Q8502|mountain", "mountains": "Q8502|mountain",
            "city": "Q515|city", "cities": "Q515|city",
            "town": "Q515|city", "towns": "Q515|city",
            "lake": "Q23397|lake", "bridge": "Q12280|bridge",
            "bridges": "Q12280|bridge", "castle": "Q23413|castle",
            "airport": "Q1248784|airport", "hospital": "Q16917|hospital",
            "restaurant": "Q11707|restaurant", "restaurants": "Q11707|restaurant",
            "pub": "Q212198|pub", "hotel": "Q27686|hotel",
            "museum": "Q33506|museum", "park": "Q22698|park",
            "university": "Q3918|university", "village": "Q532|village",
        }
        for word, t in type_map.items():
            if word in asked: return t
    tmap = [
        (r"\bcounty\b|\bcounties\b", "Q28575|county"),
        (r"\briver\b|\brivers\b", "Q4022|river"),
        (r"\bmountain", "Q8502|mountain"),
        (r"\bcity\b|\bcities\b|\btown", "Q515|city"),
        (r"\blake", "Q23397|lake"),
        (r"\bcastle", "Q23413|castle"),
        (r"\bbridge", "Q12280|bridge"),
        (r"\bairport|\baerodrome", "Q1248784|airport"),
        (r"\bhospital", "Q16917|hospital"),
        (r"\brestaurant", "Q11707|restaurant"),
        (r"\bpub\b", "Q212198|pub"),
        (r"\bhotel", "Q27686|hotel"),
        (r"\bmuseum", "Q33506|museum"),
        (r"\bpark\b|\bparks\b", "Q22698|park"),
        (r"\bvillage", "Q532|village"),
    ]
    for pat, t in tmap:
        if re.search(pat, ql): return t
    return ""


def _resolve_aq(ans_type: str) -> Optional[str]:
    m = re.match(r"(Q\d+)", ans_type or "")
    return m.group(1) if m else None


def _type_filter(aq: Optional[str], var: str = "?x") -> str:
    if not aq: return ""
    expansions = {
        "Q515":  f"{{ {var} wdt:P31 wd:Q515 }} UNION {{ {var} wdt:P31 wd:Q1549591 }}",
        "Q3918": f"{{ {var} wdt:P31 wd:Q3918 }} UNION {{ {var} wdt:P31 wd:Q875538 }}",
        "Q28575": (f"{{ {var} wdt:P31 wd:Q28575 }} UNION "
                   f"{{ {var} wdt:P31 wd:Q180673 }} UNION "
                   f"{{ {var} wdt:P31 wd:Q1187580 }}"),
    }
    return expansions.get(aq, f"{var} wdt:P31 wd:{aq} .")


def parse_coords(coord_str: str) -> tuple[Optional[float], Optional[float]]:
    if not coord_str: return None, None
    m = re.search(r"Point\(([-\d.]+)\s+([-\d.]+)\)", coord_str)
    return (float(m.group(2)), float(m.group(1))) if m else (None, None)


def build_query(
    qtype: str,
    entity_qid: Optional[str],
    relation: str = "",
    answer_type: str = "",
    entity_coords: Optional[str] = None,
    distance_km: Optional[float] = None,
    numeric_constraint: Optional[str] = None,
    secondary_qid: Optional[str] = None,
    question: str = "",
) -> dict:
    aq = _resolve_aq(answer_type)
    tf = _type_filter(aq)
    ql = question.lower()

    def lf(transitive=False):
        if not entity_qid: return ""
        return f"?x wdt:P131+ wd:{entity_qid} ." if transitive else f"?x wdt:P131 wd:{entity_qid} ."

    if qtype == "A_attribute":
        if "location" in relation.lower() or "where" in ql:
            return {"query": f"""SELECT ?coord ?adminLabel ?countryLabel WHERE {{
  OPTIONAL {{ wd:{entity_qid} wdt:P625 ?coord }}
  OPTIONAL {{ wd:{entity_qid} wdt:P131 ?admin }}
  OPTIONAL {{ wd:{entity_qid} wdt:P17 ?country }}
  {LS}
}} LIMIT 5""", "template": "A_location"}
        rp = re.match(r"(P\d+)", relation)
        if rp:
            return {"query": f"SELECT ?value ?valueLabel WHERE {{ wd:{entity_qid} wdt:{rp.group(1)} ?value . {LS} }} LIMIT 10",
                    "template": "A_property"}
        return {"query": "", "template": "A_fail"}

    if qtype == "B_boolean":
        if "border" in ql and secondary_qid:
            return {"query": f"ASK {{ wd:{entity_qid} wdt:P47 wd:{secondary_qid} }}", "template": "B_border"}
        if secondary_qid:
            return {"query": f"""ASK {{
  {{ wd:{entity_qid} wdt:P131 wd:{secondary_qid} }}
  UNION {{ wd:{entity_qid} wdt:P131/wdt:P131 wd:{secondary_qid} }}
  UNION {{ wd:{entity_qid} wdt:P17 wd:{secondary_qid} }}
}}""", "template": "B_containment"}
        return {"query": f"ASK {{ wd:{entity_qid} ?p ?o }}", "template": "B_fallback"}

    if qtype == "B_spatial":
        radius = distance_km or 5.0
        if entity_coords and aq:
            lat, lon = parse_coords(entity_coords)
            if lat:
                return {"query": f"""ASK {{
  {tf}
  SERVICE wikibase:around {{
    ?x wdt:P625 ?loc .
    bd:serviceParam wikibase:center "Point({lon} {lat})"^^geo:wktLiteral .
    bd:serviceParam wikibase:radius "{radius}" .
  }}
}}""", "template": "B_spatial_around"}
        return {"query": "", "template": "B_spatial_fail"}

    if qtype == "B_directional":
        if secondary_qid:
            return {"query": f"""SELECT ?lat1 ?lon1 ?lat2 ?lon2 WHERE {{
  wd:{entity_qid} p:P625/psv:P625 [wikibase:geoLatitude ?lat1; wikibase:geoLongitude ?lon1] .
  wd:{secondary_qid} p:P625/psv:P625 [wikibase:geoLatitude ?lat2; wikibase:geoLongitude ?lon2] .
}}""", "template": "B_directional"}
        return {"query": "", "template": "B_dir_fail"}

    if qtype == "C_class_in" and aq:
        return {"query": f"""SELECT DISTINCT ?x ?xLabel WHERE {{
  {tf}
  {{ ?x wdt:P131 wd:{entity_qid} }}
  UNION {{ ?x wdt:P131/wdt:P131 wd:{entity_qid} }}
  UNION {{ ?x wdt:P17 wd:{entity_qid} }}
  {LS}
}} LIMIT 200""", "template": "C_class_in"}

    if qtype == "C_class_spatial_rel":
        if "border" in ql:
            base = f"{tf}\n  ?x wdt:P47 wd:{entity_qid} ." if aq else f"?x wdt:P47 wd:{entity_qid} ."
            return {"query": f"SELECT DISTINCT ?x ?xLabel WHERE {{ {base} {LS} }} LIMIT 100", "template": "C_border"}
        if any(kw in ql for kw in ["cross", "flow", "discharge", "run through"]) and aq:
            return {"query": f"""SELECT DISTINCT ?x ?xLabel WHERE {{
  {tf}
  {{ ?x wdt:P177 wd:{entity_qid} }}
  UNION {{ wd:{entity_qid} wdt:P177 ?x }}
  UNION {{ ?x wdt:P206 wd:{entity_qid} }}
  {LS}
}} LIMIT 100""", "template": "C_crosses"}
        return {"query": "", "template": "C_spatial_fail"}

    if qtype in ("E_class_near", "E_class_distance"):
        radius = distance_km or 5.0
        if entity_coords:
            lat, lon = parse_coords(entity_coords)
            if lat:
                return {"query": f"""SELECT DISTINCT ?x ?xLabel ?dist WHERE {{
  {tf}
  SERVICE wikibase:around {{
    ?x wdt:P625 ?loc .
    bd:serviceParam wikibase:center "Point({lon} {lat})"^^geo:wktLiteral .
    bd:serviceParam wikibase:radius "{radius}" .
  }}
  BIND(geof:distance(?loc, "Point({lon} {lat})"^^geo:wktLiteral) AS ?dist)
  {LS}
}} ORDER BY ?dist LIMIT 50""", "template": "E_around"}
        return {"query": "", "template": "E_fail"}

    if qtype == "F_thematic_spatial":
        constraint = ""
        if numeric_constraint:
            m_gt = re.match(r"(more than|over|at least)\s+(\d+)", numeric_constraint, re.I)
            m_lt = re.match(r"(less than|at most)\s+(\d+)", numeric_constraint, re.I)
            val  = m_gt.group(2) if m_gt else (m_lt.group(2) if m_lt else None)
            op   = ">" if m_gt else ("<" if m_lt else None)
            if val and op:
                prop = ("P2044" if any(k in ql for k in ["height", "tall", "elevation"])
                        else "P1082" if "population" in ql
                        else "P2043" if "length" in ql
                        else "P2046")
                constraint = f"?x wdt:{prop} ?v . FILTER(?v {op} {val})"
        return {"query": f"""SELECT DISTINCT ?x ?xLabel ?v WHERE {{
  {tf}
  {lf(transitive=True)}
  {constraint}
  {LS}
}} LIMIT 100""", "template": "F_thematic"}

    if qtype == "G_count":
        return {"query": f"""SELECT (COUNT(DISTINCT ?x) AS ?count) WHERE {{
  {tf}
  {lf()}
}}""", "template": "G_count"}

    if qtype == "G_superlative":
        op_map = {
            "largest": ("P2046", "DESC"), "biggest": ("P2046", "DESC"),
            "longest": ("P2043", "DESC"), "highest": ("P2044", "DESC"),
            "tallest": ("P2044", "DESC"), "smallest": ("P2046", "ASC"),
            "oldest": ("P571", "ASC"),    "most populated": ("P1082", "DESC"),
        }
        for kw, (prop, order) in op_map.items():
            if kw in ql:
                return {"query": f"""SELECT ?x ?xLabel ?val WHERE {{
  {tf}
  {lf(transitive=True)}
  ?x wdt:{prop} ?val .
  {LS}
}} ORDER BY {order}(?val) LIMIT 1""", "template": "G_superlative"}
        return {"query": "", "template": "G_super_fail"}

    if qtype == "G_comparative" and secondary_qid:
        cp = ("P1082" if "population" in ql else "P2043" if "long" in ql
              else "P2044" if any(k in ql for k in ["tall", "high"]) else "P2046")
        return {"query": f"SELECT ?v1 ?v2 WHERE {{ wd:{entity_qid} wdt:{cp} ?v1 . wd:{secondary_qid} wdt:{cp} ?v2 . }}",
                "template": "G_comparative"}

    return {"query": "", "template": "UNKNOWN"}


# ── Answer formatter ──────────────────────────────────────────────────────────

def format_answer(qtype: str, result: dict, question: str = "") -> str:
    if result["ask_result"] is not None:
        return "Yes" if result["ask_result"] else "No"
    res = result.get("results", [])
    if not res: return "No results found"
    if qtype == "G_count":
        for r in res:
            if "count" in r: return str(r["count"])
        return str(len(res))
    if qtype == "G_comparative":
        try: return "Yes" if float(res[0]["v1"]) > float(res[0]["v2"]) else "No"
        except: return f"{res[0].get('v1','?')} vs {res[0].get('v2','?')}"
    if qtype == "B_directional" and res:
        try:
            r  = res[0]
            ql = question.lower()
            lat1, lat2 = float(r["lat1"]), float(r["lat2"])
            lon1, lon2 = float(r["lon1"]), float(r["lon2"])
            if "north" in ql: return "Yes" if lat1 > lat2 else "No"
            if "south" in ql: return "Yes" if lat1 < lat2 else "No"
            if "east"  in ql: return "Yes" if lon1 > lon2 else "No"
            if "west"  in ql: return "Yes" if lon1 < lon2 else "No"
        except: pass
        return "Unable to determine"
    if qtype in ("A_attribute", "G_superlative"):
        r = res[0]
        for key in ["xLabel", "valueLabel", "adminLabel", "countryLabel", "coord"]:
            if key in r and r[key] and "entity/" not in r[key]:
                return r[key]
    labels = [
        r[key] for r in res
        for key in ["xLabel", "valueLabel"]
        if key in r and "entity/" not in r[key]
    ]
    return "; ".join(labels[:30]) if labels else "No results"


# ── Full pipeline ─────────────────────────────────────────────────────────────

def answer_question(question: str, verbose: bool = False) -> dict:
    out = {"question": question, "success": False}

    bert = bert_predict(question)
    out["bert"] = bert
    if verbose:
        print(f"  BERT span='{bert['span_text']}' qtype={bert['qtype']} rel={bert['relations'][0]['id']}")

    ans_type = detect_answer_type(question)
    linking  = link_all_entities(question, bert["span_text"], bert["qtype"])
    out["linking"]      = linking
    out["answer_type"]  = ans_type

    eq = linking["primary"]["qid"]   if linking["primary"] else None
    ec = linking["primary"].get("coordinates") if linking["primary"] else None
    sq = linking["secondary"][0]["qid"] if linking["secondary"] else None

    if verbose and eq:
        print(f"  Link  {eq} ({linking['primary']['label']}) coords={'yes' if ec else 'no'}")
    if not eq:
        out["answer"] = "Entity not found"; out["error"] = "entity_linking_failed"
        return out

    rel_str = bert["relations"][0]["id"] if bert["relations"] else ""
    qr = build_query(
        qtype=bert["qtype"], entity_qid=eq, relation=rel_str,
        answer_type=ans_type, entity_coords=ec,
        distance_km=linking["distance_km"],
        numeric_constraint=linking["numeric_constraint"],
        secondary_qid=sq, question=question,
    )
    out["sparql"]   = qr["query"]
    out["template"] = qr["template"]
    if verbose: print(f"  Template {qr['template']}")

    if not qr["query"]:
        out["answer"] = "No query generated"; out["error"] = qr.get("notes", "")
        return out

    result = execute_sparql(qr["query"])

    # Fallback: drop type filter if empty results
    if result["success"] and not result["results"] and result["ask_result"] is None:
        aq = _resolve_aq(ans_type)
        if aq and f"wd:{aq}" in qr["query"]:
            fb = re.sub(r"\?x wdt:P31/wdt:P279\* wd:Q\d+ \.\s*", "", qr["query"])
            if fb.strip() != qr["query"].strip():
                r2 = execute_sparql(fb)
                if r2["success"] and (r2["results"] or r2["ask_result"] is not None):
                    result = r2; out["template"] += "_no_type"

    out["answer"]    = format_answer(bert["qtype"], result, question)
    out["success"]   = result["success"]
    out["error"]     = result.get("error")
    out["n_results"] = len(result.get("results", []))

    if verbose:
        print(f"  Exec  success={result['success']} n={out['n_results']}")
        print(f"  ANSWER: {str(out['answer'])[:120]}")
    return out


def ask(question: str) -> dict:
    """Interactive single-question helper."""
    print(f"\nQ: {question}\n" + "-" * 60)
    r = answer_question(question, verbose=True)
    print(f"\nANSWER: {r['answer']}")
    return r


# ── Batch evaluation ──────────────────────────────────────────────────────────

def run_evaluation(test_file: str, output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    df = pd.read_csv(test_file, sep="\t")
    print(f"Loaded {len(df)} test questions")

    results = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Evaluating"):
        q       = row["Question"]
        gold_qt = row.get("QType", "")
        r       = answer_question(q, verbose=False)
        results.append({
            "TestID":        row.get("TestID", _),
            "Question":      q,
            "Gold_QType":    gold_qt,
            "Pred_QType":    r["bert"]["qtype"],
            "QType_Match":   r["bert"]["qtype"] == gold_qt if gold_qt else None,
            "Pred_Span":     r["bert"]["span_text"],
            "Pred_Relation": r["bert"]["relations"][0]["id"] if r["bert"]["relations"] else "",
            "Entity_QID":    r["linking"]["primary"]["qid"] if r["linking"]["primary"] else "",
            "Entity_Label":  r["linking"]["primary"]["label"] if r["linking"]["primary"] else "",
            "Template":      r.get("template", ""),
            "Success":       r.get("success", False),
            "N_Results":     r.get("n_results", 0),
            "Pred_Answer":   str(r.get("answer", "")),
            "Gold_Answer":   str(row.get("Gold_Answer", "")),
        })
        time.sleep(0.3)

    results_df  = pd.DataFrame(results)
    detail_path = os.path.join(output_dir, "answer_evaluation.tsv")
    results_df.to_csv(detail_path, sep="\t", index=False)

    n           = len(results)
    qt_matches  = sum(bool(r["QType_Match"]) for r in results if r["QType_Match"] is not None)
    qt_total    = sum(1 for r in results if r["QType_Match"] is not None)
    n_success   = sum(r["Success"] for r in results)

    print(f"\n{'='*60}")
    print(f"  NEURALGEOQA EVALUATION")
    print(f"{'='*60}")
    print(f"  Total:              {n}")
    print(f"  QType accuracy:     {qt_matches}/{qt_total} ({100*qt_matches/max(1,qt_total):.1f}%)")
    print(f"  Pipeline success:   {n_success}/{n} ({100*n_success/n:.1f}%)")

    metrics = {
        "total": n,
        "qtype_accuracy": round(qt_matches / max(1, qt_total), 4),
        "pipeline_success_rate": round(n_success / n, 4),
    }
    with open(os.path.join(output_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nSaved: {detail_path}")
    save_cache()


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="NeuralGeoQA evaluation pipeline")
    parser.add_argument("--model_dir",  required=True, help="Path to qtype_model_v2 directory")
    parser.add_argument("--test_file",  required=True, help="TSV test file with Question/QType/Gold_Answer")
    parser.add_argument("--output_dir", required=True, help="Directory for evaluation outputs")
    parser.add_argument("--cache_dir",  default="cache", help="Wikidata cache directory")
    parser.add_argument("--question",   default="", help="Run pipeline on a single question instead")
    args = parser.parse_args()

    init_cache(args.cache_dir)
    load_model(args.model_dir)

    if args.question:
        ask(args.question)
    else:
        run_evaluation(args.test_file, args.output_dir)
