"""
main.py
-------
CLI entry point for the NeuralGeoQA pipeline.

Stages:
  preprocess  -- fetch Wikidata labels + detect subject spans (Flair/GloVe)
  train       -- fine-tune DeBERTa QA + relation classification models
  evaluate    -- run full pipeline on test set (GeoQABERT + SPARQL)
  ask         -- interactive single-question mode

Examples:
  python main.py preprocess --base_dir /data/geoqa
  python main.py train --base_dir /data/geoqa --epochs 3
  python main.py evaluate --model_dir /data/geoqa/qtype_model_v2
                          --test_file /data/geoqa/geo_test_201.tsv
                          --output_dir /data/geoqa/evaluation
  python main.py ask --model_dir /data/geoqa/qtype_model_v2
                     --question "Which rivers are in Wales?"
"""

import argparse
import sys


def cmd_preprocess(args):
    from preprocess import main as preprocess_main
    preprocess_main(base_dir=args.base_dir)


def cmd_train(args):
    from train import main as train_main
    train_main(
        base_dir=args.base_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        accumulation_steps=args.accumulation_steps,
        learning_rate=args.lr,
        max_length=args.max_length,
        model_name=args.model_name,
    )


def cmd_evaluate(args):
    import evaluate as ev
    ev.init_cache(args.cache_dir)
    ev.load_model(args.model_dir)
    ev.run_evaluation(args.test_file, args.output_dir)


def cmd_ask(args):
    import evaluate as ev
    ev.init_cache(args.cache_dir)
    ev.load_model(args.model_dir)
    ev.ask(args.question)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="neuralgeoqa",
        description="NeuralGeoQA — geographic question answering over Wikidata",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # ── preprocess ──────────────────────────────────────────────────────────
    pp = sub.add_parser("preprocess", help="Fetch Wikidata labels + detect spans")
    pp.add_argument("--base_dir", default=".", help="Project root (contains not_ans/)")

    # ── train ───────────────────────────────────────────────────────────────
    tr = sub.add_parser("train", help="Fine-tune DeBERTa QA + relation models")
    tr.add_argument("--base_dir",           default="/content/drive/MyDrive/ANS/triple_head/")
    tr.add_argument("--epochs",             type=int,   default=3)
    tr.add_argument("--batch_size",         type=int,   default=1)
    tr.add_argument("--accumulation_steps", type=int,   default=8)
    tr.add_argument("--lr",                 type=float, default=2e-5)
    tr.add_argument("--max_length",         type=int,   default=96)
    tr.add_argument("--model_name",         default="microsoft/deberta-v3-base")

    # ── evaluate ─────────────────────────────────────────────────────────────
    ev = sub.add_parser("evaluate", help="Run full pipeline on test set")
    ev.add_argument("--model_dir",  required=True, help="Path to qtype_model_v2 dir")
    ev.add_argument("--test_file",  required=True, help="TSV test file")
    ev.add_argument("--output_dir", required=True, help="Where to write results")
    ev.add_argument("--cache_dir",  default="cache")

    # ── ask ──────────────────────────────────────────────────────────────────
    ak = sub.add_parser("ask", help="Answer a single question interactively")
    ak.add_argument("--model_dir", required=True)
    ak.add_argument("--question",  required=True)
    ak.add_argument("--cache_dir", default="cache")

    return parser


def main():
    parser  = build_parser()
    args    = parser.parse_args()
    dispatch = {
        "preprocess": cmd_preprocess,
        "train":      cmd_train,
        "evaluate":   cmd_evaluate,
        "ask":        cmd_ask,
    }
    dispatch[args.command](args)


if __name__ == "__main__":
    main()
