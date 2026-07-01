#!/usr/bin/env python3
"""Run privacy_filter PII inference over texts stored in a SQLite database.

Reads up to --limit rows from a table/column and prints each text followed by
the detected PII entities.

Example
-------
    python tests/classify_database.py \
        --model ~/Downloads/privacy-filter-multilingual-q8.gguf

Defaults target the Apple SQLite database used during development; override
--db/--table/--column for other sources. The model path may also be supplied
via the PF_TEST_MODEL environment variable.
"""

from __future__ import annotations

import argparse
import os
import sqlite3
import sys

from privacy_filter import PrivacyFilter

DEFAULT_DB = "/home/virostatiq/PycharmProjects/agent-hermes/db/Apple_Microsoft.sqlite"
DEFAULT_MODEL = os.path.expanduser("~/Downloads/privacy-filter-multilingual-q8.gguf")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--db", default=DEFAULT_DB, help="path to the SQLite database")
    p.add_argument("--table", default="classifications", help="table holding the texts")
    p.add_argument("--column", default="cleaned_text", help="text column to classify")
    p.add_argument("--model", default=os.environ.get("PF_TEST_MODEL", DEFAULT_MODEL),
                   help="path to the privacy-filter GGUF model")
    p.add_argument("--limit", type=int, default=200, help="max number of texts to process")
    p.add_argument("--threshold", type=float, default=0.5, help="minimum entity score to report")
    p.add_argument("--device", default="cpu", help="cpu | gpu | cuda | vulkan (optionally :N)")
    return p.parse_args()


def fetch_texts(db: str, table: str, column: str, limit: int) -> list[str]:
    conn = sqlite3.connect(db)
    try:
        rows = conn.execute(
            f"SELECT {column} FROM {table} "
            f"WHERE {column} IS NOT NULL AND TRIM({column}) <> '' "
            f"LIMIT ?",
            (limit,),
        ).fetchall()
    finally:
        conn.close()
    return [r[0] for r in rows]


def main() -> int:
    args = parse_args()
    if not os.path.exists(args.model):
        print(f"error: model not found: {args.model}\n"
              f"pass --model or set PF_TEST_MODEL", file=sys.stderr)
        return 2
    if not os.path.exists(args.db):
        print(f"error: database not found: {args.db}", file=sys.stderr)
        return 2

    texts = fetch_texts(args.db, args.table, args.column, args.limit)
    print(f"Loaded {len(texts)} texts from {args.table}.{args.column} in {os.path.basename(args.db)}")
    print(f"Model: {os.path.basename(args.model)}  device={args.device}  threshold={args.threshold}\n")

    total_entities = 0
    with PrivacyFilter(args.model, device=args.device) as pf:
        for i, text in enumerate(texts, 1):
            entities = pf.classify(text, threshold=args.threshold)
            total_entities += len(entities)

            print("=" * 80)
            print(f"[{i}/{len(texts)}] TEXT ({len(text)} chars):")
            print(text)
            print("-" * 80)
            if entities:
                print(f"INFERENCE ({len(entities)} entities):")
                for e in entities:
                    print(f"  {e.label:14} {e.score:.3f}  {e.text(text)!r}  bytes[{e.start}:{e.end}]")
            else:
                print("INFERENCE: (no PII detected above threshold)")
            print()

    print("=" * 80)
    print(f"Done: {total_entities} entities across {len(texts)} texts.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
