#!/usr/bin/env python3
"""Import Top Kanji from CSV into the database.

Usage: python import_top_kanji.py [--csv path] [--max N]
"""
import sys, os, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from llm_learn_japanese import db

def main() -> None:
    parser = argparse.ArgumentParser(description="Import Top Kanji CSV")
    parser.add_argument("--csv", default="data/top_10000_kanji.csv")
    parser.add_argument("--max", type=int, default=10433, help="Max rows to import")
    args = parser.parse_args()

    if not os.path.exists(args.csv):
        print(f"❌ CSV not found: {args.csv}"); sys.exit(1)

    db.Base.metadata.create_all(bind=db.engine)
    count = db.import_top_kanji_csv(args.csv, max_rows=args.max)
    progress = db.get_top_kanji_progress()
    print(f"\n📊 Top Kanji: {progress['total']} total, {progress['learned']} learned, {progress['unseen']} unseen")
    if progress['next_kanji']:
        print(f"   Next: #{progress['next_rank']} 「{progress['next_kanji']}」")

if __name__ == "__main__":
    main()
