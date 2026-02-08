#!/usr/bin/env python3
"""Import Top Words from CSV into the database.

Usage: python import_top_words.py [--csv path] [--max N]
"""
import sys, os, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from llm_learn_japanese import db

def main() -> None:
    parser = argparse.ArgumentParser(description="Import Top Words CSV")
    parser.add_argument("--csv", default="data/top_20000_words.csv")
    parser.add_argument("--max", type=int, default=10000, help="Max rows to import")
    args = parser.parse_args()

    if not os.path.exists(args.csv):
        print(f"❌ CSV not found: {args.csv}"); sys.exit(1)

    db.Base.metadata.create_all(bind=db.engine)
    count = db.import_top_words_csv(args.csv, max_rows=args.max)
    progress = db.get_top_word_progress()
    print(f"\n📊 Top Words: {progress['total']} total, {progress['learned']} learned, {progress['unseen']} unseen")
    if progress['next_word']:
        print(f"   Next: #{progress['next_rank']} 「{progress['next_word']}」")

if __name__ == "__main__":
    main()
