#!/usr/bin/env python3
"""
Import Bunpro grammar points from CSV into the database.

Usage:
    python import_bunpro.py
    python import_bunpro.py --csv data/bunpro_jlptplus_usage_ranked_heuristic.csv
"""

import sys
import os
import argparse

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from llm_learn_japanese import db


def main() -> None:
    parser = argparse.ArgumentParser(description="Import Bunpro grammar CSV into the database")
    parser.add_argument(
        "--csv",
        default="data/bunpro_jlptplus_usage_ranked_heuristic.csv",
        help="Path to the Bunpro CSV file (default: data/bunpro_jlptplus_usage_ranked_heuristic.csv)",
    )
    args = parser.parse_args()

    if not os.path.exists(args.csv):
        print(f"❌ CSV file not found: {args.csv}")
        sys.exit(1)

    # Ensure tables exist
    if not db.is_db_initialized():
        db.init_db()
        print("✅ Database initialized")
    else:
        # Create the bunpro_grammar table if missing
        db.Base.metadata.create_all(bind=db.engine)

    count = db.import_bunpro_csv(args.csv)
    if count == 0:
        print("ℹ️  All grammar points already imported (0 new)")
    else:
        print(f"🎉 Successfully imported {count} grammar points!")

    # Show stats
    progress = db.get_bunpro_progress()
    print(f"\n📊 Bunpro Grammar Stats:")
    print(f"   Total grammar points: {progress['total']}")
    print(f"   Learned: {progress['learned']}")
    print(f"   Unseen: {progress['unseen']}")
    if progress['next_grammar']:
        print(f"   Next up: #{progress['next_rank']} 「{progress['next_grammar']}」")


if __name__ == "__main__":
    main()
