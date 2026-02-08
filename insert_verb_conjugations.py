#!/usr/bin/env python3
"""
One-time script to bulk insert all Japanese verb conjugation rows
into the database.

Note: This is no longer required for new users — the web app auto-imports
verb conjugations on first visit to /conjugations.  This script remains
available for manual use or re-importing.
"""

from llm_learn_japanese import db


def main() -> None:
    count = db.import_default_conjugations()
    if count == 0:
        print("ℹ️  All conjugations already imported (0 new)")
    else:
        print(f"✅ Inserted {count} conjugations.")

    # Show total
    session = db.get_session()
    total = session.query(db.VerbConjugation).count()
    session.close()
    print(f"📊 Total verb conjugations in database: {total}")


if __name__ == "__main__":
    main()
