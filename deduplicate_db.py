from llm_learn_japanese.db import get_session, Grammar, Phrase, Idiom, _check_semantic_duplicate_with_ai, _get_existing_items_for_ai_check, engine
from sqlalchemy import inspect, text

from typing import Any
from sqlalchemy.orm import Session

from llm_learn_japanese.db import DEBUG_MODE

def ensure_embeddings_columns() -> None:
    """Add missing embedding columns to existing DB tables."""
    insp = inspect(engine)
    with engine.connect() as conn:
        for table in ["vocabulary", "grammar", "kanji", "idioms", "phrases"]:
            cols = [c["name"] for c in insp.get_columns(table)]
            if "embedding" not in cols:
                conn.execute(text(f"ALTER TABLE {table} ADD COLUMN embedding BLOB"))

def deduplicate_table(model: Any, field_name: str, item_type: str) -> None:
    """Deduplicate rows in a given table by semantic duplicate detection."""
    ensure_embeddings_columns()
    from llm_learn_japanese.db import _get_embedding, _serialize_embedding, _deserialize_embedding
    session: Session = get_session()
    all_items = session.query(model).all()
    texts = [getattr(item, field_name) for item in all_items]

    if DEBUG_MODE:
        print(f"🔍 Starting deduplication for {item_type}, total items: {len(all_items)}")

    seen = set()
    to_delete = []

    for item in all_items:
        current_value = getattr(item, field_name)

        # Ensure embedding exists and is stored once
        if not getattr(item, "embedding", None):
            vec = _get_embedding(current_value)
            item.embedding = _serialize_embedding(vec)
            session.add(item)
            session.commit()
            if DEBUG_MODE:
                print(f"💾 Stored new embedding for {item_type}: '{current_value[:40]}'...")

        if DEBUG_MODE:
            print(f"➡️ Checking {item_type}: '{current_value}'")
        if current_value in seen:
            if DEBUG_MODE:
                print(f"   ⚠️ Already seen, marking duplicate: {current_value}")
            to_delete.append(item)
            continue

        # Check against already-seen items semantically
        duplicates = _get_existing_items_for_ai_check(session, item_type)
        match = _check_semantic_duplicate_with_ai(current_value, duplicates, item_type)
        if DEBUG_MODE:
            print(f"   AI check result for '{current_value}' -> {match}")
        if match and match in seen:
            if DEBUG_MODE:
                print(f"   ✅ Semantic duplicate confirmed against: {match}")
            to_delete.append(item)
        else:
            seen.add(current_value)

    # Delete duplicates
    for item in to_delete:
        val = getattr(item, field_name)
        print(f"Deleting duplicate {item_type}: {val}")
        session.delete(item)

    if DEBUG_MODE:
        print(f"🗑 Total {len(to_delete)} duplicates found for {item_type}")

    session.commit()
    session.close()


def main() -> None:
    print("Deduplicating Grammar...")
    deduplicate_table(Grammar, "point", "grammar")
    print("Deduplicating Phrase...")
    deduplicate_table(Phrase, "phrase", "phrase")
    print("Deduplicating Idiom...")
    deduplicate_table(Idiom, "idiom", "idiom")


if __name__ == "__main__":
    main()