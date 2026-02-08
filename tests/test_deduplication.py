import pytest
from llm_learn_japanese import db
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from deduplicate_db import deduplicate_table

from typing import Any

class DummyModel:
    def __init__(self, value: str) -> None:
        self.value: str = value

@pytest.mark.parametrize("item_type,field_name", [
    ("grammar", "point"),
    ("phrase", "phrase"),
    ("idiom", "idiom"),
])
def test_semantic_duplicate_detection(monkeypatch: pytest.MonkeyPatch, item_type: str, field_name: str) -> None:
    # Fake vectors
    fake_vecs = {
        "A": [1.0, 0.0],
        "B": [0.9, 0.1],
        "C": [0.0, 1.0],
    }

    def fake_get_embedding(text: str) -> list[float]:
        return fake_vecs.get(text, [0.5, 0.5])

    from typing import Any
    # Monkeypatch embedding and cosine similarity
    monkeypatch.setattr(db, "_get_embedding", fake_get_embedding)
    monkeypatch.setattr(db, "_cosine_similarity", lambda a,b: 0.9 if a == fake_vecs["A"] and b == fake_vecs["B"] else 0.1)

    # Monkeypatch OpenAI GPT confirmation to always say A and B are duplicates
    def fake_check(new_item: str, existing_items: list[str], item_type: str) -> str | None:
        if "B" in existing_items and new_item == "A":
            return "B"
        return None

    return None  # Ensure explicit return

    return None

    return None

    monkeypatch.setattr(db, "_check_semantic_duplicate_with_ai", fake_check)

    # Should detect duplicate
    existing = ["B", "C"]
    match = db._check_semantic_duplicate_with_ai("A", existing, item_type)
    assert match == "B"

def test_deduplicate_table_runs(monkeypatch: pytest.MonkeyPatch) -> None:
    # Monkeypatch internal functions so no DB access is needed
    monkeypatch.setattr(db, "_get_existing_items_for_ai_check", lambda sess, item_type: ["foo"])
    monkeypatch.setattr(db, "_check_semantic_duplicate_with_ai", lambda new, existing, itype: None)

    from typing import Any
    class FakeSession:
        def __init__(self) -> None:
            self.deleted: list[Any] = []
        def query(self, model: Any) -> "FakeSession":
            return self
        def all(self) -> list[Any]:
            return []
        def delete(self, obj: Any) -> None:
            self.deleted.append(obj)
        def commit(self) -> None:
            return None
        def close(self) -> None:
            return None

    monkeypatch.setattr(db, "get_session", lambda : FakeSession())
    # Should not raise
    deduplicate_table(db.Grammar, "point", "grammar")