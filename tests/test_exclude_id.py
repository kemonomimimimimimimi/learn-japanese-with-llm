import pytest
from llm_learn_japanese import db
from llm_learn_japanese.db import get_session, CardAspect, init_db
from sqlalchemy import create_engine
import datetime

@pytest.fixture(autouse=True)
def setup_db(tmp_path, monkeypatch):
    # Temporary SQLite DB
    test_db = str(tmp_path / "test_exclude.db")
    monkeypatch.setenv("LLM_JP_DB", test_db)
    db.engine = create_engine(f"sqlite:///{test_db}")
    db.SessionLocal = db.sessionmaker(bind=db.engine)
    init_db()
    yield

def make_card(session, parent_type="vocabulary", parent_id=1, success_count=0):
    card = CardAspect(
        parent_type=parent_type,
        parent_id=parent_id,
        aspect_type="meaning",
        prompt_template="dummy",
        interval=1,
        ease_factor=2.5,
        next_review=datetime.datetime.now(datetime.UTC),
        success_count=success_count,
    )
    session.add(card)
    session.commit()
    return card

def test_get_next_card_excludes_given_id():
    session = get_session()
    # Create two cards
    c1 = make_card(session, parent_id=1, success_count=5)
    c2 = make_card(session, parent_id=2, success_count=5)
    session.close()

    # First fetch normally
    card1 = db.get_next_card("tester")
    assert card1 is not None

    # Try fetching with exclude_id=card1.id
    card2 = db.get_next_card("tester", exclude_id=card1.id)
    assert card2 is not None
    assert card2.id != card1.id