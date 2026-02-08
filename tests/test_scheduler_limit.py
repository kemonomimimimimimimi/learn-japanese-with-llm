import datetime
import pytest
from sqlalchemy import create_engine

from llm_learn_japanese import db
from llm_learn_japanese.db import get_session, CardAspect, UserSettings, init_db


@pytest.fixture(autouse=True)
def setup_db(tmp_path, monkeypatch):
    # Use a temporary SQLite DB, and rebind engine/session to it
    test_db = str(tmp_path / "test.db")
    monkeypatch.setenv("LLM_JP_DB", test_db)
    db.engine = create_engine(f"sqlite:///{test_db}")
    db.SessionLocal = db.sessionmaker(bind=db.engine)
    db.init_db()
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


def test_review_cards_prioritized(monkeypatch):
    session = get_session()
    # Make one "review" card by setting success_count > 0
    review_card = make_card(session, success_count=5)
    # Manually set its next_review to now (due)
    review_card.next_review = datetime.datetime.now(datetime.UTC)
    session.commit()

    # Make one new card (success_count=0)
    new_card = make_card(session, parent_id=2, success_count=0)
    new_card.next_review = datetime.datetime.now(datetime.UTC)
    session.commit()
    session.close()

    card = db.get_next_card("tester")
    assert card is not None
    # Should pick the review card because it's due and has success_count > 0
    session = get_session()
    same = session.get(CardAspect, card.id)
    session.close()
    assert same.success_count > 0


def test_new_cards_limited(monkeypatch):
    session = get_session()
    # Ensure user settings exist with max_new_per_day=1
    existing = session.query(UserSettings).filter_by(user="tester").first()
    if existing:
        existing.max_new_per_day = 1
    else:
        settings = UserSettings(user="tester", max_new_per_day=1)
        session.add(settings)
    session.commit()
    # Create two fresh new cards
    c1 = make_card(session, parent_id=1, success_count=0)
    c2 = make_card(session, parent_id=2, success_count=0)
    session.commit()
    session.close()

    card1 = db.get_next_card("tester")
    assert card1 is not None

    card2 = db.get_next_card("tester")

    # Quota should be enforced: no additional new cards introduced
    session = get_session()
    settings = session.query(UserSettings).filter_by(user="tester").first()
    assert settings.new_cards_today == 1
    session.close()

    # If there is a second card, it must not be a fresh new one
    if card2 is not None:
        assert card2.success_count != 0


def test_new_card_not_repeated_immediately(monkeypatch):
    session = get_session()
    now = datetime.datetime.now(datetime.UTC)
    # Simulate atomic aspects created for the same parent concept
    for aspect_type in ("meaning", "reading", "usage"):
        session.add(
            CardAspect(
                parent_type="vocabulary",
                parent_id=999,
                aspect_type=aspect_type,
                prompt_template=f"prompt for {aspect_type}",
                interval=1,
                ease_factor=2.5,
                next_review=now,
                success_count=0,
            )
        )
    session.commit()
    session.close()

    first_card = db.get_next_card("tester")
    assert first_card is not None

    second_card = db.get_next_card("tester")
    # Once a new concept has been introduced in a session, it should not
    # immediately reappear as the exact same exercise.
    assert second_card is None or second_card.id != first_card.id