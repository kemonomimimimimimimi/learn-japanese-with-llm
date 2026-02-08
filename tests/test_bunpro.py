"""Tests for Bunpro Grammar feature."""
import os
import sys
import tempfile
import csv

# Set test mode before importing anything
os.environ["TEST_MODE"] = "1"

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.fixture(autouse=True)
def bunpro_test_db(tmp_path):
    """Create a fresh test database for each test."""
    db_path = str(tmp_path / "test_bunpro.db")
    os.environ["LLM_JP_DB"] = db_path

    # Force re-import to pick up new DB path
    import importlib
    from llm_learn_japanese import db
    importlib.reload(db)

    # Recreate engine with new path
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker
    db.engine = create_engine(f"sqlite:///{db_path}")
    db.SessionLocal = sessionmaker(bind=db.engine, expire_on_commit=False)
    db.Base.metadata.create_all(bind=db.engine)

    yield db

    # Cleanup
    if os.path.exists(db_path):
        os.unlink(db_path)


@pytest.fixture
def sample_csv(tmp_path):
    """Create a small sample CSV file for testing."""
    csv_path = str(tmp_path / "test_bunpro.csv")
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["rank", "usage_tier", "score", "register", "jlpt", "id", "grammar", "meaning", "url", "norm"])
        writer.writerow([1, "S (ubiquitous)", "110.0", "general", "N5", "1.0", "だ", "To be, Is", "https://bunpro.jp/grammar_points/da", "だ"])
        writer.writerow([2, "S (ubiquitous)", "110.0", "general", "N5", "3.0", "は", "As for... (Highlights sentence topic)", "https://bunpro.jp/grammar_points/ha", "は"])
        writer.writerow([3, "S (ubiquitous)", "110.0", "general", "N5", "4.0", "も", "Also, Too, As well", "https://bunpro.jp/grammar_points/mo", "も"])
        writer.writerow([4, "S (ubiquitous)", "110.0", "general", "N5", "12.0", "か", "Question marking particle", "https://bunpro.jp/grammar_points/ka", "か"])
        writer.writerow([5, "S (ubiquitous)", "110.0", "general", "N5", "13.0", "が", "Subject marking particle", "https://bunpro.jp/grammar_points/ga", "が"])
    return csv_path


# ── Import Tests ──────────────────────────────────────────────────

def test_import_csv(bunpro_test_db, sample_csv):
    """Test CSV import creates correct number of records."""
    count = bunpro_test_db.import_bunpro_csv(sample_csv)
    assert count == 5


def test_import_csv_idempotent(bunpro_test_db, sample_csv):
    """Test that re-importing the same CSV doesn't create duplicates."""
    count1 = bunpro_test_db.import_bunpro_csv(sample_csv)
    count2 = bunpro_test_db.import_bunpro_csv(sample_csv)
    assert count1 == 5
    assert count2 == 0  # All already imported


def test_import_csv_data_integrity(bunpro_test_db, sample_csv):
    """Test that imported data matches CSV content."""
    bunpro_test_db.import_bunpro_csv(sample_csv)
    session = bunpro_test_db.get_session()
    card = session.query(bunpro_test_db.BunproGrammar).filter_by(rank=1).first()
    session.close()

    assert card is not None
    assert card.grammar == "だ"
    assert card.meaning == "To be, Is"
    assert card.jlpt == "N5"
    assert card.usage_tier == "S (ubiquitous)"
    assert card.score == 110.0
    assert card.bunpro_id == 1.0
    assert card.success_count == 0
    assert card.lesson_viewed is False


# ── Card Retrieval Tests ──────────────────────────────────────────

def test_get_next_bunpro_card_returns_lowest_rank(bunpro_test_db, sample_csv):
    """Test that get_next_bunpro_card returns the lowest-ranked unseen card."""
    bunpro_test_db.import_bunpro_csv(sample_csv)
    card = bunpro_test_db.get_next_bunpro_card()
    assert card is not None
    assert card.rank == 1
    assert card.grammar == "だ"


def test_get_next_bunpro_card_advances_rank(bunpro_test_db, sample_csv):
    """Test that after reviewing card 1, card 2 is served next."""
    bunpro_test_db.import_bunpro_csv(sample_csv)

    # Get and review first card
    card1 = bunpro_test_db.get_next_bunpro_card()
    assert card1.rank == 1
    bunpro_test_db.review_bunpro_card(card1.id, 5)

    # Next card should be rank 2
    card2 = bunpro_test_db.get_next_bunpro_card()
    assert card2 is not None
    assert card2.rank == 2
    assert card2.grammar == "は"


def test_get_next_bunpro_card_empty_db(bunpro_test_db):
    """Test that get_next_bunpro_card returns None on empty DB."""
    card = bunpro_test_db.get_next_bunpro_card()
    assert card is None


def test_get_next_bunpro_card_daily_limit(bunpro_test_db, sample_csv):
    """Test that daily new card limit is enforced at answer time, not fetch time."""
    bunpro_test_db.import_bunpro_csv(sample_csv)

    # Set max new per day to 2
    session = bunpro_test_db.get_session()
    settings = session.query(bunpro_test_db.UserSettings).filter_by(user="default_user").first()
    if not settings:
        settings = bunpro_test_db.UserSettings(user="default_user", max_new_per_day=2)
        session.add(settings)
    else:
        settings.max_new_per_day = 2
    session.commit()
    session.close()

    # Get and answer 2 new cards (daily limit is enforced at answer time)
    card1 = bunpro_test_db.get_next_bunpro_card()
    assert card1 is not None
    bunpro_test_db.review_bunpro_card(card1.id, 5)  # answering increments new_cards_today

    card2 = bunpro_test_db.get_next_bunpro_card()
    assert card2 is not None
    assert card2.id != card1.id  # should be a different card
    bunpro_test_db.review_bunpro_card(card2.id, 5)  # new_cards_today = 2

    # Third new card should be None (limit reached after 2 answers)
    card3 = bunpro_test_db.get_next_bunpro_card()
    assert card3 is None


def test_get_next_bunpro_card_fetch_does_not_consume_quota(bunpro_test_db, sample_csv):
    """Test that merely fetching cards does NOT consume the daily new-card quota."""
    bunpro_test_db.import_bunpro_csv(sample_csv)

    # Set max new per day to 2
    session = bunpro_test_db.get_session()
    settings = session.query(bunpro_test_db.UserSettings).filter_by(user="default_user").first()
    if not settings:
        settings = bunpro_test_db.UserSettings(user="default_user", max_new_per_day=2)
        session.add(settings)
    else:
        settings.max_new_per_day = 2
    session.commit()
    session.close()

    # Fetch the same card 5 times without answering — quota should NOT be consumed
    for _ in range(5):
        card = bunpro_test_db.get_next_bunpro_card()
        assert card is not None
        assert card.rank == 1  # same unseen card every time (not marked)

    # Verify new_cards_today is still 0
    session = bunpro_test_db.get_session()
    settings = session.query(bunpro_test_db.UserSettings).filter_by(user="default_user").first()
    assert settings.new_cards_today == 0
    session.close()


def test_get_next_bunpro_card_exclude_ids(bunpro_test_db, sample_csv):
    """Test that exclude_ids prevents returning already-queued cards."""
    bunpro_test_db.import_bunpro_csv(sample_csv)

    card1 = bunpro_test_db.get_next_bunpro_card()
    assert card1 is not None
    assert card1.rank == 1

    # Fetch again excluding card1 — should get card2
    card2 = bunpro_test_db.get_next_bunpro_card(exclude_ids={card1.id})
    assert card2 is not None
    assert card2.rank == 2
    assert card2.id != card1.id


def test_get_next_bunpro_card_pending_new_count(bunpro_test_db, sample_csv):
    """Test that pending_new_count correctly limits new cards within a batch."""
    bunpro_test_db.import_bunpro_csv(sample_csv)

    # Set max new per day to 2
    session = bunpro_test_db.get_session()
    settings = session.query(bunpro_test_db.UserSettings).filter_by(user="default_user").first()
    if not settings:
        settings = bunpro_test_db.UserSettings(user="default_user", max_new_per_day=2)
        session.add(settings)
    else:
        settings.max_new_per_day = 2
    session.commit()
    session.close()

    # Simulate batch: fetch 2 new cards with pending_new_count tracking
    card1 = bunpro_test_db.get_next_bunpro_card(pending_new_count=0)
    assert card1 is not None

    card2 = bunpro_test_db.get_next_bunpro_card(exclude_ids={card1.id}, pending_new_count=1)
    assert card2 is not None

    # Third should be None: pending_new_count=2 >= max_new_per_day=2
    card3 = bunpro_test_db.get_next_bunpro_card(
        exclude_ids={card1.id, card2.id}, pending_new_count=2
    )
    assert card3 is None


def test_review_bunpro_card_increments_new_cards_today(bunpro_test_db, sample_csv):
    """Test that reviewing a new card increments new_cards_today."""
    bunpro_test_db.import_bunpro_csv(sample_csv)

    card = bunpro_test_db.get_next_bunpro_card()
    assert card is not None

    # Before reviewing: new_cards_today should be 0
    session = bunpro_test_db.get_session()
    settings = session.query(bunpro_test_db.UserSettings).filter_by(user="default_user").first()
    assert settings is None or settings.new_cards_today == 0
    session.close()

    # Review the card (first answer)
    bunpro_test_db.review_bunpro_card(card.id, 5)

    # After reviewing: new_cards_today should be 1
    session = bunpro_test_db.get_session()
    settings = session.query(bunpro_test_db.UserSettings).filter_by(user="default_user").first()
    assert settings.new_cards_today == 1
    session.close()


# ── Review Tests ──────────────────────────────────────────────────

def test_review_correct_increases_success(bunpro_test_db, sample_csv):
    """Test that a correct review increases success_count."""
    bunpro_test_db.import_bunpro_csv(sample_csv)
    card = bunpro_test_db.get_next_bunpro_card()
    initial_success = card.success_count

    bunpro_test_db.review_bunpro_card(card.id, 5)  # Perfect score

    session = bunpro_test_db.get_session()
    updated = session.get(bunpro_test_db.BunproGrammar, card.id)
    session.close()
    assert updated.success_count > initial_success


def test_review_wrong_keeps_in_pool(bunpro_test_db, sample_csv):
    """Test that a wrong answer keeps the card in the review pool."""
    bunpro_test_db.import_bunpro_csv(sample_csv)
    card = bunpro_test_db.get_next_bunpro_card()

    bunpro_test_db.review_bunpro_card(card.id, 1)  # Wrong answer

    session = bunpro_test_db.get_session()
    updated = session.get(bunpro_test_db.BunproGrammar, card.id)
    session.close()
    # Should stay >= 1 so it remains in review pool
    assert updated.success_count >= 1


def test_review_updates_scheduling(bunpro_test_db, sample_csv):
    """Test that review updates interval and next_review."""
    bunpro_test_db.import_bunpro_csv(sample_csv)
    card = bunpro_test_db.get_next_bunpro_card()

    bunpro_test_db.review_bunpro_card(card.id, 5)

    session = bunpro_test_db.get_session()
    updated = session.get(bunpro_test_db.BunproGrammar, card.id)
    session.close()

    assert updated.interval >= 1
    assert updated.repetitions >= 1


# ── Lesson Tracking Tests ─────────────────────────────────────────

def test_mark_lesson_viewed(bunpro_test_db, sample_csv):
    """Test marking a lesson as viewed."""
    bunpro_test_db.import_bunpro_csv(sample_csv)

    session = bunpro_test_db.get_session()
    card = session.query(bunpro_test_db.BunproGrammar).filter_by(rank=1).first()
    assert card.lesson_viewed is False
    card_id = card.id
    session.close()

    bunpro_test_db.mark_bunpro_lesson_viewed(card_id)

    session = bunpro_test_db.get_session()
    card = session.get(bunpro_test_db.BunproGrammar, card_id)
    assert card.lesson_viewed is True
    session.close()


# ── Progress Tests ────────────────────────────────────────────────

def test_progress_initial(bunpro_test_db, sample_csv):
    """Test initial progress stats."""
    bunpro_test_db.import_bunpro_csv(sample_csv)
    progress = bunpro_test_db.get_bunpro_progress()
    assert progress["total"] == 5
    assert progress["learned"] == 0
    assert progress["unseen"] == 5
    assert progress["due_now"] == 0
    assert progress["next_grammar"] == "だ"
    assert progress["next_rank"] == 1


def test_progress_after_learning(bunpro_test_db, sample_csv):
    """Test progress after studying some cards."""
    bunpro_test_db.import_bunpro_csv(sample_csv)

    # Study first card
    card = bunpro_test_db.get_next_bunpro_card()
    bunpro_test_db.review_bunpro_card(card.id, 5)

    progress = bunpro_test_db.get_bunpro_progress()
    assert progress["learned"] == 1
    assert progress["unseen"] == 4
    assert progress["next_grammar"] == "は"
    assert progress["next_rank"] == 2


# ── Distractor Tests ──────────────────────────────────────────────

def test_distractors_returns_correct_count(bunpro_test_db, sample_csv):
    """Test that get_bunpro_distractors returns the right number of distractors."""
    bunpro_test_db.import_bunpro_csv(sample_csv)

    session = bunpro_test_db.get_session()
    card = session.query(bunpro_test_db.BunproGrammar).filter_by(rank=1).first()
    session.close()

    distractors = bunpro_test_db.get_bunpro_distractors(card, count=3)
    assert len(distractors) >= 1  # At least some distractors
    assert len(distractors) <= 3  # No more than requested


def test_distractors_exclude_target(bunpro_test_db, sample_csv):
    """Test that distractors don't include the target card."""
    bunpro_test_db.import_bunpro_csv(sample_csv)

    session = bunpro_test_db.get_session()
    card = session.query(bunpro_test_db.BunproGrammar).filter_by(rank=1).first()
    session.close()

    distractors = bunpro_test_db.get_bunpro_distractors(card, count=3)
    for d in distractors:
        assert d["grammar"] != card.grammar or d["meaning"] != card.meaning


# ── Exercise Generation Tests ─────────────────────────────────────

def test_bunpro_fallback_question(bunpro_test_db, sample_csv):
    """Test the fallback question generator works."""
    from llm_learn_japanese.exercises import _bunpro_fallback_question

    bunpro_test_db.import_bunpro_csv(sample_csv)

    session = bunpro_test_db.get_session()
    card = session.query(bunpro_test_db.BunproGrammar).filter_by(rank=1).first()
    session.close()

    distractors = [
        {"grammar": "は", "meaning": "As for...", "jlpt": "N5"},
        {"grammar": "も", "meaning": "Also, Too", "jlpt": "N5"},
        {"grammar": "か", "meaning": "Question particle", "jlpt": "N5"},
    ]

    result = _bunpro_fallback_question(card, distractors)

    assert "question" in result
    assert "choices" in result
    assert "correct_key" in result
    assert "question_type" in result
    assert len(result["choices"]) == 4
    assert result["question_type"] == "meaning_match"

    # Verify correct answer is in choices
    correct = next(c for c in result["choices"] if c["key"] == result["correct_key"])
    assert correct["text"] == card.meaning


def test_pick_question_type_beginner():
    """Test question type selection for beginners."""
    from llm_learn_japanese.exercises import _pick_bunpro_question_type, BUNPRO_BEGINNER_TYPES

    # With low success count, should pick beginner types
    for _ in range(20):
        qtype = _pick_bunpro_question_type(0)
        assert qtype in BUNPRO_BEGINNER_TYPES


def test_pick_question_type_advanced():
    """Test question type selection for advanced learners."""
    from llm_learn_japanese.exercises import _pick_bunpro_question_type, BUNPRO_ADVANCED_TYPES

    for _ in range(20):
        qtype = _pick_bunpro_question_type(10)
        assert qtype in BUNPRO_ADVANCED_TYPES
