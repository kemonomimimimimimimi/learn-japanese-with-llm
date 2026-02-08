"""Tests for Top Kanji and Top Words features."""
import os
import sys
import tempfile
import csv

os.environ["TEST_MODE"] = "1"

import pytest
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.fixture(autouse=True)
def test_db(tmp_path):
    """Create fresh test database."""
    db_path = str(tmp_path / "test_kw.db")
    os.environ["LLM_JP_DB"] = db_path
    import importlib
    from llm_learn_japanese import db
    importlib.reload(db)
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker
    db.engine = create_engine(f"sqlite:///{db_path}")
    db.SessionLocal = sessionmaker(bind=db.engine, expire_on_commit=False)
    db.Base.metadata.create_all(bind=db.engine)
    yield db
    if os.path.exists(db_path):
        os.unlink(db_path)


@pytest.fixture
def kanji_csv(tmp_path):
    csv_path = str(tmp_path / "kanji.csv")
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["rank", "kanji", "count", "on_readings", "kun_readings", "nanori", "meanings_en"])
        writer.writerow([1, "人", "68196111", "", "", "", ""])
        writer.writerow([2, "言", "60173333", "", "", "", ""])
        writer.writerow([3, "見", "51447023", "", "", "", ""])
        writer.writerow([4, "一", "50614149", "", "", "", ""])
        writer.writerow([5, "出", "46902264", "", "", "", ""])
    return csv_path


@pytest.fixture
def words_csv(tmp_path):
    csv_path = str(tmp_path / "words.csv")
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["rank", "lemma", "reading", "count", "meanings_en", "pos_tags", "alt_readings_kana", "alt_spellings_kanji"])
        writer.writerow([1, "する", "シ", "233535711", "bad meaning", "verb", "する", ""])
        writer.writerow([2, "それ", "ソレ", "48376756", "it; that", "pronoun", "それ", ""])
        writer.writerow([3, "その", "ソノ", "42599453", "wrong meaning", "noun", "その", ""])
        writer.writerow([4, "言う", "イッ", "40966083", "to say", "verb", "いう", ""])
        writer.writerow([5, "いう", "イウ", "36175977", "to say", "verb", "いう", ""])
    return csv_path


# ── Kanji Import Tests ────────────────────────────────────────────

def test_kanji_import(test_db, kanji_csv):
    count = test_db.import_top_kanji_csv(kanji_csv)
    assert count == 5


def test_kanji_import_idempotent(test_db, kanji_csv):
    c1 = test_db.import_top_kanji_csv(kanji_csv)
    c2 = test_db.import_top_kanji_csv(kanji_csv)
    assert c1 == 5
    assert c2 == 0


def test_kanji_import_data(test_db, kanji_csv):
    test_db.import_top_kanji_csv(kanji_csv)
    session = test_db.get_session()
    card = session.query(test_db.TopKanji).filter_by(rank=1).first()
    session.close()
    assert card.kanji == "人"
    assert card.count == 68196111
    assert card.annotated is False
    assert card.meanings_en is None
    assert card.success_count == 0


# ── Kanji Card Retrieval Tests ────────────────────────────────────

def test_kanji_get_next_by_rank(test_db, kanji_csv):
    test_db.import_top_kanji_csv(kanji_csv)
    card = test_db.get_next_top_kanji()
    assert card is not None
    assert card.rank == 1
    assert card.kanji == "人"


def test_kanji_advances_rank(test_db, kanji_csv):
    test_db.import_top_kanji_csv(kanji_csv)
    c1 = test_db.get_next_top_kanji()
    test_db.review_top_kanji(c1.id, 5)
    c2 = test_db.get_next_top_kanji()
    assert c2.rank == 2
    assert c2.kanji == "言"


def test_kanji_empty_db(test_db):
    card = test_db.get_next_top_kanji()
    assert card is None


def test_kanji_fetch_does_not_consume_quota(test_db, kanji_csv):
    """Fetching cards does NOT consume the daily new-card quota."""
    test_db.import_top_kanji_csv(kanji_csv)

    # Fetch the same card 3 times without answering
    for _ in range(3):
        card = test_db.get_next_top_kanji()
        assert card is not None
        assert card.rank == 1

    # new_cards_today should still be 0
    session = test_db.get_session()
    settings = session.query(test_db.UserSettings).filter_by(user="default_user").first()
    assert settings is None or settings.new_cards_today == 0
    session.close()


def test_kanji_exclude_ids(test_db, kanji_csv):
    """exclude_ids prevents returning already-queued cards."""
    test_db.import_top_kanji_csv(kanji_csv)

    c1 = test_db.get_next_top_kanji()
    assert c1 is not None
    assert c1.rank == 1

    c2 = test_db.get_next_top_kanji(exclude_ids={c1.id})
    assert c2 is not None
    assert c2.rank == 2
    assert c2.id != c1.id


def test_kanji_review_increments_new_cards_today(test_db, kanji_csv):
    """Reviewing a new card increments new_cards_today."""
    test_db.import_top_kanji_csv(kanji_csv)

    card = test_db.get_next_top_kanji()
    assert card is not None

    test_db.review_top_kanji(card.id, 5)

    session = test_db.get_session()
    settings = session.query(test_db.UserSettings).filter_by(user="default_user").first()
    assert settings.new_cards_today == 1
    session.close()


# ── Kanji Annotation Tests ───────────────────────────────────────

def test_kanji_annotate(test_db, kanji_csv):
    test_db.import_top_kanji_csv(kanji_csv)
    session = test_db.get_session()
    card = session.query(test_db.TopKanji).filter_by(rank=1).first()
    card_id = card.id
    session.close()

    test_db.annotate_top_kanji(card_id, "ジン, ニン", "ひと", "person, people", "N5")

    session = test_db.get_session()
    card = session.get(test_db.TopKanji, card_id)
    session.close()
    assert card.annotated is True
    assert card.on_readings == "ジン, ニン"
    assert card.kun_readings == "ひと"
    assert card.meanings_en == "person, people"
    assert card.jlpt_level == "N5"


# ── Kanji Review Tests ───────────────────────────────────────────

def test_kanji_review_updates_schedule(test_db, kanji_csv):
    test_db.import_top_kanji_csv(kanji_csv)
    card = test_db.get_next_top_kanji()
    test_db.review_top_kanji(card.id, 5)
    session = test_db.get_session()
    updated = session.get(test_db.TopKanji, card.id)
    session.close()
    assert updated.success_count >= 1
    assert updated.repetitions >= 1


# ── Kanji Progress Tests ─────────────────────────────────────────

def test_kanji_progress(test_db, kanji_csv):
    test_db.import_top_kanji_csv(kanji_csv)
    progress = test_db.get_top_kanji_progress()
    assert progress["total"] == 5
    assert progress["unseen"] == 5
    assert progress["learned"] == 0
    assert progress["next_kanji"] == "人"


# ── Kanji Distractors Tests ──────────────────────────────────────

def test_kanji_distractors_need_annotation(test_db, kanji_csv):
    """Distractors only return annotated kanji."""
    test_db.import_top_kanji_csv(kanji_csv)
    session = test_db.get_session()
    card = session.query(test_db.TopKanji).filter_by(rank=1).first()
    session.close()
    distractors = test_db.get_kanji_distractors(card, count=3)
    assert len(distractors) == 0  # No annotated kanji yet


def test_kanji_distractors_with_annotated(test_db, kanji_csv):
    """After annotating some kanji, distractors are returned."""
    test_db.import_top_kanji_csv(kanji_csv)
    session = test_db.get_session()
    cards = session.query(test_db.TopKanji).all()
    session.close()
    # Annotate all except first
    for c in cards[1:]:
        test_db.annotate_top_kanji(c.id, "ON", "kun", f"meaning of {c.kanji}", "")
    distractors = test_db.get_kanji_distractors(cards[0], count=3)
    assert len(distractors) >= 1


# ── Words Import Tests ────────────────────────────────────────────

def test_words_import(test_db, words_csv):
    count = test_db.import_top_words_csv(words_csv)
    assert count == 5


def test_words_import_skeleton_only(test_db, words_csv):
    """Words import should store lemma and reading_csv but NOT dubious meanings."""
    test_db.import_top_words_csv(words_csv)
    session = test_db.get_session()
    card = session.query(test_db.TopWord).filter_by(rank=1).first()
    session.close()
    assert card.lemma == "する"
    assert card.reading_csv == "シ"
    assert card.annotated is False
    assert card.meanings_en is None  # Not imported from CSV (dubious)
    assert card.reading is None  # Will be populated by LLM


# ── Words Card Retrieval Tests ────────────────────────────────────

def test_words_get_next_by_rank(test_db, words_csv):
    test_db.import_top_words_csv(words_csv)
    card = test_db.get_next_top_word()
    assert card is not None
    assert card.rank == 1
    assert card.lemma == "する"


def test_words_advances_rank(test_db, words_csv):
    test_db.import_top_words_csv(words_csv)
    c1 = test_db.get_next_top_word()
    test_db.review_top_word(c1.id, 5)
    c2 = test_db.get_next_top_word()
    assert c2.rank == 2
    assert c2.lemma == "それ"


def test_words_fetch_does_not_consume_quota(test_db, words_csv):
    """Fetching word cards does NOT consume the daily new-card quota."""
    test_db.import_top_words_csv(words_csv)

    for _ in range(3):
        card = test_db.get_next_top_word()
        assert card is not None
        assert card.rank == 1

    session = test_db.get_session()
    settings = session.query(test_db.UserSettings).filter_by(user="default_user").first()
    assert settings is None or settings.new_cards_today == 0
    session.close()


def test_words_exclude_ids(test_db, words_csv):
    """exclude_ids prevents returning already-queued word cards."""
    test_db.import_top_words_csv(words_csv)

    c1 = test_db.get_next_top_word()
    assert c1 is not None
    assert c1.rank == 1

    c2 = test_db.get_next_top_word(exclude_ids={c1.id})
    assert c2 is not None
    assert c2.rank == 2
    assert c2.id != c1.id


def test_words_review_increments_new_cards_today(test_db, words_csv):
    """Reviewing a new word card increments new_cards_today."""
    test_db.import_top_words_csv(words_csv)

    card = test_db.get_next_top_word()
    assert card is not None

    test_db.review_top_word(card.id, 5)

    session = test_db.get_session()
    settings = session.query(test_db.UserSettings).filter_by(user="default_user").first()
    assert settings.new_cards_today == 1
    session.close()


# ── Words Annotation Tests ───────────────────────────────────────

def test_words_annotate(test_db, words_csv):
    test_db.import_top_words_csv(words_csv)
    session = test_db.get_session()
    card = session.query(test_db.TopWord).filter_by(rank=1).first()
    card_id = card.id
    session.close()

    test_db.annotate_top_word(card_id, "する", "to do, to make", "verb (irregular)", "N5")

    session = test_db.get_session()
    card = session.get(test_db.TopWord, card_id)
    session.close()
    assert card.annotated is True
    assert card.reading == "する"
    assert card.meanings_en == "to do, to make"
    assert card.pos_tags == "verb (irregular)"


# ── Words Review Tests ───────────────────────────────────────────

def test_words_review(test_db, words_csv):
    test_db.import_top_words_csv(words_csv)
    card = test_db.get_next_top_word()
    test_db.review_top_word(card.id, 5)
    session = test_db.get_session()
    updated = session.get(test_db.TopWord, card.id)
    session.close()
    assert updated.success_count >= 1


# ── Words Progress Tests ─────────────────────────────────────────

def test_words_progress(test_db, words_csv):
    test_db.import_top_words_csv(words_csv)
    progress = test_db.get_top_word_progress()
    assert progress["total"] == 5
    assert progress["unseen"] == 5
    assert progress["next_word"] == "する"


# ── Exercise Fallback Tests ──────────────────────────────────────

def test_kanji_fallback_question():
    from llm_learn_japanese.exercises import _kanji_fallback_question

    class FakeCard:
        kanji = "人"
        meanings_en = "person, people"
        success_count = 0

    distractors = [
        {"kanji": "山", "meanings_en": "mountain", "on_readings": "サン", "kun_readings": "やま"},
        {"kanji": "水", "meanings_en": "water", "on_readings": "スイ", "kun_readings": "みず"},
        {"kanji": "火", "meanings_en": "fire", "on_readings": "カ", "kun_readings": "ひ"},
    ]
    result = _kanji_fallback_question(FakeCard(), distractors)
    assert len(result["choices"]) == 4
    assert result["correct_key"] in ["A", "B", "C", "D"]
    correct = next(c for c in result["choices"] if c["key"] == result["correct_key"])
    assert correct["text"] == "person, people"


def test_word_fallback_question():
    from llm_learn_japanese.exercises import _word_fallback_question

    class FakeCard:
        lemma = "する"
        meanings_en = "to do, to make"
        success_count = 0

    distractors = [
        {"lemma": "いく", "reading": "いく", "meanings_en": "to go"},
        {"lemma": "くる", "reading": "くる", "meanings_en": "to come"},
        {"lemma": "みる", "reading": "みる", "meanings_en": "to see"},
    ]
    result = _word_fallback_question(FakeCard(), distractors)
    assert len(result["choices"]) == 4
    correct = next(c for c in result["choices"] if c["key"] == result["correct_key"])
    assert correct["text"] == "to do, to make"
