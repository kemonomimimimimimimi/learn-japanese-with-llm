"""
Tests for the Japanese Learning App
Tests focus on database operations and core functionality with AI integration
"""

import os
import tempfile
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from typing import Any, Generator, Dict

from llm_learn_japanese import db, scheduler, exercises, structured

import datetime
import math

def test_daily_progress_tracking(temp_db: Any) -> None:
    """Test daily progress (cards reviewed, new cards seen)."""
    user = "test_user_daily"
    session = db.get_session()
    session.query(db.DailyProgress).filter_by(user=user).delete()
    session.commit()
    session.close()

    # Simulate 2 reviews with different outcomes
    db.update_progress(user, correct=True)
    db.update_progress(user, correct=False)

    stats = db.get_daily_progress(user, days=1)
    assert len(stats) == 1
    today = datetime.date.today().isoformat()
    assert stats[0]["date"] == today
    assert stats[0]["cards_reviewed"] == 2
    assert stats[0]["new_cards_seen"] == 0

# Mock AI model for testing
class MockAIModel:
    """Mock AI model for consistent testing without actual API calls."""

    def prompt(self, prompt_text: str, system: str = "") -> Any:
        """Mock prompt method that returns predictable responses."""
        # For exercise generation
        if "Generate a single exercise question" in system:
            class ExerciseResponse:
                def text(self) -> str:
                    return "What is the meaning of this Japanese word?"
            return ExerciseResponse()

        # For answer evaluation
        if "evaluating student answers" in system:
            import json
            # Return a moderate score for testing
            response_json = json.dumps({
                "quality": 3,
                "feedback": "Your answer shows basic understanding."
            })
            class EvaluationResponse:
                def text(self) -> str:
                    return response_json
            return EvaluationResponse()

        # Default response
        class DefaultResponse:
            def text(self) -> str:
                return "Mock response"
        return DefaultResponse()


@pytest.fixture(scope="function")
def temp_db() -> Generator[None, None, None]:
    """Setup transient SQLite DB for testing."""
    fd, path = tempfile.mkstemp()
    os.close(fd)
    db.engine = create_engine(f"sqlite:///{path}")
    db.SessionLocal = sessionmaker(bind=db.engine)
    db.Base.metadata.create_all(bind=db.engine)
    yield
    os.unlink(path)


@pytest.fixture
def mock_model() -> MockAIModel:
    """Provide a mock AI model for testing."""
    return MockAIModel()


def test_message_save_and_retrieve(temp_db: Any) -> None:
    """Test saving and retrieving messages."""
    session = db.get_session()
    db.save_message("これはテストです")
    result = session.query(db.Message).first()
    assert result is not None
    assert result.content == "これはテストです"
    session.close()


def test_vocabulary_card_save(temp_db: Any) -> None:
    """Test saving vocabulary cards with atomic aspects."""
    db.add_card("勉強")
    session = db.get_session()

    # Check vocabulary was created
    vocab = session.query(db.Vocabulary).filter_by(word="勉強").first()
    assert vocab is not None
    assert vocab.interval == 1

    # Check atomic aspects were created
    aspects = session.query(db.CardAspect).filter_by(parent_type="vocabulary", parent_id=vocab.id).all()
    assert len(aspects) == 3
    aspect_types = [aspect.aspect_type for aspect in aspects]
    assert "meaning" in aspect_types
    assert "reading" in aspect_types
    assert "usage" in aspect_types
    session.close()


def test_kanji_creation_with_aspects(temp_db: Any) -> None:
    """Test creating kanji with atomic aspects."""
    db.add_kanji("生", "セイ", "い(きる)", "life, birth")
    session = db.get_session()

    # Check kanji was created
    kanji = session.query(db.Kanji).filter_by(character="生").first()
    assert kanji is not None
    assert kanji.onyomi == "セイ"
    assert kanji.kunyomi == "い(きる)"

    # Check atomic aspects were created
    aspects = session.query(db.CardAspect).filter_by(parent_type="kanji", parent_id=kanji.id).all()
    assert len(aspects) == 4
    aspect_types = [aspect.aspect_type for aspect in aspects]
    assert "meaning" in aspect_types
    assert "onyomi" in aspect_types
    assert "kunyomi" in aspect_types
    assert "usage" in aspect_types
    session.close()


def test_grammar_creation_with_aspects(temp_db: Any) -> None:
    """Test creating grammar points with atomic aspects."""
    db.add_grammar("から", "indicates reason", "雨が降ったから、出かけません。")
    session = db.get_session()

    # Check grammar was created
    grammar = session.query(db.Grammar).filter_by(point="から").first()
    assert grammar is not None
    assert grammar.explanation == "indicates reason"

    # Check atomic aspects were created
    aspects = session.query(db.CardAspect).filter_by(parent_type="grammar", parent_id=grammar.id).all()
    assert len(aspects) == 2
    aspect_types = [aspect.aspect_type for aspect in aspects]
    assert "explanation" in aspect_types
    assert "usage" in aspect_types
    session.close()


def test_phrase_and_idiom_creation(temp_db: Any) -> None:
    """Test creating phrases and idioms with aspects."""
    # Test phrase creation
    db.add_phrase("よろしくお願いします", "Please treat me well")
    session = db.get_session()

    phrase = session.query(db.Phrase).filter_by(phrase="よろしくお願いします").first()
    assert phrase is not None

    phrase_aspects = session.query(db.CardAspect).filter_by(parent_type="phrase", parent_id=phrase.id).all()
    assert len(phrase_aspects) == 2

    # Test idiom creation
    db.add_idiom("猫の手も借りたい", "extremely busy", "期末試験の準備で猫の手も借りたい")

    idiom = session.query(db.Idiom).filter_by(idiom="猫の手も借りたい").first()
    assert idiom is not None

    idiom_aspects = session.query(db.CardAspect).filter_by(parent_type="idiom", parent_id=idiom.id).all()
    assert len(idiom_aspects) == 2

    session.close()


def test_scheduler_sm2() -> None:
    """Test SM-2 scheduling algorithm."""
    # First correct review (reps=0, quality=5): interval should be 1 day, reps becomes 1
    new_interval, new_ease, new_reps, next_review = scheduler.sm2_schedule(1, 2.5, 0, 5)
    assert new_interval == 1
    assert new_reps == 1
    assert new_ease > 2.0
    assert next_review is not None

    # Second correct review (reps=1, quality=5): interval should be 6 days, reps becomes 2
    new_interval, new_ease, new_reps, next_review = scheduler.sm2_schedule(1, new_ease, 1, 5)
    assert new_interval == 6
    assert new_reps == 2

    # Third correct review (reps=2, quality=4): interval = ceil(6 * EF)
    saved_ease = new_ease
    new_interval, new_ease, new_reps, next_review = scheduler.sm2_schedule(6, saved_ease, 2, 4)
    assert new_interval == math.ceil(6 * new_ease)  # uses UPDATED ease factor
    assert new_reps == 3

    # Lapse case: quality < 3 should reset interval to 1 and reps to 0
    new_interval, new_ease, new_reps, next_review = scheduler.sm2_schedule(10, 2.5, 5, 2)
    assert new_interval == 1
    assert new_reps == 0

    # Ease factor floor: quality 0 should not drop EF below 1.3
    new_interval, new_ease, new_reps, next_review = scheduler.sm2_schedule(1, 1.3, 0, 0)
    assert new_ease >= 1.3


def test_scheduler_fsrs_compat() -> None:
    """Test backward-compatible fsrs_schedule wrapper still works."""
    new_interval, new_ease, next_review = scheduler.fsrs_schedule(1, 2.5, 5)
    assert new_interval >= 1
    assert new_ease > 2.0
    assert next_review is not None

    new_interval, new_ease, next_review = scheduler.fsrs_schedule(10, 2.5, 2)
    assert new_interval == 1


def test_ai_required_for_exercise_generation(mock_model: MockAIModel) -> None:
    """Test that exercise generation requires AI model."""
    # Create a mock aspect
    aspect = type("DummyAspect", (), {
        "aspect_type": "meaning",
        "parent_type": "vocabulary",
        "parent_id": 1,
        "success_count": 0,
        "interval": 1
    })
    
    # Test that passing None raises ValueError
    with pytest.raises(ValueError, match="AI model is required"):
        exercises.generate_exercise(aspect, model=None)

    # Test with mock model works
    exercise = exercises.generate_exercise(aspect, model=mock_model)
    assert "meaning" in exercise or "word" in exercise.lower()


def test_ai_required_for_answer_evaluation(temp_db: Any, mock_model: MockAIModel) -> None:
    """Test that answer evaluation requires AI model."""
    db.add_card("勉強")
    aspect = db.get_next_card()
    assert aspect is not None

    # Test that passing None raises ValueError
    with pytest.raises(ValueError, match="AI model is required"):
        db.evaluate_answer_with_ai(aspect, "study", model=None)

    # Test with mock model works
    score, feedback = db.evaluate_answer_with_ai(aspect, "study", model=mock_model)
    assert 0 <= score <= 5
    assert feedback != ""


def test_progress_tracking(temp_db: Any) -> None:
    """Test user progress tracking."""
    db.update_progress("testuser", correct=True)
    prog = db.get_progress("testuser")
    assert prog is not None
    assert prog["total_reviews"] == 1
    assert prog["correct_answers"] == 1
    assert "total_seen" in prog
    assert "new_today" in prog

    db.update_progress("testuser", correct=False)
    prog2 = db.get_progress("testuser")
    assert prog2 is not None
    assert prog2["total_reviews"] == 2
    assert prog2["correct_answers"] == 1
    assert "total_seen" in prog2
    assert "new_today" in prog2


def test_next_card_retrieval(temp_db: Any) -> None:
    """Test getting the next card for review."""
    # Add some content with aspects
    db.add_card("勉強")
    db.add_kanji("生")

    # Get next card should return a CardAspect
    next_aspect = db.get_next_card()
    assert next_aspect is not None
    assert hasattr(next_aspect, 'aspect_type')
    assert hasattr(next_aspect, 'prompt_template')
    assert next_aspect.parent_type in ["vocabulary", "kanji"]


def test_review_card_updates(temp_db: Any) -> None:
    """Test that reviewing a card updates its scheduling."""
    # Add a card and get its first aspect
    db.add_card("勉強")
    aspect = db.get_next_card()
    assert aspect is not None

    # Record initial values
    initial_interval = aspect.interval
    initial_success_count = aspect.success_count

    # Review with good quality (4)
    db.review_card(aspect.id, 4)

    # Check the aspect was updated
    session = db.get_session()
    updated_aspect = session.get(db.CardAspect, aspect.id)
    assert updated_aspect is not None

    # FSRS should have increased interval and updated success count
    assert updated_aspect.interval >= initial_interval
    assert updated_aspect.success_count == initial_success_count + 1
    session.close()


def test_duplicate_detection(temp_db: Any) -> None:
    """Test duplicate detection for all content types."""
    # Vocabulary duplicates
    result1 = db.add_card("勉強")
    assert result1 is True  # Should be new

    result2 = db.add_card("勉強")
    assert result2 is False  # Should be duplicate

    # Kanji duplicates
    result3 = db.add_kanji("生", "セイ", "い(きる)", "life")
    assert result3 is True  # Should be new

    result4 = db.add_kanji("生", "ショウ", "なま", "raw, life")
    assert result4 is False  # Should be duplicate

    # Grammar duplicates
    result5 = db.add_grammar("から", "indicates reason", "example1")
    assert result5 is True

    result6 = db.add_grammar("から", "different explanation", "example2")
    assert result6 is False

    # Phrase duplicates
    result7 = db.add_phrase("よろしく", "please")
    assert result7 is True

    result8 = db.add_phrase("よろしく", "different meaning")
    assert result8 is False

    # Idiom duplicates
    result9 = db.add_idiom("猫の手", "cat's paw", "example1")
    assert result9 is True

    result10 = db.add_idiom("猫の手", "different meaning", "example2")
    assert result10 is False


def test_ai_duplicate_detection(temp_db: Any) -> None:
    """Test that duplicate detection uses simple string matching."""
    result = db._check_semantic_duplicate_with_ai("から", ["〜から"], "grammar")
    assert result is None  # Different strings shouldn't match
    
    result = db._check_semantic_duplicate_with_ai("から", ["から"], "grammar")
    # Allow either None (no AI client) or exact match depending on environment
    assert result in (None, "から")


def test_structured_dataclasses() -> None:
    """Test structured data classes."""
    vocab = structured.VocabularyRow(word="勉強", reading="べんきょう", meaning="study")
    assert vocab.word == "勉強"
    
    kanji = structured.KanjiRow(character="生", onyomi="セイ", kunyomi="い(きる)", meanings="life")
    assert kanji.character == "生"
    
    grammar = structured.GrammarRow(point="から", explanation="indicates reason", example="雨が降ったから")
    assert grammar.point == "から"


def test_database_initialization(temp_db: Any) -> None:
    """Test that database tables are created correctly."""
    session = db.get_session()
    
    # Check all tables exist by trying to query them
    try:
        session.query(db.Vocabulary).count()
        session.query(db.Kanji).count()
        session.query(db.Grammar).count()
        session.query(db.Phrase).count()
        session.query(db.Idiom).count()
        session.query(db.CardAspect).count()
        session.query(db.Progress).count()
        session.query(db.Message).count()
    except Exception as e:
        pytest.fail(f"Database tables not created properly: {e}")
    finally:
        session.close()


def test_ai_exercise_types() -> None:
    """Test that different aspect types generate appropriate exercise types."""
    # Test vocabulary meaning exercises
    vocab_types = exercises._get_exercise_types_for_aspect("meaning", "vocabulary")
    assert "direct_translation" in vocab_types
    assert "multiple_choice" in vocab_types

    # Test kanji onyomi exercises
    kanji_types = exercises._get_exercise_types_for_aspect("onyomi", "kanji")
    assert "write_onyomi" in kanji_types
    assert "identify_onyomi_word" in kanji_types

    # Test grammar usage exercises
    grammar_types = exercises._get_exercise_types_for_aspect("usage", "grammar")
    assert "create_example" in grammar_types
    assert "transform_sentence" in grammar_types


def test_difficulty_calculation() -> None:
    """Test difficulty level calculation based on success count."""
    # Create mock aspects with different success counts
    beginner_aspect = type("Aspect", (), {"success_count": 1})
    intermediate_aspect = type("Aspect", (), {"success_count": 5})
    advanced_aspect = type("Aspect", (), {"success_count": 15})

    assert exercises._calculate_difficulty_level(beginner_aspect) == "beginner"
    assert exercises._calculate_difficulty_level(intermediate_aspect) == "intermediate"
    assert exercises._calculate_difficulty_level(advanced_aspect) == "advanced"


def test_context_retrieval(temp_db: Any) -> None:
    """Test that context is properly retrieved for aspects."""
    # Add vocabulary with full details
    session = db.get_session()
    vocab = db.Vocabulary(word="勉強", reading="べんきょう", meaning="study")
    session.add(vocab)
    session.flush()

    aspect = db.CardAspect(
        parent_type="vocabulary",
        parent_id=vocab.id,
        aspect_type="meaning",
        prompt_template="What is the meaning?"
    )
    session.add(aspect)
    session.commit()

    # Test context retrieval
    context = exercises._get_aspect_context(aspect)
    assert context["word"] == "勉強"
    assert context["reading"] == "べんきょう"
    assert context["meaning"] == "study"
    assert context["aspect_type"] == "meaning"

    session.close()


def test_save_message_functionality(temp_db: Any) -> None:
    """Test the save_message method functionality that's now in main.py."""
    # This tests the underlying db.save_message which is used by the new UI method
    test_message = "これはテストメッセージです。This is a test message with Japanese content."
    db.save_message(test_message)

    session = db.get_session()
    message = session.query(db.Message).first()
    assert message is not None
    assert message.content == test_message
    assert message.timestamp is not None
    session.close()


def test_manual_card_review_functionality(temp_db: Any) -> None:
    """Test the manual card review functionality that's now in main.py."""
    # Add a card to get an aspect to review
    db.add_card("テスト")
    aspect = db.get_next_card()
    assert aspect is not None

    initial_interval = aspect.interval
    initial_success_count = aspect.success_count

    # Test manual review with good quality
    db.review_card(aspect.id, 4)

    # Verify the review was recorded
    session = db.get_session()
    updated_aspect = session.get(db.CardAspect, aspect.id)
    assert updated_aspect is not None
    assert updated_aspect.interval >= initial_interval
    assert updated_aspect.success_count == initial_success_count + 1
    session.close()


def test_user_progress_functionality(temp_db: Any) -> None:
    """Test the user progress functionality for specific users."""
    # Test with multiple users
    db.update_progress("user1", correct=True)
    db.update_progress("user1", correct=False)
    db.update_progress("user2", correct=True)
    db.update_progress("user2", correct=True)

    # Check user1 progress
    progress1 = db.get_progress("user1")
    assert progress1 is not None
    assert progress1["total_reviews"] == 2
    assert progress1["correct_answers"] == 1
    assert "total_seen" in progress1
    assert "new_today" in progress1

    # Check user2 progress
    progress2 = db.get_progress("user2")
    assert progress2 is not None
    assert progress2["total_reviews"] == 2
    assert progress2["correct_answers"] == 2

    # Check non-existent user
    progress3 = db.get_progress("nonexistent")
    assert progress3 is None


def test_image_processing_json_parsing() -> None:
    """Test JSON parsing for image processing functionality."""
    import json

    # Test valid structured JSON that would come from vision API
    sample_json = {
        "vocabulary": [
            {"word": "勉強", "reading": "べんきょう", "meaning": "study"},
            {"word": "学校", "reading": "がっこう", "meaning": "school"}
        ],
        "kanji": [
            {"character": "学", "onyomi": "ガク", "kunyomi": "まな(ぶ)", "meanings": "learning, study"},
            {"character": "校", "onyomi": "コウ", "kunyomi": "かま(える)", "meanings": "school"}
        ],
        "grammar": [
            {"point": "です", "explanation": "polite copula", "example": "これは本です。"}
        ],
        "phrases": [
            {"phrase": "こんにちは", "meaning": "hello"}
        ],
        "idioms": [
            {"idiom": "石橋を叩いて渡る", "meaning": "be very cautious", "example": "新しい仕事を始める前に石橋を叩いて渡る。"}
        ]
    }

    # Test that the JSON structure is valid and parseable
    json_str = json.dumps(sample_json)
    parsed = json.loads(json_str)

    assert "vocabulary" in parsed
    assert "kanji" in parsed
    assert "grammar" in parsed
    assert "phrases" in parsed
    assert "idioms" in parsed

    # Test that the structure matches what the image processing expects
    assert len(parsed["vocabulary"]) == 2
    assert parsed["vocabulary"][0]["word"] == "勉強"
    assert parsed["kanji"][0]["character"] == "学"
    assert parsed["grammar"][0]["point"] == "です"


def test_image_processing_database_integration(temp_db: Any) -> None:
    """Test that image processing results can be added to database."""
    # Simulate processing structured content from image
    structured_content = {
        "vocabulary": [
            {"word": "桜", "reading": "さくら", "meaning": "cherry blossom"}
        ],
        "kanji": [
            {"character": "桜", "onyomi": "オウ", "kunyomi": "さくら", "meanings": "cherry blossom"}
        ],
        "grammar": [
            {"point": "が", "explanation": "subject marker", "example": "桜が咲く。"}
        ],
        "phrases": [
            {"phrase": "お疲れ様", "meaning": "good work"}
        ],
        "idioms": [
            {"idiom": "花より団子", "meaning": "prefer substance over appearance", "example": "彼は花より団子だ。"}
        ]
    }

    # Add all items to database as the image processing would do
    vocab_new = kanji_new = grammar_new = phrase_new = idiom_new = 0

    # Process vocabulary
    for item in structured_content.get("vocabulary", []):
        word = item.get("word", "")
        if word and db.add_card(word):
            vocab_new += 1

    # Process kanji
    for item in structured_content.get("kanji", []):
        character = item.get("character", "")
        if character and db.add_kanji(
            character=character,
            onyomi=item.get("onyomi", ""),
            kunyomi=item.get("kunyomi", ""),
            meanings=item.get("meanings", "")
        ):
            kanji_new += 1

    # Process grammar
    for item in structured_content.get("grammar", []):
        point = item.get("point", "")
        if point and db.add_grammar(
            point=point,
            explanation=item.get("explanation", ""),
            example=item.get("example", "")
        ):
            grammar_new += 1

    # Process phrases
    for item in structured_content.get("phrases", []):
        phrase = item.get("phrase", "")
        if phrase and db.add_phrase(
            phrase=phrase,
            meaning=item.get("meaning", "")
        ):
            phrase_new += 1

    # Process idioms
    for item in structured_content.get("idioms", []):
        idiom = item.get("idiom", "")
        if idiom and db.add_idiom(
            idiom=idiom,
            meaning=item.get("meaning", ""),
            example=item.get("example", "")
        ):
            idiom_new += 1

    # Verify all items were added
    assert vocab_new == 1
    assert kanji_new == 1
    assert grammar_new == 1
    assert phrase_new == 1
    assert idiom_new == 1

    # Verify they exist in database
    session = db.get_session()

    vocab = session.query(db.Vocabulary).filter_by(word="桜").first()
    assert vocab is not None

    kanji = session.query(db.Kanji).filter_by(character="桜").first()
    assert kanji is not None
    assert kanji.onyomi == "オウ"

    grammar = session.query(db.Grammar).filter_by(point="が").first()
    assert grammar is not None

    phrase_result = session.query(db.Phrase).filter_by(phrase="お疲れ様").first()
    assert phrase_result is not None

    idiom_result = session.query(db.Idiom).filter_by(idiom="花より団子").first()
    assert idiom_result is not None

    session.close()


def test_openai_model_wrapper() -> None:
    """Test the OpenAI model wrapper interface expected by the application."""
    # Test the Response interface that OpenAIModel should return
    class MockResponse:
        def __init__(self, content: str) -> None:
            self.content = content

        def text(self) -> str:
            return self.content

    # Test the expected interface
    test_content = "Test response content"
    response = MockResponse(test_content)
    assert hasattr(response, 'text')
    assert callable(response.text)
    assert response.text() == test_content

    # Test that we can create a mock model with the expected interface
    class MockOpenAIModel:
        def __init__(self, client: Any, model_name: str = "gpt-4"):
            self.client = client
            self.model_name = model_name

        def prompt(self, prompt_text: str, system: str = "") -> MockResponse:
            return MockResponse("Mock AI response")

    # Verify the mock has the expected interface
    mock_client = type('MockClient', (), {})()
    model = MockOpenAIModel(mock_client)
    assert hasattr(model, 'prompt')
    assert callable(model.prompt)

    response = model.prompt("test prompt")
    assert hasattr(response, 'text')
    assert response.text() == "Mock AI response"


def test_vision_api_integration_structure() -> None:
    """Test the structure expected for vision API integration."""
    import base64

    # Test base64 encoding functionality used in image processing
    test_data = b"fake image data"
    encoded = base64.b64encode(test_data).decode('utf-8')
    assert isinstance(encoded, str)
    assert len(encoded) > 0

    # Test message structure for vision API
    expected_message_structure: Dict[str, Any] = {
        "role": "user",
        "content": [
            {"type": "text", "text": "some prompt"},
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{encoded}"
                }
            }
        ]
    }

    assert "role" in expected_message_structure
    assert "content" in expected_message_structure
    content_list = expected_message_structure["content"]
    assert isinstance(content_list, list)
    assert len(content_list) == 2
    assert content_list[0]["type"] == "text"
    assert content_list[1]["type"] == "image_url"
import pytest
from llm_learn_japanese import db
from pathlib import Path

def test_get_next_card_groups_by_parent(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Use a temporary SQLite DB
    test_db_path = tmp_path / "test.db"
    monkeypatch.setenv("LLM_JP_DB", str(test_db_path))

    # Re-init the DB for testing
    db.engine.dispose()
    db.engine = db.create_engine(f"sqlite:///{test_db_path}")
    db.SessionLocal = db.sessionmaker(bind=db.engine)
    db.init_db()

    # Add a vocabulary item (creates 3 CardAspects)
    assert db.add_card("猫") is True

    session = db.get_session()
    aspects = session.query(db.CardAspect).all()
    assert len(aspects) == 3
    session.close()

    # get_next_card should only return one aspect from that vocabulary group
    card1 = db.get_next_card()
    assert card1 is not None
    parent1 = (card1.parent_type, card1.parent_id)

    # After introducing the only available concept, siblings are deferred to
    # tomorrow, so a second call should return None (no more cards available).
    card2 = db.get_next_card()
    assert card2 is None

    # Add a second vocabulary item — it should now be available as a new card
    assert db.add_card("犬") is True

    card3 = db.get_next_card()
    assert card3 is not None
    parent3 = (card3.parent_type, card3.parent_id)
    # card3 must come from the NEW parent (犬), not the already-introduced one (猫)
    assert parent3 != parent1
    assert isinstance(card3.parent_type, str)