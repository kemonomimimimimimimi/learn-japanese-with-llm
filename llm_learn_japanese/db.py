from __future__ import annotations
from sqlalchemy import create_engine, Column, Date, Integer, Float as SAFloat, String, DateTime, Text, LargeBinary, Boolean
from sqlalchemy.orm import DeclarativeBase, sessionmaker, Session, Mapped, mapped_column
import datetime
import os
import os
from typing import Optional, List, Any, Dict, TYPE_CHECKING, cast
if TYPE_CHECKING:
    import numpy as np
from numpy.typing import NDArray

DEBUG_MODE = os.getenv("DEBUG", "0") == "1"


class Base(DeclarativeBase):
    pass
DB_PATH: str = os.environ.get("LLM_JP_DB", "japanese_learning.db")
engine = create_engine(f"sqlite:///{DB_PATH}")
# Prevent attribute expiration on commit so returned objects remain accessible
SessionLocal = sessionmaker(bind=engine, expire_on_commit=False)


class Message(Base):
    __tablename__ = "messages"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    timestamp: Mapped[datetime.datetime] = mapped_column(DateTime, default=lambda: datetime.datetime.now(datetime.UTC))


class Vocabulary(Base):
    __tablename__ = "vocabulary"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    word: Mapped[str] = mapped_column(String, nullable=False, unique=True)
    reading: Mapped[Optional[str]] = mapped_column(String)
    meaning: Mapped[Optional[str]] = mapped_column(Text)
    next_review: Mapped[datetime.datetime] = mapped_column(DateTime, default=lambda: datetime.datetime.now(datetime.UTC))
    interval: Mapped[int] = mapped_column(Integer, default=1)
    ease_factor: Mapped[float] = mapped_column(Integer, default=250)
    embedding: Mapped[Optional[bytes]] = mapped_column(LargeBinary)  # Store NumPy .tobytes()


class Grammar(Base):
    __tablename__ = "grammar"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    point: Mapped[str] = mapped_column(String, nullable=False, unique=True)
    explanation: Mapped[Optional[str]] = mapped_column(Text)
    example: Mapped[Optional[str]] = mapped_column(Text)
    next_review: Mapped[datetime.datetime] = mapped_column(DateTime, default=lambda: datetime.datetime.now(datetime.UTC))
    interval: Mapped[int] = mapped_column(Integer, default=1)
    ease_factor: Mapped[float] = mapped_column(Integer, default=250)
    embedding: Mapped[Optional[bytes]] = mapped_column(LargeBinary)


class Kanji(Base):
    __tablename__ = "kanji"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    character: Mapped[str] = mapped_column(String, nullable=False, unique=True)
    onyomi: Mapped[Optional[str]] = mapped_column(String)  # Chinese reading
    kunyomi: Mapped[Optional[str]] = mapped_column(String)  # Native Japanese reading
    meanings: Mapped[Optional[str]] = mapped_column(Text)  # Multiple possible meanings
    next_review: Mapped[datetime.datetime] = mapped_column(DateTime, default=lambda: datetime.datetime.now(datetime.UTC))
    interval: Mapped[int] = mapped_column(Integer, default=1)
    ease_factor: Mapped[float] = mapped_column(Integer, default=250)
    embedding: Mapped[Optional[bytes]] = mapped_column(LargeBinary)


class Idiom(Base):
    __tablename__ = "idioms"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    idiom: Mapped[str] = mapped_column(String, nullable=False, unique=True)
    meaning: Mapped[Optional[str]] = mapped_column(Text)
    example: Mapped[Optional[str]] = mapped_column(Text)
    next_review: Mapped[datetime.datetime] = mapped_column(DateTime, default=lambda: datetime.datetime.now(datetime.UTC))
    interval: Mapped[int] = mapped_column(Integer, default=1)
    ease_factor: Mapped[float] = mapped_column(Integer, default=250)
    embedding: Mapped[Optional[bytes]] = mapped_column(LargeBinary)


class VerbConjugation(Base):
    __tablename__ = "verb_conjugations"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    category: Mapped[str] = mapped_column(String, nullable=False)  # e.g. "Core verb forms", "Polite paradigm"
    label: Mapped[str] = mapped_column(String, nullable=False, unique=True)  # e.g. "Dictionary form"
    description: Mapped[Optional[str]] = mapped_column(Text)  # Explanation of usage
    example: Mapped[Optional[str]] = mapped_column(Text)  # Example sentence
    next_review: Mapped[datetime.datetime] = mapped_column(DateTime, default=lambda: datetime.datetime.now(datetime.UTC))
    interval: Mapped[int] = mapped_column(Integer, default=1)
    ease_factor: Mapped[float] = mapped_column(Integer, default=250)
    embedding: Mapped[Optional[bytes]] = mapped_column(LargeBinary)
    lesson_json: Mapped[Optional[str]] = mapped_column(Text, nullable=True)


class Phrase(Base):
    __tablename__ = "phrases"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    phrase: Mapped[str] = mapped_column(String, nullable=False, unique=True)
    meaning: Mapped[Optional[str]] = mapped_column(Text)
    next_review: Mapped[datetime.datetime] = mapped_column(DateTime, default=lambda: datetime.datetime.now(datetime.UTC))
    interval: Mapped[int] = mapped_column(Integer, default=1)
    ease_factor: Mapped[float] = mapped_column(Integer, default=250)
    embedding: Mapped[Optional[bytes]] = mapped_column(LargeBinary)
    lesson_json: Mapped[Optional[str]] = mapped_column(Text)


class CardAspect(Base):
    __tablename__ = "card_aspects"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    parent_type: Mapped[str] = mapped_column(String, nullable=False)  # e.g. "kanji", "vocabulary"
    parent_id: Mapped[int] = mapped_column(Integer, nullable=False)
    aspect_type: Mapped[str] = mapped_column(String, nullable=False)  # e.g. "onyomi", "kunyomi", "meaning"
    prompt_template: Mapped[str] = mapped_column(Text, nullable=False)
    next_review: Mapped[datetime.datetime] = mapped_column(DateTime, default=lambda: datetime.datetime.now(datetime.UTC))
    interval: Mapped[int] = mapped_column(Integer, default=1)
    ease_factor: Mapped[float] = mapped_column(Integer, default=250)
    repetitions: Mapped[int] = mapped_column(Integer, default=0)
    success_count: Mapped[int] = mapped_column(Integer, default=0)


class DailyProgress(Base):
    __tablename__ = "daily_progress"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user: Mapped[str] = mapped_column(String, nullable=False)
    date: Mapped[datetime.date] = mapped_column(Date, default=lambda: datetime.date.today())
    cards_reviewed: Mapped[int] = mapped_column(Integer, default=0)
    new_cards_seen: Mapped[int] = mapped_column(Integer, default=0)


class Progress(Base):
    __tablename__ = "progress"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user: Mapped[str] = mapped_column(String, nullable=False)
    total_reviews: Mapped[int] = mapped_column(Integer, default=0)
    correct_answers: Mapped[int] = mapped_column(Integer, default=0)
    last_updated: Mapped[datetime.datetime] = mapped_column(DateTime, default=lambda: datetime.datetime.now(datetime.UTC), onupdate=lambda: datetime.datetime.now(datetime.UTC))


class UserSettings(Base):
    __tablename__ = "user_settings"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user: Mapped[str] = mapped_column(String, nullable=False, unique=True)
    max_new_per_day: Mapped[int] = mapped_column(Integer, default=20)
    new_cards_today: Mapped[int] = mapped_column(Integer, default=0)
    last_reset: Mapped[datetime.datetime] = mapped_column(DateTime, default=lambda: datetime.datetime.now(datetime.UTC))


class BunproGrammar(Base):
    """Bunpro grammar points imported from CSV, with independent SRS scheduling."""
    __tablename__ = "bunpro_grammar"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    rank: Mapped[int] = mapped_column(Integer, nullable=False, unique=True)
    usage_tier: Mapped[str] = mapped_column(String, nullable=False)
    score: Mapped[Optional[float]] = mapped_column(SAFloat)
    register: Mapped[Optional[str]] = mapped_column(String)
    jlpt: Mapped[Optional[str]] = mapped_column(String)
    bunpro_id: Mapped[Optional[float]] = mapped_column(SAFloat)
    grammar: Mapped[str] = mapped_column(String, nullable=False)
    meaning: Mapped[Optional[str]] = mapped_column(Text)
    url: Mapped[Optional[str]] = mapped_column(String)
    norm: Mapped[Optional[str]] = mapped_column(String)
    # SRS fields
    next_review: Mapped[datetime.datetime] = mapped_column(DateTime, default=lambda: datetime.datetime.now(datetime.UTC))
    interval: Mapped[int] = mapped_column(Integer, default=1)
    ease_factor: Mapped[float] = mapped_column(SAFloat, default=2.5)
    repetitions: Mapped[int] = mapped_column(Integer, default=0)
    success_count: Mapped[int] = mapped_column(Integer, default=0)  # 0=unseen, -1=introduced
    # Lesson tracking
    lesson_viewed: Mapped[bool] = mapped_column(Boolean, default=False)


class TopKanji(Base):
    """Top kanji by frequency, with lazy LLM annotation and SRS scheduling."""
    __tablename__ = "top_kanji"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    rank: Mapped[int] = mapped_column(Integer, nullable=False, unique=True)
    kanji: Mapped[str] = mapped_column(String, nullable=False)
    count: Mapped[Optional[int]] = mapped_column(Integer)  # corpus frequency
    # LLM-populated fields (initially empty)
    on_readings: Mapped[Optional[str]] = mapped_column(Text)
    kun_readings: Mapped[Optional[str]] = mapped_column(Text)
    meanings_en: Mapped[Optional[str]] = mapped_column(Text)
    jlpt_level: Mapped[Optional[str]] = mapped_column(String)
    annotated: Mapped[bool] = mapped_column(Boolean, default=False)
    # SRS fields
    next_review: Mapped[datetime.datetime] = mapped_column(DateTime, default=lambda: datetime.datetime.now(datetime.UTC))
    interval: Mapped[int] = mapped_column(Integer, default=1)
    ease_factor: Mapped[float] = mapped_column(SAFloat, default=2.5)
    repetitions: Mapped[int] = mapped_column(Integer, default=0)
    success_count: Mapped[int] = mapped_column(Integer, default=0)
    lesson_viewed: Mapped[bool] = mapped_column(Boolean, default=False)


class TopWord(Base):
    """Top words by frequency, with lazy LLM annotation and SRS scheduling."""
    __tablename__ = "top_words"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    rank: Mapped[int] = mapped_column(Integer, nullable=False, unique=True)
    lemma: Mapped[str] = mapped_column(String, nullable=False)
    reading_csv: Mapped[Optional[str]] = mapped_column(String)  # original from CSV
    count: Mapped[Optional[int]] = mapped_column(Integer)  # corpus frequency
    # LLM-populated fields (replaces dubious CSV data)
    reading: Mapped[Optional[str]] = mapped_column(Text)
    meanings_en: Mapped[Optional[str]] = mapped_column(Text)
    pos_tags: Mapped[Optional[str]] = mapped_column(Text)
    jlpt_level: Mapped[Optional[str]] = mapped_column(String)
    annotated: Mapped[bool] = mapped_column(Boolean, default=False)
    # SRS fields
    next_review: Mapped[datetime.datetime] = mapped_column(DateTime, default=lambda: datetime.datetime.now(datetime.UTC))
    interval: Mapped[int] = mapped_column(Integer, default=1)
    ease_factor: Mapped[float] = mapped_column(SAFloat, default=2.5)
    repetitions: Mapped[int] = mapped_column(Integer, default=0)
    success_count: Mapped[int] = mapped_column(Integer, default=0)
    lesson_viewed: Mapped[bool] = mapped_column(Boolean, default=False)


def is_db_initialized() -> bool:
    """Check if the database is already initialized by checking if tables exist."""
    from sqlalchemy import inspect
    inspector = inspect(engine)
    table_names = inspector.get_table_names()

    # Check if key tables exist
    required_tables = {'vocabulary', 'kanji', 'grammar', 'card_aspects', 'progress', 'messages'}
    return required_tables.issubset(set(table_names))


def init_db() -> None:
    """Initialize the database by creating all tables."""
    Base.metadata.create_all(bind=engine)


def get_session() -> Session:
    return SessionLocal()


def save_message(message: str) -> None:
    session: Session = get_session()
    msg = Message(content=message)
    session.add(msg)
    session.commit()
    session.close()


from .scheduler import sm2_schedule, fsrs_schedule
from openai import OpenAI
from typing import Any
try:
    import numpy as np
    import faiss  # type: ignore
except ImportError:
    np = None  # type: ignore
    faiss = None  # type: ignore

# Initialize OpenAI client (only if API key is present)
_openai_client: Optional[OpenAI] = None
if "OPENAI_API_KEY" in os.environ:
    _openai_client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

# Optional dependency typing fallback
import typing
try:
    import numpy as np  # type: ignore
except ImportError:
    np = None  # type: ignore

# Cache for embeddings to avoid repeated API calls in same session
_embedding_cache: dict[str, Any] = {}

def _get_embedding(text: str) -> Any:
    """Get embedding vector from OpenAI and cache results.
    In TEST_MODE, return a deterministic fake embedding."""
    if text in _embedding_cache:
        return _embedding_cache[text]

    # Short-circuit if in test mode
    if os.getenv("TEST_MODE") == "1":
        import numpy as _np
        # Deterministic fake embedding based on text hash
        h = abs(hash(text)) % 1000
        import numpy as np  # type: ignore
        fake_vec: Any = _np.array(
            [float((h + i) % 100) / 100.0 for i in range(16)], dtype="float32"
        )
        _embedding_cache[text] = fake_vec
        return fake_vec

    if _openai_client is None:
        raise RuntimeError("OpenAI client not initialized. Set OPENAI_API_KEY to use embeddings.")
    response = _openai_client.embeddings.create(
        input=text,
        model="text-embedding-3-small"
    )
    try:
        import numpy as _np
        vec: Any = _np.array(response.data[0].embedding, dtype="float32")
    except Exception:
        vec = response.data[0].embedding  # type: ignore
    _embedding_cache[text] = vec
    return vec

def _serialize_embedding(vec: Any) -> bytes:
    import numpy as _np
    return _np.asarray(vec, dtype="float32").tobytes()

def _deserialize_embedding(blob: bytes) -> Any:
    import numpy as _np
    return _np.frombuffer(blob, dtype="float32") if blob is not None else None

def _cosine_similarity(vec_a: Any, vec_b: Any) -> float:
    try:
        import numpy as _np
        norm_a = float(_np.linalg.norm(vec_a))
        norm_b = float(_np.linalg.norm(vec_b))
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(_np.dot(vec_a, vec_b) / (norm_a * norm_b))
    except Exception:
        return 0.0



# Re-export public API functions for external modules/tests
__all__ = [
    "get_session", "save_message",
    "evaluate_answer_with_ai", "get_next_card",
    "review_card", "update_progress", "get_progress",
    "add_card", "add_kanji", "add_grammar",
    "add_phrase", "add_idiom",
    "_check_semantic_duplicate_with_ai", "_get_existing_items_for_ai_check",
    "get_daily_progress",
    "get_next_bunpro_card", "review_bunpro_card", "get_bunpro_progress",
    "import_bunpro_csv", "BunproGrammar",
    "TopKanji", "TopWord",
    "import_top_kanji_csv", "import_top_words_csv",
    "get_next_top_kanji", "review_top_kanji", "get_top_kanji_progress", "get_kanji_distractors",
    "annotate_top_kanji", "mark_top_kanji_lesson_viewed",
    "get_next_top_word", "review_top_word", "get_top_word_progress", "get_word_distractors",
    "annotate_top_word", "mark_top_word_lesson_viewed",
]


# ----------------------------------------------------------------------
# Semantic Duplicate Detection
# ----------------------------------------------------------------------
def _check_semantic_duplicate_with_ai(new_item: str, existing_items: List[str], item_type: str) -> Optional[str]:
    """
    Use embeddings for top-10 similarity search, then confirm with GPT-5-mini if duplicate.
    Uses FAISS if available for faster search.
    """

    if _openai_client is None:
        if DEBUG_MODE:
            print("⚠️ Skipping AI duplicate check because OpenAI client is not initialized")
        # In testing as well, return None consistently for determinism
        return None

    new_vec = _get_embedding(new_item)
    if DEBUG_MODE:
        print(f"🔎 Generated embedding for new {item_type}: '{new_item[:40]}'...")

    # Build embeddings matrix
    session = get_session()
    emb_list = []
    valid_items = []
    for existing in existing_items:
        row: Optional[Any] = None
        if item_type == "grammar":
            row = session.query(Grammar).filter_by(point=existing).first()
        elif item_type == "phrase":
            row = session.query(Phrase).filter_by(phrase=existing).first()
        elif item_type == "idiom":
            row = session.query(Idiom).filter_by(idiom=existing).first()
        if row and hasattr(row, 'embedding') and row.embedding:
            emb_vec = _deserialize_embedding(row.embedding)
            emb_list.append(emb_vec)
            valid_items.append(existing)
    session.close()

    if not emb_list:
        return None

    import numpy as _np
    emb_matrix = _np.stack(emb_list)
    query_vec = _np.expand_dims(new_vec, axis=0)

    # Use FAISS if available, else fallback to slow loop
    scores = []
    if faiss is not None:
        index = faiss.IndexFlatIP(emb_matrix.shape[1])
        emb_norm = emb_matrix / _np.linalg.norm(emb_matrix, axis=1, keepdims=True)
        q_norm = query_vec / _np.linalg.norm(query_vec)
        index.add(emb_norm.astype("float32"))  # type: ignore
        D, I = index.search(q_norm.astype("float32"), min(10, len(valid_items)))  # type: ignore
        for idx, score in zip(I[0], D[0]):
            if idx != -1:
                scores.append((valid_items[idx], float(score)))
    else:
        for existing, emb_vec in zip(valid_items, emb_list):
            score = _cosine_similarity(new_vec, emb_vec)
            scores.append((existing, score))

    if not scores:
        return None

    scores.sort(key=lambda x: x[1], reverse=True)
    candidates = [s[0] for s in scores[:10]]
    if DEBUG_MODE:
        print(f"📊 Top {len(candidates)} candidates: {candidates}")

    # Ask GPT if any candidate is a duplicate
    prompt = f"""
    New item: "{new_item}"
    Candidates: {candidates}

    Task: Determine if the new item is a semantic duplicate of any candidate.
    Respond with JSON strictly in this format:
    {{
        "duplicate": true/false,
        "match": "<the candidate string if duplicate, else empty>"
    }}
    """

    try:
        import json
        response = _openai_client.chat.completions.create(
            model="gpt-5-mini",
            messages=[
                {"role": "system", "content": "You are an assistant for detecting duplicates in Japanese language learning content."},
                {"role": "user", "content": prompt}
            ],
            temperature=1
        )
        raw_content: str = response.choices[0].message.content or "{}"
        result: Dict[str, Any] = json.loads(raw_content)
        match_val: Optional[str] = result.get("match") if result.get("duplicate") else None
        if isinstance(match_val, str):
            return match_val
    except Exception as e:
        if DEBUG_MODE:
            print(f"❌ LLM duplicate check failed: {e}")

    return None


def _get_existing_items_for_ai_check(session: Session, item_type: str) -> List[str]:
    """Get existing items of the specified type for AI duplicate checking."""
    if item_type == "grammar":
        items = session.query(Grammar.point).all()
        return [item[0] for item in items]
    elif item_type == "phrase":
        items = session.query(Phrase.phrase).all()
        return [item[0] for item in items]
    elif item_type == "idiom":
        items = session.query(Idiom.idiom).all()
        return [item[0] for item in items]
    return []


def evaluate_answer_with_ai(aspect: CardAspect, user_answer: str, model: Any, asked_question: Optional[str] = None) -> tuple[int, str]:
    """
    AI-powered evaluation of a user's answer using semantic understanding.
    Model parameter is REQUIRED - no fallback to rule-based evaluation.
    Pass the actually delivered exercise prompt via `asked_question` so grading aligns with what the learner saw.

    Returns (quality_score, feedback_message).
    Quality scores:
        0 - No answer or completely wrong
        1 - Very poor understanding
        2 - Poor understanding with major errors
        3 - Acceptable with some errors
        4 - Good understanding with minor issues
        5 - Perfect or excellent answer
    """
    if model is None:
        raise ValueError("AI model is required for answer evaluation. Please ensure OpenAI API key is configured.")

    session = get_session()

    # Reattach aspect to this session if detached
    if getattr(aspect, "id", None) is not None:
        loaded = session.get(CardAspect, aspect.id)
        if loaded is not None:
            aspect = loaded

    # Retrieve full context
    context = _get_full_context_for_evaluation(aspect, session)

    # Check for empty answer
    if not user_answer.strip():
        session.close()
        return 0, "No answer provided. Try again!"

    # Use AI for semantic evaluation
    quality, feedback = _ai_semantic_evaluation(
        model,
        aspect,
        user_answer,
        context,
        asked_question=asked_question
    )
    session.close()
    return quality, feedback


def _get_full_context_for_evaluation(aspect: CardAspect, session: Session) -> Dict[str, Any]:
    """Retrieve comprehensive context for evaluation."""
    context: Dict[str, Any] = {
        "aspect_type": aspect.aspect_type,
        "parent_type": aspect.parent_type,
        "prompt": aspect.prompt_template,
        "expected_answers": []
    }

    # Get parent content based on type
    if aspect.parent_type == "vocabulary":
        vocab = session.get(Vocabulary, aspect.parent_id)
        if vocab:
            context["word"] = vocab.word
            context["reading"] = vocab.reading
            context["meaning"] = vocab.meaning
            context["expected_answers"] = _get_expected_vocab_answers(vocab, aspect.aspect_type)
    elif aspect.parent_type == "kanji":
        kanji = session.get(Kanji, aspect.parent_id)
        if kanji:
            context["character"] = kanji.character
            context["onyomi"] = kanji.onyomi
            context["kunyomi"] = kanji.kunyomi
            context["meanings"] = kanji.meanings
            context["expected_answers"] = _get_expected_kanji_answers(kanji, aspect.aspect_type)
    elif aspect.parent_type == "grammar":
        grammar = session.get(Grammar, aspect.parent_id)
        if grammar:
            context["point"] = grammar.point
            context["explanation"] = grammar.explanation
            context["example"] = grammar.example
            context["expected_answers"] = _get_expected_grammar_answers(grammar, aspect.aspect_type)
    elif aspect.parent_type == "phrase":
        phrase = session.get(Phrase, aspect.parent_id)
        if phrase:
            context["phrase"] = phrase.phrase
            context["meaning"] = phrase.meaning
            context["expected_answers"] = _get_expected_phrase_answers(phrase, aspect.aspect_type)
    elif aspect.parent_type == "idiom":
        idiom = session.get(Idiom, aspect.parent_id)
        if idiom:
            context["idiom"] = idiom.idiom
            context["meaning"] = idiom.meaning
            context["example"] = idiom.example
            context["expected_answers"] = _get_expected_idiom_answers(idiom, aspect.aspect_type)

    return context


def _get_expected_vocab_answers(vocab: Vocabulary, aspect_type: str) -> List[str]:
    """Get expected answers for vocabulary aspects."""
    if aspect_type == "meaning":
        return [vocab.meaning] if vocab.meaning else []
    elif aspect_type == "reading":
        return [vocab.reading] if vocab.reading else []
    elif aspect_type == "usage":
        # For usage, we expect a sentence containing the word
        return [vocab.word]
    return []


def _get_expected_kanji_answers(kanji: Kanji, aspect_type: str) -> List[str]:
    """Get expected answers for kanji aspects."""
    if aspect_type == "meaning":
        return [kanji.meanings] if kanji.meanings else []
    elif aspect_type == "onyomi":
        return [kanji.onyomi] if kanji.onyomi else []
    elif aspect_type == "kunyomi":
        return [kanji.kunyomi] if kanji.kunyomi else []
    elif aspect_type == "usage":
        return [kanji.character]
    return []


def _get_expected_grammar_answers(grammar: Grammar, aspect_type: str) -> List[str]:
    """Get expected answers for grammar aspects."""
    if aspect_type == "explanation":
        return [grammar.explanation] if grammar.explanation else []
    elif aspect_type == "usage":
        return [grammar.point]
    return []


def _get_expected_phrase_answers(phrase: Phrase, aspect_type: str) -> List[str]:
    """Get expected answers for phrase aspects."""
    if aspect_type == "meaning":
        return [phrase.meaning] if phrase.meaning else []
    elif aspect_type == "usage":
        return [phrase.phrase]
    return []


def _get_expected_idiom_answers(idiom: Idiom, aspect_type: str) -> List[str]:
    """Get expected answers for idiom aspects."""
    if aspect_type == "meaning":
        return [idiom.meaning] if idiom.meaning else []
    elif aspect_type == "usage":
        return [idiom.idiom]
    return []


def _ai_semantic_evaluation(
    model: Any,
    aspect: CardAspect,
    user_answer: str,
    context: Dict[str, Any],
    asked_question: Optional[str] = None
) -> tuple[int, str]:
    """Use AI to evaluate answer semantically."""

    system_prompt = """You are a Japanese language learning assistant evaluating student answers.
    Evaluate the answer based on correctness, completeness, and understanding.
    Consider partial credit for partially correct answers.
    Be encouraging but accurate in your assessment.

    Return your evaluation in exactly this JSON format:
    {
        "quality": <0-5 integer>,
        "feedback": "<constructive feedback string>"
    }

    Quality scoring guidelines:
    0 - Completely wrong or no attempt
    1 - Very poor, fundamental misunderstanding
    2 - Poor, major errors or missing key elements
    3 - Acceptable, shows understanding but has errors
    4 - Good, mostly correct with minor issues
    5 - Excellent, perfect or creative correct answer
    """

    # Build evaluation prompt
    question_text = asked_question if asked_question else context.get("prompt", "Not provided")

    evaluation_prompt = f"""
    Question Type: {context['parent_type']} - {context['aspect_type']}
    Question Asked: {question_text}

    Expected/Acceptable Answers: {', '.join(context.get('expected_answers', ['Not specified']))}

    Student's Answer: {user_answer}

    Additional Context:
    """

    # Add relevant context based on type
    if context['parent_type'] == "vocabulary":
        evaluation_prompt += f"""
        Word: {context.get('word', 'N/A')}
        Reading: {context.get('reading', 'N/A')}
        Meaning: {context.get('meaning', 'N/A')}
        """
    elif context['parent_type'] == "kanji":
        evaluation_prompt += f"""
        Character: {context.get('character', 'N/A')}
        Onyomi: {context.get('onyomi', 'N/A')}
        Kunyomi: {context.get('kunyomi', 'N/A')}
        Meanings: {context.get('meanings', 'N/A')}
        """
    elif context['parent_type'] == "grammar":
        evaluation_prompt += f"""
        Grammar Point: {context.get('point', 'N/A')}
        Explanation: {context.get('explanation', 'N/A')}
        Example: {context.get('example', 'N/A')}
        """
    elif context['parent_type'] == "phrase":
        evaluation_prompt += f"""
        Phrase: {context.get('phrase', 'N/A')}
        Meaning: {context.get('meaning', 'N/A')}
        """
    elif context['parent_type'] == "idiom":
        evaluation_prompt += f"""
        Idiom: {context.get('idiom', 'N/A')}
        Meaning: {context.get('meaning', 'N/A')}
        Example: {context.get('example', 'N/A')}
        """

    evaluation_prompt += """

    Evaluate the student's answer considering:
    1. Semantic correctness (not just exact match)
    2. Understanding of the concept
    3. Appropriate usage if it's a usage question
    4. Partial credit for partially correct answers
    5. Accept synonyms and alternative correct forms

    For usage questions, check if they've used the item correctly in context.
    For meaning questions, accept reasonable synonyms and paraphrases.
    For reading questions, check hiragana/katakana accuracy.
    """

    try:
        import json
        if DEBUG_MODE:
            print(f"🧠 AI Evaluation Details:")
            print(f"   System prompt length: {len(system_prompt)} characters")
            print(f"   Evaluation prompt length: {len(evaluation_prompt)} characters")
            print(f"   Context keys: {list(context.keys())}")
            print(f"   Expected answers: {context.get('expected_answers', [])}")

        response = model.prompt(evaluation_prompt, system=system_prompt)
        response_text = response.text()
        if DEBUG_MODE:
            print(f"   AI evaluation raw response: {response_text}")

        result = json.loads(response_text)
        if DEBUG_MODE:
            print(f"   Parsed JSON result: {result}")

        quality = int(result.get("quality", 3))
        feedback = str(result.get("feedback", "Answer evaluated."))
        quality = max(0, min(5, quality))
        print(f"✅ Evaluation complete: Quality={quality}, Feedback sample: {feedback[:60]}...")
        return quality, feedback

    except Exception as e:
        print(f"❌ Error in AI evaluation: {e} ({type(e).__name__})")
        if DEBUG_MODE:
            import traceback
            print(f"   Traceback: {traceback.format_exc()}")
        return 3, f"Answer received: '{user_answer}'. Please review manually."


# Remove the _simple_evaluation_fallback function as it's no longer needed


def get_next_card(user: str = "default_user", exclude_id: Optional[int] = None) -> Optional[CardAspect]:
    """Get the next CardAspect to review.
    Prioritize due review cards, and limit new cards per day using UserSettings.
    Optionally exclude a card by ID (e.g. when preloading next card).
    """
    from sqlalchemy import func
    import random, datetime

    session: Session = get_session()
    now = datetime.datetime.now(datetime.UTC)

    # First priority: due review cards (group by parent, ensure success_count > 0 and due)
    subq = (session.query(
                CardAspect.parent_type,
                CardAspect.parent_id,
                func.min(CardAspect.next_review).label("earliest")
            )
            .filter(CardAspect.success_count > 0)
            .group_by(CardAspect.parent_type, CardAspect.parent_id)
            .subquery())

    rows = session.query(subq).filter(subq.c.earliest <= now).all()
    if rows:
        random.shuffle(rows)
        for row in rows:
            query = session.query(CardAspect).filter(
                CardAspect.parent_type == row.parent_type,
                CardAspect.parent_id == row.parent_id
            ).order_by(CardAspect.next_review)
            if exclude_id is not None:
                query = query.filter(CardAspect.id != exclude_id)
            card = query.first()
            if card is not None:
                # Ensure this remains a review card (success_count >= 1)
                if card.success_count < 1:
                    card.success_count = 1
                    session.commit()
                session.refresh(card)
                session.expunge(card)
                session.close()
                return card

    # Otherwise consider new cards (never studied, success_count == 0)
    settings = session.query(UserSettings).filter_by(user=user).first()
    if not settings:
        # Only create if none exists in DB (avoids IntegrityError with parallel sessions/tests)
        settings = UserSettings(user=user)
        session.add(settings)
        session.commit()

    # Reset if it's a new day
    if now.date() != settings.last_reset.date():
        settings.last_reset = now
        settings.new_cards_today = 0
        session.commit()

    # Refresh settings to ensure we see committed values from prior calls
    session.refresh(settings)

    # Enforce quota check BEFORE selecting a card
    if settings.new_cards_today >= settings.max_new_per_day:
        # Quota reached: still allow returning due REVIEW cards, but no fresh new cards
        review_subq = (session.query(
                            CardAspect.parent_type,
                            CardAspect.parent_id,
                            func.min(CardAspect.next_review).label("earliest")
                        )
                        .filter(CardAspect.success_count > 0)
                        .group_by(CardAspect.parent_type, CardAspect.parent_id)
                        .subquery())

        review_rows = session.query(review_subq).filter(review_subq.c.earliest <= now).all()
        if review_rows:
            random.shuffle(review_rows)
            for chosen in review_rows:
                query = session.query(CardAspect).filter(
                    CardAspect.parent_type == chosen.parent_type,
                    CardAspect.parent_id == chosen.parent_id
                ).order_by(CardAspect.next_review)
                if exclude_id is not None:
                    query = query.filter(CardAspect.id != exclude_id)
                review_card = query.first()
                if review_card:
                    if review_card.success_count < 1:
                        review_card.success_count = 1
                        session.commit()
                    session.expunge(review_card)
                    session.close()
                    return review_card
        session.close()
        return None

    # Pick one new card at random at the parent level (avoid multiple aspects of same concept)
    new_subq = (session.query(
                    CardAspect.parent_type,
                    CardAspect.parent_id,
                    func.min(CardAspect.id).label("any_id")
                )
                .filter(CardAspect.success_count == 0)
                .filter(CardAspect.next_review <= now)
                .group_by(CardAspect.parent_type, CardAspect.parent_id)
                .subquery())

    new_rows = session.query(new_subq).all()
    if new_rows:
        random.shuffle(new_rows)
        for chosen in new_rows:
            query = session.query(CardAspect).filter(
                CardAspect.parent_type == chosen.parent_type,
                CardAspect.parent_id == chosen.parent_id,
                CardAspect.success_count == 0,
                CardAspect.next_review <= now
            ).order_by(CardAspect.id)
            if exclude_id is not None:
                query = query.filter(CardAspect.id != exclude_id)
            new_card = query.first()
            if new_card:
                defer_until = now + datetime.timedelta(days=1)
                siblings = (
                    session.query(CardAspect)
                    .filter(
                        CardAspect.parent_type == new_card.parent_type,
                        CardAspect.parent_id == new_card.parent_id,
                        CardAspect.id != new_card.id,
                        CardAspect.success_count == 0
                    )
                    .all()
                )
                for sibling in siblings:
                    sibling.next_review = defer_until
                # Increment counter first to persist limit
                settings.new_cards_today += 1
                if new_card.success_count == 0:
                    new_card.success_count = -1  # sentinel meaning "introduced but not reviewed yet"
                new_card.next_review = now
                session.add(settings)
                session.commit()
                session.refresh(settings)
                session.refresh(new_card)
                session.expunge(new_card)
                # Update DailyProgress for new cards seen
                today = now.date()
                daily = session.query(DailyProgress).filter_by(user=user, date=today).first()
                if not daily:
                    daily = DailyProgress(user=user, date=today, cards_reviewed=0, new_cards_seen=0)
                    session.add(daily)
                if daily.new_cards_seen is None:
                    daily.new_cards_seen = 0
                daily.new_cards_seen += 1
                session.commit()
                session.close()
                return new_card

    session.close()
    return None


def update_progress(user: str, correct: bool) -> None:
    """Update progress stats for the given user."""
    session: Session = get_session()
    prog: Optional[Progress] = session.query(Progress).filter_by(user=user).first()
    if not prog:
        prog = Progress(user=user, total_reviews=0, correct_answers=0)
        session.add(prog)
    prog.total_reviews = prog.total_reviews + 1
    if correct:
        prog.correct_answers = prog.correct_answers + 1
    # Update DailyProgress
    today = datetime.date.today()
    daily = session.query(DailyProgress).filter_by(user=user, date=today).one_or_none()
    if daily is None:
        daily = DailyProgress(user=user, date=today, cards_reviewed=0, new_cards_seen=0)
        session.add(daily)
    if daily.cards_reviewed is None:
        daily.cards_reviewed = 0
    daily.cards_reviewed += 1
    session.commit()


def get_daily_progress(user: str, days: int = 30) -> list[dict[str, object]]:
    """
    Return the last `days` of daily progress stats for a user, ordered by date ascending.
    """
    import datetime
    session: Session = get_session()
    today = datetime.date.today()
    start_date = today - datetime.timedelta(days=days - 1)

    rows = (
        session.query(DailyProgress)
        .filter(DailyProgress.user == user, DailyProgress.date >= start_date)
        .order_by(DailyProgress.date.asc())
        .all()
    )
    session.close()

    results: list[dict[str, object]] = []
    for row in rows:
        results.append({
            "date": row.date.isoformat(),
            "cards_reviewed": row.cards_reviewed or 0,
            "new_cards_seen": row.new_cards_seen or 0,
        })
    return results
    session.close()


def get_progress(user: str) -> Optional[dict[str, Any]]:
    """Return progress plus additional stats."""
    session: Session = get_session()
    prog: Optional[Progress] = session.query(Progress).filter_by(user=user).first()
    settings: Optional[UserSettings] = session.query(UserSettings).filter_by(user=user).first()

    # Count how many unique parent items have ever been seen (success_count != 0)
    from sqlalchemy import func
    total_seen = (session.query(CardAspect.parent_type, CardAspect.parent_id)
                         .filter(CardAspect.success_count != 0)
                         .group_by(CardAspect.parent_type, CardAspect.parent_id)
                         .count())

    # How many new cards introduced today
    new_today = settings.new_cards_today if settings else 0

    # Count remaining unseen unique parents
    unseen_count = (session.query(CardAspect.parent_type, CardAspect.parent_id)
                           .filter(CardAspect.success_count == 0)
                           .group_by(CardAspect.parent_type, CardAspect.parent_id)
                           .count())

    result = None
    if prog:
        result = {
            "total_reviews": prog.total_reviews,
            "correct_answers": prog.correct_answers,
            "last_updated": prog.last_updated,
            "total_seen": total_seen,
            "new_today": new_today,
            "remaining_unseen": unseen_count,
        }

    session.close()
    return result


def get_analytics_data(user: str) -> Dict[str, Any]:
    """Compute advanced analytics for the progress page.

    Returns a dict with keys:
        streak            – {current, longest}
        heatmap           – [{date, count}, …] for last 90 days
        content_breakdown – [{type, total, studied}, …]
        srs_stages        – {new, learning, young, mature}
        srs_by_type       – [{type, new, learning, young, mature}, …]
        forecast          – [{date, count}, …] for next 14 days
        weakest           – [{id, parent_type, aspect_type, label, ease_factor, interval, next_review}, …]
        strongest         – [{id, parent_type, aspect_type, label, interval, ease_factor, next_review}, …]
        module_progress   – {bunpro: {total, studied}, kanji: …, words: …}
        averages          – {avg_ease, avg_interval, mature_count}
    """
    import datetime as _dt
    from sqlalchemy import func, case

    session: Session = get_session()
    now = _dt.datetime.now(_dt.timezone.utc)
    today = _dt.date.today()

    result: Dict[str, Any] = {}

    # ── Streak ────────────────────────────────────────────────────────
    rows = (
        session.query(DailyProgress.date)
        .filter(DailyProgress.user == user, DailyProgress.cards_reviewed > 0)
        .order_by(DailyProgress.date.desc())
        .all()
    )
    studied_dates = {r.date for r in rows}
    # current streak (allow today or yesterday as start)
    current_streak = 0
    check = today
    if check not in studied_dates:
        check = today - _dt.timedelta(days=1)
    while check in studied_dates:
        current_streak += 1
        check -= _dt.timedelta(days=1)
    # longest streak
    longest_streak = 0
    if studied_dates:
        sorted_dates = sorted(studied_dates)
        run = 1
        for i in range(1, len(sorted_dates)):
            if sorted_dates[i] - sorted_dates[i - 1] == _dt.timedelta(days=1):
                run += 1
            else:
                longest_streak = max(longest_streak, run)
                run = 1
        longest_streak = max(longest_streak, run)
    result["streak"] = {"current": current_streak, "longest": longest_streak}

    # ── Heatmap (last 90 days) ────────────────────────────────────────
    heatmap_start = today - _dt.timedelta(days=89)
    heatmap_rows = (
        session.query(DailyProgress.date, DailyProgress.cards_reviewed)
        .filter(DailyProgress.user == user, DailyProgress.date >= heatmap_start)
        .order_by(DailyProgress.date.asc())
        .all()
    )
    heatmap_map = {r.date: (r.cards_reviewed or 0) for r in heatmap_rows}
    heatmap: list[Dict[str, Any]] = []
    for offset in range(90):
        d = heatmap_start + _dt.timedelta(days=offset)
        heatmap.append({"date": d.isoformat(), "count": heatmap_map.get(d, 0)})
    result["heatmap"] = heatmap

    # ── Content breakdown by parent_type ──────────────────────────────
    breakdown_rows = (
        session.query(
            CardAspect.parent_type,
            func.count(func.distinct(CardAspect.parent_id)).label("total"),
        )
        .group_by(CardAspect.parent_type)
        .all()
    )
    studied_rows = (
        session.query(
            CardAspect.parent_type,
            func.count(func.distinct(CardAspect.parent_id)).label("studied"),
        )
        .filter(CardAspect.success_count != 0)
        .group_by(CardAspect.parent_type)
        .all()
    )
    total_map = {r.parent_type: r.total for r in breakdown_rows}
    studied_map = {r.parent_type: r.studied for r in studied_rows}
    content_breakdown: list[Dict[str, Any]] = []
    for ptype in sorted(total_map.keys()):
        content_breakdown.append({
            "type": ptype,
            "total": total_map.get(ptype, 0),
            "studied": studied_map.get(ptype, 0),
        })
    result["content_breakdown"] = content_breakdown

    # ── SRS stage distribution ────────────────────────────────────────
    all_aspects = (
        session.query(CardAspect.parent_type, CardAspect.interval, CardAspect.success_count)
        .all()
    )
    stages: Dict[str, int] = {"new": 0, "learning": 0, "young": 0, "mature": 0}
    srs_by_type: Dict[str, Dict[str, int]] = {}
    for a in all_aspects:
        if a.success_count == 0:
            stage = "new"
        elif a.interval <= 6:
            stage = "learning"
        elif a.interval <= 30:
            stage = "young"
        else:
            stage = "mature"
        stages[stage] += 1
        if a.parent_type not in srs_by_type:
            srs_by_type[a.parent_type] = {"new": 0, "learning": 0, "young": 0, "mature": 0}
        srs_by_type[a.parent_type][stage] += 1
    result["srs_stages"] = stages
    result["srs_by_type"] = [
        {"type": k, **v} for k, v in sorted(srs_by_type.items())
    ]

    # ── Review forecast (next 14 days) ────────────────────────────────
    forecast_end = now + _dt.timedelta(days=14)
    forecast_rows = (
        session.query(CardAspect.next_review)
        .filter(CardAspect.success_count > 0, CardAspect.next_review <= forecast_end)
        .all()
    )
    forecast_map: Dict[str, int] = {}
    for offset in range(15):
        d = today + _dt.timedelta(days=offset)
        forecast_map[d.isoformat()] = 0
    for r in forecast_rows:
        d = r.next_review.date().isoformat() if r.next_review else None
        if d and d in forecast_map:
            forecast_map[d] += 1
        elif d and r.next_review.date() <= today:
            # Overdue cards count as today
            forecast_map[today.isoformat()] += 1
    result["forecast"] = [{"date": k, "count": v} for k, v in sorted(forecast_map.items())]

    # ── Weakest cards (lowest ease_factor, studied only) ──────────────
    weakest_rows = (
        session.query(CardAspect)
        .filter(CardAspect.success_count > 0)
        .order_by(CardAspect.ease_factor.asc())
        .limit(10)
        .all()
    )
    weakest: list[Dict[str, Any]] = []
    for c in weakest_rows:
        label = _get_card_label(session, c)
        weakest.append({
            "id": c.id,
            "parent_type": c.parent_type,
            "aspect_type": c.aspect_type,
            "label": label,
            "ease_factor": round(c.ease_factor, 2),
            "interval": c.interval,
            "next_review": c.next_review.isoformat() if c.next_review else None,
        })
    result["weakest"] = weakest

    # ── Strongest cards (highest interval) ────────────────────────────
    strongest_rows = (
        session.query(CardAspect)
        .filter(CardAspect.success_count > 0)
        .order_by(CardAspect.interval.desc())
        .limit(10)
        .all()
    )
    strongest: list[Dict[str, Any]] = []
    for c in strongest_rows:
        label = _get_card_label(session, c)
        strongest.append({
            "id": c.id,
            "parent_type": c.parent_type,
            "aspect_type": c.aspect_type,
            "label": label,
            "ease_factor": round(c.ease_factor, 2),
            "interval": c.interval,
            "next_review": c.next_review.isoformat() if c.next_review else None,
        })
    result["strongest"] = strongest

    # ── Module progress (Bunpro, TopKanji, TopWord) ───────────────────
    module_progress: Dict[str, Dict[str, int]] = {}
    for Model, key in [(BunproGrammar, "bunpro"), (TopKanji, "kanji"), (TopWord, "words")]:
        try:
            total = session.query(func.count(Model.id)).scalar() or 0
            studied = (session.query(func.count(Model.id))
                       .filter(Model.success_count > 0).scalar() or 0)
            module_progress[key] = {"total": total, "studied": studied}
        except Exception:
            module_progress[key] = {"total": 0, "studied": 0}
    result["module_progress"] = module_progress

    # ── Averages ──────────────────────────────────────────────────────
    avg_row = (
        session.query(
            func.avg(CardAspect.ease_factor).label("avg_ease"),
            func.avg(CardAspect.interval).label("avg_interval"),
        )
        .filter(CardAspect.success_count > 0)
        .one()
    )
    mature_count = (
        session.query(func.count(CardAspect.id))
        .filter(CardAspect.success_count > 0, CardAspect.interval > 30)
        .scalar() or 0
    )
    result["averages"] = {
        "avg_ease": round(float(avg_row.avg_ease or 0), 2),
        "avg_interval": round(float(avg_row.avg_interval or 0), 1),
        "mature_count": mature_count,
    }

    session.close()
    return result


def _get_card_label(session: Session, card: CardAspect) -> str:
    """Look up the human-readable label for a CardAspect by joining to its parent table."""
    parent_type = card.parent_type
    parent_id = card.parent_id
    label = f"{parent_type} #{parent_id}"
    try:
        if parent_type == "vocabulary":
            row = session.get(Vocabulary, parent_id)
            if row:
                label = row.word
        elif parent_type == "kanji":
            row = session.get(Kanji, parent_id)
            if row:
                label = row.character
        elif parent_type == "grammar":
            row = session.get(Grammar, parent_id)
            if row:
                label = row.point
        elif parent_type == "phrase":
            row = session.get(Phrase, parent_id)
            if row:
                label = row.phrase
        elif parent_type == "idiom":
            row = session.get(Idiom, parent_id)
            if row:
                label = row.idiom
    except Exception:
        pass
    return label


def review_card(card_id: int, quality: int) -> None:
    """Review a CardAspect and update its scheduling data using SM-2.
    Propagates scheduling updates to all sibling aspects of the same parent.
    """
    session: Session = get_session()
    card: Optional[CardAspect] = session.get(CardAspect, card_id)
    if card:
        new_interval, new_ease, new_reps, next_review = sm2_schedule(
            card.interval, card.ease_factor, card.repetitions, quality
        )

        # Update all sibling aspects for the same parent (keep parent-level scheduling)
        siblings = (
            session.query(CardAspect)
            .filter(CardAspect.parent_type == card.parent_type,
                    CardAspect.parent_id == card.parent_id)
            .all()
        )

        for sibling in siblings:
            sibling.interval = new_interval
            sibling.ease_factor = new_ease
            sibling.repetitions = new_reps
            sibling.next_review = next_review
            if quality >= 3:
                # Increment consistently (tests expect -1 -> 0, 0 -> 1, etc.)
                sibling.success_count = sibling.success_count + 1

        session.commit()
    session.close()


def add_card(content: str) -> bool:
    """Add a vocabulary card with atomic aspects. Returns True if new, False if duplicate."""
    session: Session = get_session()

    # Check for existing vocabulary
    existing = session.query(Vocabulary).filter_by(word=content).first()
    if existing:
        session.close()
        return False

    vocab = Vocabulary(word=content, meaning="")
    vocab_vec = _get_embedding(content)
    vocab.embedding = _serialize_embedding(vocab_vec)
    session.add(vocab)
    session.flush()  # Get the ID

    # Create atomic aspects for this vocabulary item
    aspects = [
        CardAspect(
            parent_type="vocabulary",
            parent_id=vocab.id,
            aspect_type="meaning",
            prompt_template=f"What is the English meaning of: {content}?"
        ),
        CardAspect(
            parent_type="vocabulary",
            parent_id=vocab.id,
            aspect_type="reading",
            prompt_template=f"What is the reading (hiragana/katakana) of: {content}?"
        ),
        CardAspect(
            parent_type="vocabulary",
            parent_id=vocab.id,
            aspect_type="usage",
            prompt_template=f"Use {content} in a natural Japanese sentence."
        )
    ]

    for aspect in aspects:
        session.add(aspect)

    session.commit()
    session.close()
    return True


def add_kanji(character: str, onyomi: str = "", kunyomi: str = "", meanings: str = "") -> bool:
    """Add a kanji with atomic aspects. Returns True if new, False if duplicate."""
    session: Session = get_session()

    # Check for existing kanji
    existing = session.query(Kanji).filter_by(character=character).first()
    if existing:
        session.close()
        return False

    kanji = Kanji(character=character, onyomi=onyomi, kunyomi=kunyomi, meanings=meanings)
    kanji_vec = _get_embedding(character)
    kanji.embedding = _serialize_embedding(kanji_vec)
    session.add(kanji)
    session.flush()

    # Create atomic aspects for this kanji
    aspects = [
        CardAspect(
            parent_type="kanji",
            parent_id=kanji.id,
            aspect_type="meaning",
            prompt_template=f"What are the meanings of the kanji: {character}?"
        ),
        CardAspect(
            parent_type="kanji",
            parent_id=kanji.id,
            aspect_type="onyomi",
            prompt_template=f"What is the onyomi (Chinese reading) of: {character}?"
        ),
        CardAspect(
            parent_type="kanji",
            parent_id=kanji.id,
            aspect_type="kunyomi",
            prompt_template=f"What is the kunyomi (Japanese reading) of: {character}?"
        ),
        CardAspect(
            parent_type="kanji",
            parent_id=kanji.id,
            aspect_type="usage",
            prompt_template=f"Write a word or sentence using the kanji: {character}"
        )
    ]

    for aspect in aspects:
        session.add(aspect)

    session.commit()
    session.close()
    return True


def add_grammar(point: str, explanation: str = "", example: str = "") -> bool:
    """Add a grammar point with atomic aspects. Returns True if new, False if duplicate."""
    session: Session = get_session()

    # First check: exact string match
    existing = session.query(Grammar).filter_by(point=point).first()
    if existing:
        session.close()
        return False

    # Second check: AI semantic duplicate detection
    existing_points = _get_existing_items_for_ai_check(session, "grammar")
    ai_duplicate = _check_semantic_duplicate_with_ai(point, existing_points, "grammar")
    if ai_duplicate:
        session.close()
        return False  # AI detected a semantic duplicate

    grammar = Grammar(point=point, explanation=explanation, example=example)
    grammar_vec = _get_embedding(point)
    grammar.embedding = _serialize_embedding(grammar_vec)
    session.add(grammar)
    session.flush()

    # Create atomic aspects for this grammar point
    aspects = [
        CardAspect(
            parent_type="grammar",
            parent_id=grammar.id,
            aspect_type="explanation",
            prompt_template=f"Explain the grammar point: {point}"
        ),
        CardAspect(
            parent_type="grammar",
            parent_id=grammar.id,
            aspect_type="usage",
            prompt_template=f"Write a Japanese sentence using this grammar point: {point}"
        )
    ]

    for aspect in aspects:
        session.add(aspect)

    session.commit()
    session.close()
    return True


def add_phrase(phrase: str, meaning: str = "") -> bool:
    """Add a phrase with atomic aspects. Returns True if new, False if duplicate."""
    session: Session = get_session()

    # First check: exact string match
    existing = session.query(Phrase).filter_by(phrase=phrase).first()
    if existing:
        session.close()
        return False

    # Second check: AI semantic duplicate detection
    existing_phrases = _get_existing_items_for_ai_check(session, "phrase")
    ai_duplicate = _check_semantic_duplicate_with_ai(phrase, existing_phrases, "phrase")
    if ai_duplicate:
        session.close()
        return False  # AI detected a semantic duplicate

    phrase_obj = Phrase(phrase=phrase, meaning=meaning)
    phrase_vec = _get_embedding(phrase)
    phrase_obj.embedding = _serialize_embedding(phrase_vec)
    session.add(phrase_obj)
    session.flush()

    # Create atomic aspects for this phrase
    aspects = [
        CardAspect(
            parent_type="phrase",
            parent_id=phrase_obj.id,
            aspect_type="meaning",
            prompt_template=f"What does this phrase mean: {phrase}?"
        ),
        CardAspect(
            parent_type="phrase",
            parent_id=phrase_obj.id,
            aspect_type="usage",
            prompt_template=f"Use this phrase in a natural conversation: {phrase}"
        )
    ]

    for aspect in aspects:
        session.add(aspect)

    session.commit()
    session.close()
    return True


def add_conjugation(label: str, category: str = "", description: str = "", example: str = "") -> bool:
    """Add a verb conjugation with atomic aspects. Returns True if new, False if duplicate."""
    session: Session = get_session()

    existing = session.query(VerbConjugation).filter_by(label=label).first()
    if existing:
        session.close()
        return False

    conj = VerbConjugation(category=category, label=label, description=description, example=example)
    vec = _get_embedding(label)
    conj.embedding = _serialize_embedding(vec)
    session.add(conj)
    session.flush()

    # Create atomic aspects for this conjugation
    aspects = [
        CardAspect(
            parent_type="conjugation",
            parent_id=conj.id,
            aspect_type="explanation",
            prompt_template=f"Explain when and how to use the conjugation: {label}"
        ),
        CardAspect(
            parent_type="conjugation",
            parent_id=conj.id,
            aspect_type="example",
            prompt_template=f"Provide a Japanese sentence using the conjugation: {label}"
        ),
        CardAspect(
            parent_type="conjugation",
            parent_id=conj.id,
            aspect_type="drill",
            prompt_template=f"Conjugate the verb '食べる' into: {label} form"
        ),
    ]

    for aspect in aspects:
        session.add(aspect)

    session.commit()
    session.close()
    return True


def add_idiom(idiom: str, meaning: str = "", example: str = "") -> bool:
    """Add an idiom with atomic aspects. Returns True if new, False if duplicate."""
    session: Session = get_session()

    # First check: exact string match
    existing = session.query(Idiom).filter_by(idiom=idiom).first()
    if existing:
        session.close()
        return False

    # Second check: AI semantic duplicate detection
    existing_idioms = _get_existing_items_for_ai_check(session, "idiom")
    ai_duplicate = _check_semantic_duplicate_with_ai(idiom, existing_idioms, "idiom")
    if ai_duplicate:
        session.close()
        return False  # AI detected a semantic duplicate

    idiom_obj = Idiom(idiom=idiom, meaning=meaning, example=example)
    idiom_vec = _get_embedding(idiom)
    idiom_obj.embedding = _serialize_embedding(idiom_vec)
    session.add(idiom_obj)
    session.flush()

    # Create atomic aspects for this idiom
    aspects = [
        CardAspect(
            parent_type="idiom",
            parent_id=idiom_obj.id,
            aspect_type="meaning",
            prompt_template=f"What does this idiom mean: {idiom}?"
        ),
        CardAspect(
            parent_type="idiom",
            parent_id=idiom_obj.id,
            aspect_type="usage",
            prompt_template=f"Write a dialogue or sentence using this idiom: {idiom}"
        )
    ]

    for aspect in aspects:
        session.add(aspect)

    session.commit()
    session.close()
    return True


# ----------------------------------------------------------------------
# Bunpro Grammar Functions
# ----------------------------------------------------------------------

def import_bunpro_csv(csv_path: str = "data/bunpro_jlptplus_usage_ranked_heuristic.csv") -> int:
    """Import Bunpro grammar points from CSV. Skips existing rows by rank.
    Returns the number of newly imported rows."""
    import csv

    session: Session = get_session()
    imported = 0

    with open(csv_path, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rank_val = int(row["rank"])
            existing = session.query(BunproGrammar).filter_by(rank=rank_val).first()
            if existing:
                continue

            bunpro_id_raw = row.get("id", "")
            bunpro_id_val: Optional[float] = float(bunpro_id_raw) if bunpro_id_raw else None
            score_raw = row.get("score", "")
            score_val: Optional[float] = float(score_raw) if score_raw else None

            entry = BunproGrammar(
                rank=rank_val,
                usage_tier=row.get("usage_tier", ""),
                score=score_val,
                register=row.get("register", ""),
                jlpt=row.get("jlpt", ""),
                bunpro_id=bunpro_id_val,
                grammar=row.get("grammar", ""),
                meaning=row.get("meaning", ""),
                url=row.get("url", ""),
                norm=row.get("norm", ""),
            )
            session.add(entry)
            imported += 1

    session.commit()
    session.close()
    print(f"✅ Imported {imported} Bunpro grammar points")
    return imported


def get_next_bunpro_card(user: str = "default_user", exclude_ids: Optional[set] = None,
                         pending_new_count: int = 0) -> Optional[BunproGrammar]:
    """Get the next BunproGrammar card to study.

    Cards are NOT marked as introduced here; that happens in review_bunpro_card()
    when the user actually answers the question.

    Priority:
    1. Due review cards (success_count > 0, next_review <= now), random order
    2. Next unseen card by rank order (success_count == 0), subject to daily limit

    Args:
        exclude_ids: Card IDs to skip (already in queue / current batch).
        pending_new_count: Number of new (unseen) cards already fetched in this
            batch but not yet answered.  Used to enforce the daily new-card limit
            across a single batch request.
    """
    import random

    session: Session = get_session()
    now = datetime.datetime.now(datetime.UTC)

    # Priority 1: due review cards
    query = session.query(BunproGrammar).filter(
        BunproGrammar.success_count > 0, BunproGrammar.next_review <= now
    )
    if exclude_ids:
        query = query.filter(BunproGrammar.id.notin_(exclude_ids))
    due_cards = query.all()
    if due_cards:
        card = random.choice(due_cards)
        session.expunge(card)
        session.close()
        return card

    # Priority 2: next unseen card by rank, subject to daily new-card limit
    settings = session.query(UserSettings).filter_by(user=user).first()
    if not settings:
        settings = UserSettings(user=user)
        session.add(settings)
        session.commit()

    # Reset if new day
    if now.date() != settings.last_reset.date():
        settings.last_reset = now
        settings.new_cards_today = 0
        session.commit()

    session.refresh(settings)

    if settings.new_cards_today + pending_new_count >= settings.max_new_per_day:
        session.close()
        return None

    # Pick unseen card with lowest rank (most common grammar first)
    unseen_query = session.query(BunproGrammar).filter(BunproGrammar.success_count == 0)
    if exclude_ids:
        unseen_query = unseen_query.filter(BunproGrammar.id.notin_(exclude_ids))
    new_card = unseen_query.order_by(BunproGrammar.rank.asc()).first()
    if new_card:
        # Do NOT mark as introduced here — that happens in review_bunpro_card()
        session.expunge(new_card)
        session.close()
        return new_card

    session.close()
    return None


def review_bunpro_card(card_id: int, quality: int, user: str = "default_user") -> None:
    """Review a BunproGrammar card and update its SM-2 scheduling.

    If the card has never been answered before (success_count <= 0), this also
    counts it as a newly introduced card for the daily new-card quota.
    """
    session: Session = get_session()
    card: Optional[BunproGrammar] = session.get(BunproGrammar, card_id)
    if card:
        is_first_review = card.success_count <= 0

        new_interval, new_ease, new_reps, next_review = sm2_schedule(
            card.interval, card.ease_factor, card.repetitions, quality
        )
        card.interval = new_interval
        card.ease_factor = new_ease
        card.repetitions = new_reps
        card.next_review = next_review
        if quality >= 3:
            card.success_count = max(card.success_count, 0) + 1
        else:
            # Wrong answer: reset to 1 so it stays in the review pool
            card.success_count = max(card.success_count, 1)

        # Count as a new card introduction on first answer
        if is_first_review:
            now = datetime.datetime.now(datetime.UTC)
            settings = session.query(UserSettings).filter_by(user=user).first()
            if settings:
                if now.date() != settings.last_reset.date():
                    settings.last_reset = now
                    settings.new_cards_today = 0
                settings.new_cards_today += 1

        session.commit()
    session.close()


def mark_bunpro_lesson_viewed(card_id: int) -> None:
    """Mark a BunproGrammar card's lesson as viewed."""
    session: Session = get_session()
    card: Optional[BunproGrammar] = session.get(BunproGrammar, card_id)
    if card:
        card.lesson_viewed = True
        session.commit()
    session.close()


def get_bunpro_progress(user: str = "default_user") -> Dict[str, Any]:
    """Get Bunpro Grammar learning progress stats."""
    session: Session = get_session()
    now = datetime.datetime.now(datetime.UTC)

    total = session.query(BunproGrammar).count()
    learned = session.query(BunproGrammar).filter(BunproGrammar.success_count > 0).count()
    introduced = session.query(BunproGrammar).filter(BunproGrammar.success_count == -1).count()
    unseen = session.query(BunproGrammar).filter(BunproGrammar.success_count == 0).count()
    due_now = session.query(BunproGrammar).filter(
        BunproGrammar.success_count > 0,
        BunproGrammar.next_review <= now
    ).count()

    # Next unseen card info
    next_card = (
        session.query(BunproGrammar)
        .filter(BunproGrammar.success_count == 0)
        .order_by(BunproGrammar.rank.asc())
        .first()
    )
    next_grammar = next_card.grammar if next_card else None
    next_rank = next_card.rank if next_card else None

    session.close()
    return {
        "total": total,
        "learned": learned,
        "introduced": introduced,
        "unseen": unseen,
        "due_now": due_now,
        "next_grammar": next_grammar,
        "next_rank": next_rank,
    }


def get_bunpro_distractors(card: BunproGrammar, count: int = 3) -> List[Dict[str, Any]]:
    """Get distractor grammar points for multiple-choice questions.

    Selects grammar points near the target's rank and same JLPT level
    to provide plausible but incorrect choices.
    """
    import random

    session: Session = get_session()

    # Prefer same JLPT level, within ±50 rank
    candidates = (
        session.query(BunproGrammar)
        .filter(
            BunproGrammar.id != card.id,
            BunproGrammar.jlpt == card.jlpt,
            BunproGrammar.rank.between(max(1, card.rank - 50), card.rank + 50),
        )
        .all()
    )

    # Fallback: any JLPT, wider rank range
    if len(candidates) < count:
        candidates = (
            session.query(BunproGrammar)
            .filter(BunproGrammar.id != card.id)
            .all()
        )

    session.close()

    if len(candidates) <= count:
        selected = candidates
    else:
        selected = random.sample(candidates, count)

    return [{"grammar": c.grammar, "meaning": c.meaning, "jlpt": c.jlpt} for c in selected]


# ----------------------------------------------------------------------
# Top Kanji Functions
# ----------------------------------------------------------------------

def import_top_kanji_csv(csv_path: str = "data/top_10000_kanji.csv", max_rows: int = 10433) -> int:
    """Import top kanji from CSV (skeleton data only). Returns count of new rows."""
    import csv
    session: Session = get_session()
    imported = 0
    with open(csv_path, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if i >= max_rows:
                break
            rank_val = int(row["rank"])
            if session.query(TopKanji).filter_by(rank=rank_val).first():
                continue
            count_raw = row.get("count", "")
            entry = TopKanji(
                rank=rank_val,
                kanji=row.get("kanji", ""),
                count=int(count_raw) if count_raw else None,
            )
            session.add(entry)
            imported += 1
    session.commit()
    session.close()
    print(f"✅ Imported {imported} Top Kanji")
    return imported


def get_next_top_kanji(user: str = "default_user", exclude_ids: Optional[set] = None,
                       pending_new_count: int = 0) -> Optional[TopKanji]:
    """Get the next TopKanji card to study (by rank order for new, random for reviews).

    Cards are NOT marked as introduced here; that happens in review_top_kanji().

    Args:
        exclude_ids: Card IDs to skip (already in queue / current batch).
        pending_new_count: Number of new cards already fetched in this batch.
    """
    import random
    session: Session = get_session()
    now = datetime.datetime.now(datetime.UTC)

    # Priority 1: due review cards
    query = session.query(TopKanji).filter(
        TopKanji.success_count > 0, TopKanji.next_review <= now
    )
    if exclude_ids:
        query = query.filter(TopKanji.id.notin_(exclude_ids))
    due_cards = query.all()
    if due_cards:
        card = random.choice(due_cards)
        session.expunge(card)
        session.close()
        return card

    # Priority 2: next unseen by rank, subject to daily limit
    settings = session.query(UserSettings).filter_by(user=user).first()
    if not settings:
        settings = UserSettings(user=user)
        session.add(settings)
        session.commit()
    if now.date() != settings.last_reset.date():
        settings.last_reset = now
        settings.new_cards_today = 0
        session.commit()
    session.refresh(settings)
    if settings.new_cards_today + pending_new_count >= settings.max_new_per_day:
        session.close()
        return None

    unseen_query = session.query(TopKanji).filter(TopKanji.success_count == 0)
    if exclude_ids:
        unseen_query = unseen_query.filter(TopKanji.id.notin_(exclude_ids))
    new_card = unseen_query.order_by(TopKanji.rank.asc()).first()
    if new_card:
        # Do NOT mark as introduced here — that happens in review_top_kanji()
        session.expunge(new_card)
        session.close()
        return new_card

    session.close()
    return None


def review_top_kanji(card_id: int, quality: int, user: str = "default_user") -> None:
    """Review a TopKanji card with SM-2 scheduling.

    If the card has never been answered before (success_count <= 0), this also
    counts it as a newly introduced card for the daily new-card quota.
    """
    session: Session = get_session()
    card: Optional[TopKanji] = session.get(TopKanji, card_id)
    if card:
        is_first_review = card.success_count <= 0

        new_interval, new_ease, new_reps, next_review = sm2_schedule(
            card.interval, card.ease_factor, card.repetitions, quality
        )
        card.interval = new_interval
        card.ease_factor = new_ease
        card.repetitions = new_reps
        card.next_review = next_review
        if quality >= 3:
            card.success_count = max(card.success_count, 0) + 1
        else:
            card.success_count = max(card.success_count, 1)

        if is_first_review:
            now = datetime.datetime.now(datetime.UTC)
            settings = session.query(UserSettings).filter_by(user=user).first()
            if settings:
                if now.date() != settings.last_reset.date():
                    settings.last_reset = now
                    settings.new_cards_today = 0
                settings.new_cards_today += 1

        session.commit()
    session.close()


def annotate_top_kanji(card_id: int, on_readings: str, kun_readings: str,
                       meanings_en: str, jlpt_level: str = "") -> None:
    """Store LLM-generated annotations for a kanji card."""
    session: Session = get_session()
    card: Optional[TopKanji] = session.get(TopKanji, card_id)
    if card:
        card.on_readings = on_readings
        card.kun_readings = kun_readings
        card.meanings_en = meanings_en
        card.jlpt_level = jlpt_level
        card.annotated = True
        session.commit()
    session.close()


def mark_top_kanji_lesson_viewed(card_id: int) -> None:
    """Mark a TopKanji card's lesson as viewed."""
    session: Session = get_session()
    card: Optional[TopKanji] = session.get(TopKanji, card_id)
    if card:
        card.lesson_viewed = True
        session.commit()
    session.close()


def get_top_kanji_progress(user: str = "default_user") -> Dict[str, Any]:
    """Get Top Kanji learning progress stats."""
    session: Session = get_session()
    now = datetime.datetime.now(datetime.UTC)
    total = session.query(TopKanji).count()
    learned = session.query(TopKanji).filter(TopKanji.success_count > 0).count()
    unseen = session.query(TopKanji).filter(TopKanji.success_count == 0).count()
    due_now = session.query(TopKanji).filter(
        TopKanji.success_count > 0, TopKanji.next_review <= now
    ).count()
    annotated = session.query(TopKanji).filter(TopKanji.annotated == True).count()
    next_card = (
        session.query(TopKanji)
        .filter(TopKanji.success_count == 0)
        .order_by(TopKanji.rank.asc())
        .first()
    )
    session.close()
    return {
        "total": total, "learned": learned, "unseen": unseen,
        "due_now": due_now, "annotated": annotated,
        "next_kanji": next_card.kanji if next_card else None,
        "next_rank": next_card.rank if next_card else None,
    }


def get_kanji_distractors(card: TopKanji, count: int = 3) -> List[Dict[str, Any]]:
    """Get distractor kanji for MC questions. Prefers annotated kanji near same rank."""
    import random
    session: Session = get_session()
    candidates = (
        session.query(TopKanji)
        .filter(TopKanji.id != card.id, TopKanji.annotated == True)
        .all()
    )
    session.close()
    if not candidates:
        return []
    selected = random.sample(candidates, min(count, len(candidates)))
    return [{"kanji": c.kanji, "meanings_en": c.meanings_en,
             "on_readings": c.on_readings, "kun_readings": c.kun_readings} for c in selected]


# ----------------------------------------------------------------------
# Top Word Functions
# ----------------------------------------------------------------------

def import_top_words_csv(csv_path: str = "data/top_20000_words.csv", max_rows: int = 10000) -> int:
    """Import top words from CSV (skeleton data only). Returns count of new rows."""
    import csv
    session: Session = get_session()
    imported = 0
    with open(csv_path, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if i >= max_rows:
                break
            rank_val = int(row["rank"])
            if session.query(TopWord).filter_by(rank=rank_val).first():
                continue
            count_raw = row.get("count", "")
            entry = TopWord(
                rank=rank_val,
                lemma=row.get("lemma", ""),
                reading_csv=row.get("reading", ""),
                count=int(count_raw) if count_raw else None,
            )
            session.add(entry)
            imported += 1
    session.commit()
    session.close()
    print(f"✅ Imported {imported} Top Words")
    return imported


def get_next_top_word(user: str = "default_user", exclude_ids: Optional[set] = None,
                      pending_new_count: int = 0) -> Optional[TopWord]:
    """Get the next TopWord card to study (by rank order for new, random for reviews).

    Cards are NOT marked as introduced here; that happens in review_top_word().

    Args:
        exclude_ids: Card IDs to skip (already in queue / current batch).
        pending_new_count: Number of new cards already fetched in this batch.
    """
    import random
    session: Session = get_session()
    now = datetime.datetime.now(datetime.UTC)

    query = session.query(TopWord).filter(
        TopWord.success_count > 0, TopWord.next_review <= now
    )
    if exclude_ids:
        query = query.filter(TopWord.id.notin_(exclude_ids))
    due_cards = query.all()
    if due_cards:
        card = random.choice(due_cards)
        session.expunge(card)
        session.close()
        return card

    settings = session.query(UserSettings).filter_by(user=user).first()
    if not settings:
        settings = UserSettings(user=user)
        session.add(settings)
        session.commit()
    if now.date() != settings.last_reset.date():
        settings.last_reset = now
        settings.new_cards_today = 0
        session.commit()
    session.refresh(settings)
    if settings.new_cards_today + pending_new_count >= settings.max_new_per_day:
        session.close()
        return None

    unseen_query = session.query(TopWord).filter(TopWord.success_count == 0)
    if exclude_ids:
        unseen_query = unseen_query.filter(TopWord.id.notin_(exclude_ids))
    new_card = unseen_query.order_by(TopWord.rank.asc()).first()
    if new_card:
        # Do NOT mark as introduced here — that happens in review_top_word()
        session.expunge(new_card)
        session.close()
        return new_card

    session.close()
    return None


def review_top_word(card_id: int, quality: int, user: str = "default_user") -> None:
    """Review a TopWord card with SM-2 scheduling.

    If the card has never been answered before (success_count <= 0), this also
    counts it as a newly introduced card for the daily new-card quota.
    """
    session: Session = get_session()
    card: Optional[TopWord] = session.get(TopWord, card_id)
    if card:
        is_first_review = card.success_count <= 0

        new_interval, new_ease, new_reps, next_review = sm2_schedule(
            card.interval, card.ease_factor, card.repetitions, quality
        )
        card.interval = new_interval
        card.ease_factor = new_ease
        card.repetitions = new_reps
        card.next_review = next_review
        if quality >= 3:
            card.success_count = max(card.success_count, 0) + 1
        else:
            card.success_count = max(card.success_count, 1)

        if is_first_review:
            now = datetime.datetime.now(datetime.UTC)
            settings = session.query(UserSettings).filter_by(user=user).first()
            if settings:
                if now.date() != settings.last_reset.date():
                    settings.last_reset = now
                    settings.new_cards_today = 0
                settings.new_cards_today += 1

        session.commit()
    session.close()


def annotate_top_word(card_id: int, reading: str, meanings_en: str,
                      pos_tags: str = "", jlpt_level: str = "") -> None:
    """Store LLM-generated annotations for a word card."""
    session: Session = get_session()
    card: Optional[TopWord] = session.get(TopWord, card_id)
    if card:
        card.reading = reading
        card.meanings_en = meanings_en
        card.pos_tags = pos_tags
        card.jlpt_level = jlpt_level
        card.annotated = True
        session.commit()
    session.close()


def mark_top_word_lesson_viewed(card_id: int) -> None:
    """Mark a TopWord card's lesson as viewed."""
    session: Session = get_session()
    card: Optional[TopWord] = session.get(TopWord, card_id)
    if card:
        card.lesson_viewed = True
        session.commit()
    session.close()


def get_top_word_progress(user: str = "default_user") -> Dict[str, Any]:
    """Get Top Words learning progress stats."""
    session: Session = get_session()
    now = datetime.datetime.now(datetime.UTC)
    total = session.query(TopWord).count()
    learned = session.query(TopWord).filter(TopWord.success_count > 0).count()
    unseen = session.query(TopWord).filter(TopWord.success_count == 0).count()
    due_now = session.query(TopWord).filter(
        TopWord.success_count > 0, TopWord.next_review <= now
    ).count()
    annotated = session.query(TopWord).filter(TopWord.annotated == True).count()
    next_card = (
        session.query(TopWord)
        .filter(TopWord.success_count == 0)
        .order_by(TopWord.rank.asc())
        .first()
    )
    session.close()
    return {
        "total": total, "learned": learned, "unseen": unseen,
        "due_now": due_now, "annotated": annotated,
        "next_word": next_card.lemma if next_card else None,
        "next_rank": next_card.rank if next_card else None,
    }


def get_word_distractors(card: TopWord, count: int = 3) -> List[Dict[str, Any]]:
    """Get distractor words for MC questions. Prefers annotated words."""
    import random
    session: Session = get_session()
    candidates = (
        session.query(TopWord)
        .filter(TopWord.id != card.id, TopWord.annotated == True)
        .all()
    )
    session.close()
    if not candidates:
        return []
    selected = random.sample(candidates, min(count, len(candidates)))
    return [{"lemma": c.lemma, "reading": c.reading,
             "meanings_en": c.meanings_en} for c in selected]
