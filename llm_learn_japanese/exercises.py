import random
import json
import os
from typing import Any, Dict, List
from . import db

DEBUG_MODE = os.getenv("DEBUG", "0") == "1"


def generate_exercise(aspect: Any, model: Any) -> str:
    """
    Generate exercise prompts dynamically using AI.
    Model parameter is REQUIRED - no fallback to templates.

    Args:
        aspect: CardAspect database object containing learning content
        model: Required LLM model for AI-based generation

    Returns:
        A dynamically generated exercise prompt

    Raises:
        ValueError: If no model is provided
    """
    if model is None:
        raise ValueError("AI model is required for exercise generation. Please ensure OpenAI API key is configured.")

    # Get context information for better exercise generation
    context = _get_aspect_context(aspect)

    # Generate AI-powered exercise based on aspect type
    exercise_prompt = _generate_ai_exercise(model, aspect, context)

    return exercise_prompt


def _get_aspect_context(aspect: Any) -> Dict[str, Any]:
    """Retrieve full context for an aspect from the database."""
    if DEBUG_MODE:
        print(f"🔍 Retrieving context for aspect:")
        print(f"   Aspect ID: {aspect.id}")
        print(f"   Parent type: {aspect.parent_type}")
        print(f"   Parent ID: {aspect.parent_id}")
        print(f"   Aspect type: {aspect.aspect_type}")

    session = db.get_session()
    context = {
        "aspect_type": aspect.aspect_type,
        "parent_type": aspect.parent_type,
        "success_count": aspect.success_count,
        "interval": aspect.interval
    }

    # Get parent content based on type
    if aspect.parent_type == "vocabulary":
        vocab = session.get(db.Vocabulary, aspect.parent_id)
        if vocab:
            context["word"] = vocab.word
            context["reading"] = vocab.reading
            context["meaning"] = vocab.meaning
            if DEBUG_MODE:
                print(f"   Retrieved vocabulary: {vocab.word} (reading: {vocab.reading}, meaning: {vocab.meaning})")
        else:
            print(f"❌ No vocabulary found for parent_id: {aspect.parent_id}")
    elif aspect.parent_type == "kanji":
        kanji = session.get(db.Kanji, aspect.parent_id)
        if kanji:
            context["character"] = kanji.character
            context["onyomi"] = kanji.onyomi
            context["kunyomi"] = kanji.kunyomi
            context["meanings"] = kanji.meanings
            if DEBUG_MODE:
                print(f"   Retrieved kanji: {kanji.character} (onyomi: {kanji.onyomi}, kunyomi: {kanji.kunyomi})")
        else:
            print(f"❌ No kanji found for parent_id: {aspect.parent_id}")
    elif aspect.parent_type == "grammar":
        grammar = session.get(db.Grammar, aspect.parent_id)
        if grammar:
            context["point"] = grammar.point
            context["explanation"] = grammar.explanation
            context["example"] = grammar.example
            if DEBUG_MODE:
                print(f"   Retrieved grammar: {grammar.point}")
        else:
            print(f"❌ No grammar found for parent_id: {aspect.parent_id}")
    elif aspect.parent_type == "phrase":
        phrase = session.get(db.Phrase, aspect.parent_id)
        if phrase:
            context["phrase"] = phrase.phrase
            context["meaning"] = phrase.meaning
            if DEBUG_MODE:
                print(f"   Retrieved phrase: {phrase.phrase}")
        else:
            print(f"❌ No phrase found for parent_id: {aspect.parent_id}")
    elif aspect.parent_type == "idiom":
        idiom = session.get(db.Idiom, aspect.parent_id)
        if idiom:
            context["idiom"] = idiom.idiom
            context["meaning"] = idiom.meaning
            context["example"] = idiom.example
            if DEBUG_MODE:
                print(f"   Retrieved idiom: {idiom.idiom}")
        else:
            print(f"❌ No idiom found for parent_id: {aspect.parent_id}")

    session.close()
    print(f"✅ Context retrieval complete: {len(context)} fields")
    return context


def _generate_ai_exercise(model: Any, aspect: Any, context: Dict[str, Any]) -> str:
    """Generate an AI-powered exercise based on aspect type and context."""

    # Determine difficulty based on success history
    difficulty_level = _calculate_difficulty_level(aspect)
    print(f"🎯 Exercise Generation: difficulty={difficulty_level}, aspect={aspect.aspect_type}, parent={context['parent_type']}")

    exercise_types = _get_exercise_types_for_aspect(aspect.aspect_type, context["parent_type"])
    selected_type = random.choice(exercise_types)
    if DEBUG_MODE:
        print(f"   → Available types: {exercise_types}")
    print(f"   → Selected exercise type: {selected_type}")

    system_prompt = """You are a Japanese language learning assistant. Generate a single exercise question
    for the student based on the provided context. The exercise should be clear, educational, and
    appropriately challenging. Return ONLY the exercise question, nothing else.
    DO NOT PUT THE ANSWER IN THE EXERCISE, the student must figure it out on their own!
    Ensure that the exercise is short enough that the user can easily answer it when
    typing on their smartphone, but that is also causes the student to demonstrate
    knowledge of the concept. Ensure that the question is comprehensible by an
    English native speaker (do not write the entirety in Japanese)."""

    prompt = _build_generation_prompt(context, selected_type, difficulty_level)

    if DEBUG_MODE:
        print(f"   Generated prompt length: {len(prompt)} characters")
        print(f"   Prompt preview: {prompt[:200]}...")
        print("🤖 OpenAI API Call Details:")
        print(f"   Model: {getattr(model, 'name', 'unknown')}")
        print(f"   System prompt length: {len(system_prompt)} characters")
        print(f"   User prompt length: {len(prompt)} characters")
    # In normal mode, suppress these OpenAI API debug details

    try:
        response = model.prompt(prompt, system=system_prompt)
        exercise = response.text().strip()
        if DEBUG_MODE:
            print(f"✅ Full Exercise: {exercise}")
        else:
            print(f"✅ Exercise created: {exercise[:80]}{'...' if len(exercise) > 80 else ''}")

        if len(exercise) < 10:
            print(f"❌ AI generated insufficient exercise content: '{exercise}' (length: {len(exercise)})")
            raise ValueError("AI generated insufficient exercise content")

        if DEBUG_MODE:
            print(f"✅ Exercise validation passed, length: {len(exercise)} characters")
        return str(exercise)

    except Exception as e:
        print(f"❌ Exercise generation failed: {str(e)} ({type(e).__name__})")
        raise


def _calculate_difficulty_level(aspect: Any) -> str:
    """Calculate appropriate difficulty based on user's performance."""
    if aspect.success_count < 3:
        return "beginner"
    elif aspect.success_count < 10:
        return "intermediate"
    else:
        return "advanced"


def _get_exercise_types_for_aspect(aspect_type: str, parent_type: str) -> List[str]:
    """Get appropriate exercise types for a given aspect."""
    exercise_map = {
        ("vocabulary", "meaning"): [
            "direct_translation",
            "multiple_choice",
            "fill_in_blank",
            "context_sentence"
        ],
        ("vocabulary", "reading"): [
            "write_reading",
            "choose_correct_reading",
            "reading_from_context"
        ],
        ("vocabulary", "usage"): [
            "create_sentence",
            "complete_dialogue",
            "appropriate_context"
        ],
        ("kanji", "meaning"): [
            "kanji_to_meaning",
            "meaning_to_kanji",
            "compound_meaning"
        ],
        ("kanji", "onyomi"): [
            "write_onyomi",
            "identify_onyomi_word",
            "onyomi_compound"
        ],
        ("kanji", "kunyomi"): [
            "write_kunyomi",
            "identify_kunyomi_word",
            "kunyomi_usage"
        ],
        ("kanji", "usage"): [
            "kanji_in_sentence",
            "compound_creation",
            "reading_choice"
        ],
        ("grammar", "explanation"): [
            "explain_usage",
            "identify_pattern",
            "error_correction"
        ],
        ("grammar", "usage"): [
            "create_example",
            "transform_sentence",
            "complete_with_grammar"
        ],
        ("phrase", "meaning"): [
            "translate_phrase",
            "situational_usage",
            "cultural_context"
        ],
        ("phrase", "usage"): [
            "dialogue_completion",
            "appropriate_response",
            "role_play_scenario"
        ],
        ("idiom", "meaning"): [
            "literal_vs_figurative",
            "explain_meaning",
            "match_situation"
        ],
        ("idiom", "usage"): [
            "use_in_context",
            "identify_appropriate_situation",
            "create_dialogue"
        ]
    }

    key = (parent_type, aspect_type)
    return exercise_map.get(key, ["default_question"])


def _build_generation_prompt(context: Dict[str, Any], exercise_type: str, difficulty: str) -> str:
    """Build a detailed prompt for AI exercise generation."""

    base_info = f"""
    Content Type: {context['parent_type']}
    Aspect: {context['aspect_type']}
    Exercise Type: {exercise_type}
    Difficulty: {difficulty}
    Student has reviewed this {context['success_count']} times
    """

    # Add content-specific information
    if context['parent_type'] == "vocabulary":
        content_info = f"""
        Word: {context.get('word', '')}
        Reading: {context.get('reading', 'not provided')}
        Meaning: {context.get('meaning', 'not provided')}
        """
    elif context['parent_type'] == "kanji":
        content_info = f"""
        Character: {context.get('character', '')}
        Onyomi: {context.get('onyomi', 'not provided')}
        Kunyomi: {context.get('kunyomi', 'not provided')}
        Meanings: {context.get('meanings', 'not provided')}
        """
    elif context['parent_type'] == "grammar":
        content_info = f"""
        Grammar Point: {context.get('point', '')}
        Explanation: {context.get('explanation', 'not provided')}
        Example: {context.get('example', 'not provided')}
        """
    elif context['parent_type'] == "phrase":
        content_info = f"""
        Phrase: {context.get('phrase', '')}
        Meaning: {context.get('meaning', 'not provided')}
        """
    elif context['parent_type'] == "idiom":
        content_info = f"""
        Idiom: {context.get('idiom', '')}
        Meaning: {context.get('meaning', 'not provided')}
        Example: {context.get('example', 'not provided')}
        """
    else:
        content_info = "Content details not available"

    # Exercise-specific instructions
    exercise_instructions = _get_exercise_instructions(exercise_type, difficulty)

    return f"""{base_info}

    Content Details:
    {content_info}

    Instructions:
    {exercise_instructions}

    Generate a single, clear exercise question for the student. Make it {difficulty} level appropriate.

    Remember, the student is an English native speaker, so ensure that the user can read your question, even if you mix Japanese and English.
    """


def _get_exercise_instructions(exercise_type: str, difficulty: str) -> str:
    """Get specific instructions for each exercise type."""
    instructions = {
        "direct_translation": "Ask for the English meaning of the Japanese word",
        "multiple_choice": "Create a multiple choice question with 4 options (mark the correct one)",
        "fill_in_blank": "Create a sentence with a blank to fill in",
        "context_sentence": "Ask for the meaning based on a sentence using the word",
        "write_reading": "Ask for the hiragana/katakana reading",
        "choose_correct_reading": "Multiple choice for the correct reading",
        "reading_from_context": "Provide a sentence and ask for the reading of the highlighted word",
        "create_sentence": "Ask the student to create an original sentence",
        "complete_dialogue": "Provide a dialogue with a missing part to complete",
        "appropriate_context": "Ask when/where to use this expression",
        "kanji_to_meaning": "Ask for the meaning(s) of the kanji",
        "meaning_to_kanji": "Give meaning and ask for the kanji",
        "compound_meaning": "Ask about a compound word using this kanji",
        "write_onyomi": "Ask for the Chinese reading (onyomi)",
        "identify_onyomi_word": "Ask to identify a word using the onyomi reading",
        "onyomi_compound": "Ask about compound words using onyomi",
        "write_kunyomi": "Ask for the Japanese reading (kunyomi)",
        "identify_kunyomi_word": "Ask to identify a word using the kunyomi reading",
        "kunyomi_usage": "Ask for examples using kunyomi",
        "kanji_in_sentence": "Ask to use the kanji in a sentence",
        "compound_creation": "Ask to create compound words",
        "reading_choice": "Multiple choice for correct reading in context",
        "explain_usage": "Ask to explain when/how to use this grammar",
        "identify_pattern": "Provide sentences and ask to identify the grammar pattern",
        "error_correction": "Provide incorrect sentence to correct",
        "create_example": "Ask for an original example using the grammar",
        "transform_sentence": "Ask to transform a sentence using the grammar",
        "complete_with_grammar": "Complete a sentence using the grammar point",
        "translate_phrase": "Ask for the translation of the phrase",
        "situational_usage": "Ask when this phrase would be used",
        "cultural_context": "Ask about cultural significance or context",
        "dialogue_completion": "Complete a dialogue using the phrase",
        "appropriate_response": "Choose/provide appropriate response using the phrase",
        "role_play_scenario": "Create a scenario to use the phrase",
        "literal_vs_figurative": "Explain literal vs figurative meaning",
        "explain_meaning": "Explain what the idiom means",
        "match_situation": "Match the idiom to appropriate situations",
        "use_in_context": "Use the idiom in an appropriate context",
        "identify_appropriate_situation": "Identify when to use the idiom",
        "create_dialogue": "Create a dialogue using the idiom",
        "default_question": "Ask about this content in an appropriate way"
    }

    base_instruction = instructions.get(exercise_type, instructions["default_question"])

    # Add difficulty modifiers
    if difficulty == "beginner":
        base_instruction += ". Include helpful hints or context clues."
    elif difficulty == "advanced":
        base_instruction += ". Make it challenging with nuanced usage or less common contexts."

    return base_instruction


# ----------------------------------------------------------------------
# Bunpro Grammar Multiple-Choice Exercise Generation
# ----------------------------------------------------------------------

BUNPRO_QUESTION_TYPES = [
    "meaning_match",
    "grammar_identification",
    "fill_in_blank",
    "english_to_grammar",
    "correct_usage",
    "nuance_context",
]

BUNPRO_BEGINNER_TYPES = ["meaning_match", "english_to_grammar", "grammar_identification"]
BUNPRO_INTERMEDIATE_TYPES = BUNPRO_QUESTION_TYPES  # all types
BUNPRO_ADVANCED_TYPES = ["correct_usage", "nuance_context", "fill_in_blank", "grammar_identification"]


def _pick_bunpro_question_type(success_count: int) -> str:
    """Pick a question type based on how well the user knows this card."""
    if success_count < 3:
        pool = BUNPRO_BEGINNER_TYPES
    elif success_count < 8:
        pool = BUNPRO_INTERMEDIATE_TYPES
    else:
        pool = BUNPRO_ADVANCED_TYPES
    return random.choice(pool)


def generate_bunpro_quiz(
    card: Any,
    distractors: List[Dict[str, Any]],
    model: Any,
) -> List[Dict[str, Any]]:
    """Generate a 3-question quiz for a Bunpro grammar card.

    Question types:
      1. meaning_match — identify the English meaning of the grammar point
      2. sentence_translation — translate a Japanese sentence that uses this grammar (J→E)
      3. correct_usage — identify which sentence correctly uses the grammar point

    Args:
        card: BunproGrammar database object
        distractors: List of distractor dicts with keys: grammar, meaning, jlpt
        model: LLM model for generation

    Returns:
        List of 3 Dicts, each with keys: question, question_type, choices (list of {key, text}), correct_key
    """
    if model is None:
        raise ValueError("AI model is required for Bunpro quiz generation.")

    system_prompt = """You are a Japanese grammar quiz generator. Generate exactly 3 distinct multiple-choice questions
for the provided grammar point. Each question must have exactly 4 answer choices (A, B, C, D).

Return ONLY valid JSON in this format:
{
  "questions": [
    {
      "question_type": "meaning_match",
      "question": "The question text",
      "choices": [
        {"key": "A", "text": "Choice A text"},
        {"key": "B", "text": "Choice B text"},
        {"key": "C", "text": "Choice C text"},
        {"key": "D", "text": "Choice D text"}
      ],
      "correct_key": "B"
    },
    ... (2 more questions, 3 total)
  ]
}

RULES:
- Randomize the position of the correct answer for each question.
- Ensure questions are comprehensible by an English-speaking Japanese learner.
- Make distractors plausible but clearly wrong.
- Keep questions concise and clear.
- Sentences must be natural Japanese that a native speaker would actually say.
"""

    distractor_info = "\n".join(
        f"- {d['grammar']}: {d['meaning']}" for d in distractors
    )

    prompt = f"""Target grammar point: {card.grammar}
Meaning: {card.meaning}
JLPT Level: {card.jlpt}
Register: {card.register}

Other grammar points available as distractors:
{distractor_info}

Generate exactly 3 questions, one for each type:

1. meaning_match: Show a SHORT example sentence in Japanese that uses 「{card.grammar}」 with the meaning "{card.meaning}",
   followed by its English translation. Then ask "In this context, what does 「{card.grammar}」 mean?"
   This is important because some grammar points (like が, は, で) have multiple meanings depending on context.
   The example sentence should clearly demonstrate the specific meaning "{card.meaning}".
   Choices are English meanings. Use the correct meaning and 3 plausible distractor meanings
   (from the distractor grammar points or similar-sounding alternatives).

2. sentence_translation: Create a SHORT, natural Japanese sentence (8-15 words) that uses 「{card.grammar}」.
   Ask "Choose the correct English translation of this sentence:"
   Show the Japanese sentence in the question text.
   Choices are 4 English translations — only one is correct.
   The correct translation should clearly demonstrate understanding of what 「{card.grammar}」 means.
   The wrong translations should be plausible but mistranslate the part involving 「{card.grammar}」.

3. correct_usage: Show 4 Japanese sentences (with English translations in parentheses).
   Only ONE correctly uses 「{card.grammar}」.
   The other 3 should have subtle errors: wrong conjugation, wrong context, wrong word order,
   or using the grammar in a situation where it doesn't apply.
   Ask "Which sentence correctly uses 「{card.grammar}」?"

Return ONLY valid JSON with a "questions" array containing exactly 3 objects.
"""

    if DEBUG_MODE:
        print(f"🎯 Bunpro Quiz Generation: grammar={card.grammar}")

    try:
        response = model.prompt(prompt, system=system_prompt)
        raw = response.text().strip()

        # Strip markdown code fences if present
        if raw.startswith("```"):
            lines = raw.split("\n")
            lines = [l for l in lines if not l.strip().startswith("```")]
            raw = "\n".join(lines)

        result = json.loads(raw)
        questions = result.get("questions", [])

        if len(questions) < 3:
            print(f"⚠️ Expected 3 questions, got {len(questions)}. Padding with fallbacks.")
            while len(questions) < 3:
                questions.append(_bunpro_fallback_question(card, distractors))

        # Validate structure of each question
        for q in questions:
            if "question" not in q or "choices" not in q or "correct_key" not in q:
                pass
            if "question_type" not in q:
                q["question_type"] = "unknown"

        if DEBUG_MODE:
            print(f"✅ Bunpro quiz generated with {len(questions)} questions.")

        return questions[:3]

    except Exception as e:
        print(f"❌ Bunpro quiz generation failed: {e}")
        fallback = _bunpro_fallback_question(card, distractors)
        return [fallback] * 3


def generate_bunpro_exercise(
    card: Any,
    distractors: List[Dict[str, Any]],
    model: Any,
) -> Dict[str, Any]:
    """Legacy wrapper for single question generation (keeps backward compatibility if needed)."""
    quiz = generate_bunpro_quiz(card, distractors, model)
    return quiz[0] if quiz else _bunpro_fallback_question(card, distractors)


def _build_bunpro_prompt(card: Any, question_type: str, distractor_info: str) -> str:
    """Build the prompt for a specific Bunpro question type."""

    base_context = f"""
Target grammar point: {card.grammar}
Meaning: {card.meaning}
JLPT Level: {card.jlpt}
Register: {card.register}

Other grammar points available as distractors:
{distractor_info}
"""

    type_instructions = {
        "meaning_match": f"""Create a "What does this grammar point mean?" question.

Show the grammar point 「{card.grammar}」 and ask the student to pick the correct English meaning.
Use the correct meaning as one choice and create 3 plausible but incorrect meanings
using the distractor grammar points' meanings or similar-sounding alternatives.""",

        "grammar_identification": f"""Create a sentence in Japanese that uses 「{card.grammar}」, then ask
"Which grammar point is used in this sentence?" The choices should be the target grammar
point and 3 distractor grammar points. The sentence should make the grammar point's usage
clear but not trivially obvious.""",

        "fill_in_blank": f"""Create a Japanese sentence with a blank (＿＿＿) where 「{card.grammar}」
should go. Ask the student to pick the correct grammar to fill in the blank.
The 3 wrong choices should be grammatically plausible but incorrect in context.
Include the English translation of the complete sentence as a hint.""",

        "english_to_grammar": f"""Give an English sentence or intention (like "I want to express: because/since...")
and ask which Japanese grammar point achieves that meaning.
The correct answer is 「{card.grammar}」 ({card.meaning}).
Distractors should be grammar points with related but different meanings.""",

        "correct_usage": f"""Show 4 Japanese sentences. Only ONE correctly uses 「{card.grammar}」.
The other 3 should have subtle errors: wrong conjugation before the grammar point,
wrong word order, or using the grammar in an impossible context.
Include brief English translations for each sentence.""",

        "nuance_context": f"""Ask about the appropriate usage context, register, or nuance of 「{card.grammar}」.
For example: "When would you most naturally use 「{card.grammar}」?" or
"What is the difference between 「{card.grammar}」 and [similar grammar]?"
Focus on practical understanding of when and why to use this grammar.""",
    }

    instruction = type_instructions.get(question_type, type_instructions["meaning_match"])

    return f"""{base_context}

Question Type: {question_type}

{instruction}

Remember: Return ONLY valid JSON with question, choices (4 items with key and text), and correct_key."""


def _bunpro_fallback_question(card: Any, distractors: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Generate a simple deterministic fallback question if AI fails."""
    choices = [{"key": "A", "text": card.meaning or "Unknown meaning"}]

    for i, d in enumerate(distractors[:3]):
        key = chr(ord("B") + i)
        choices.append({"key": key, "text": d.get("meaning", f"Distractor {i+1}")})

    # Shuffle choices and track correct answer
    correct_text = choices[0]["text"]
    random.shuffle(choices)
    # Re-assign keys after shuffle
    correct_key = "A"
    for i, c in enumerate(choices):
        c["key"] = chr(ord("A") + i)
        if c["text"] == correct_text:
            correct_key = c["key"]

    return {
        "question": f"What does 「{card.grammar}」 mean?",
        "question_type": "meaning_match",
        "choices": choices,
        "correct_key": correct_key,
    }


def generate_bunpro_lesson(card: Any, model: Any) -> Dict[str, Any]:
    """Generate an AI lesson for a Bunpro grammar point.

    Returns a structured lesson dict with explanation, examples, tips, etc.
    """
    if model is None:
        raise ValueError("AI model is required for lesson generation.")

    system_prompt = """You are a Japanese grammar teacher. Generate a concise but thorough
lesson about a grammar point for an English native language speaker who is learning Japanese. Return ONLY valid JSON in this format:
{
  "grammar_point": "the grammar in Japanese",
  "meaning": "English meaning",
  "jlpt_level": "N5",
  "explanation": "Clear explanation of what this grammar does and when to use it (2-3 sentences)",
  "formation": "How to form/conjugate with this grammar (e.g. Verb-て form + いる)",
  "examples": [
    {"japanese": "日本語の例文", "english": "English translation"},
    {"japanese": "もう一つの例文", "english": "Another translation"}
  ],
  "tips": "Practical tips for remembering or using this grammar correctly",
  "comparison": "Brief comparison with similar grammar points (if applicable)"
}

Keep it concise and practical. Target an English-speaking learner."""

    prompt = f"""Generate a lesson for this grammar point:

Grammar: {card.grammar}
Meaning: {card.meaning}
JLPT Level: {card.jlpt}
Register: {card.register}
Bunpro URL: {card.url}

Create 2-3 natural example sentences at an appropriate difficulty level for {card.jlpt}.
Focus on practical, everyday usage."""

    try:
        response = model.prompt(prompt, system=system_prompt)
        raw = response.text().strip()

        # Strip markdown code fences if present
        if raw.startswith("```"):
            lines = raw.split("\n")
            lines = [l for l in lines if not l.strip().startswith("```")]
            raw = "\n".join(lines)

        lesson = json.loads(raw)

        # Ensure required fields
        lesson.setdefault("grammar_point", card.grammar)
        lesson.setdefault("meaning", card.meaning)
        lesson.setdefault("jlpt_level", card.jlpt)
        lesson.setdefault("url", card.url)

        return lesson

    except Exception as e:
        print(f"❌ Bunpro lesson generation failed: {e}")
        # Return a minimal fallback lesson
        return {
            "grammar_point": card.grammar,
            "meaning": card.meaning,
            "jlpt_level": card.jlpt,
            "explanation": f"{card.grammar} means '{card.meaning}'. It is a {card.jlpt} level grammar point.",
            "formation": "See Bunpro for formation details.",
            "examples": [],
            "tips": f"This is ranked #{card.rank} by usage frequency.",
            "comparison": "",
            "url": card.url,
        }


def generate_bunpro_feedback(card: Any, is_correct: bool, selected_text: str, correct_text: str, model: Any) -> str:
    """Generate brief AI feedback after answering a Bunpro MC question."""
    if model is None:
        if is_correct:
            return f"Correct! 「{card.grammar}」 means '{card.meaning}'."
        else:
            return f"Not quite. 「{card.grammar}」 means '{card.meaning}'. You selected: {selected_text}"

    try:
        system_prompt = "You are a Japanese grammar tutor. Give a brief (1-2 sentence) feedback response **predominantly in English**. Be encouraging."

        if is_correct:
            prompt = f"""The student correctly identified that 「{card.grammar}」 means '{card.meaning}'.
Give a brief congratulatory response that reinforces the grammar point.
Maybe add a quick usage tip or example. Keep it to 1-2 sentences."""
        else:
            prompt = f"""The student incorrectly answered a question about 「{card.grammar}」 ({card.meaning}).
They selected "{selected_text}" but the correct answer was "{correct_text}".
Give a brief corrective response that helps them remember the right answer.
Keep it to 1-2 sentences. Be encouraging."""

        response = model.prompt(prompt, system=system_prompt)
        return response.text().strip()
    except Exception:
        if is_correct:
            return f"Correct! 「{card.grammar}」 means '{card.meaning}'."
        else:
            return f"The correct answer was: {correct_text}. 「{card.grammar}」 means '{card.meaning}'."


# ======================================================================
# Top Kanji — LLM Annotation + MC Questions
# ======================================================================

KANJI_QUESTION_TYPES = ["meaning_match", "on_reading_match", "kun_reading_match", "kanji_from_meaning", "kanji_in_context"]
KANJI_BEGINNER_TYPES = ["meaning_match", "kanji_from_meaning"]
KANJI_ADVANCED_TYPES = ["on_reading_match", "kun_reading_match", "kanji_in_context"]


def annotate_kanji_with_llm(card: Any, model: Any) -> Dict[str, str]:
    """Call LLM to populate a kanji card's annotations. Returns the annotation dict."""
    system_prompt = "You are a Japanese kanji dictionary. Return ONLY valid JSON."
    prompt = f"""Given the kanji character 「{card.kanji}」 (frequency rank #{card.rank}), provide its standard dictionary information.

Return JSON:
{{
  "on_readings": "comma-separated onyomi in katakana",
  "kun_readings": "comma-separated kunyomi in hiragana (with okurigana markers like まな.ぶ)",
  "meanings_en": "comma-separated English meanings (most common 2-4 meanings)",
  "jlpt_level": "N5/N4/N3/N2/N1 or empty if unknown"
}}"""

    try:
        response = model.prompt(prompt, system=system_prompt)
        raw = response.text().strip()
        if raw.startswith("```"):
            lines = raw.split("\n")
            lines = [l for l in lines if not l.strip().startswith("```")]
            raw = "\n".join(lines)
        data = json.loads(raw)
        return {
            "on_readings": str(data.get("on_readings", "")),
            "kun_readings": str(data.get("kun_readings", "")),
            "meanings_en": str(data.get("meanings_en", "")),
            "jlpt_level": str(data.get("jlpt_level", "")),
        }
    except Exception as e:
        print(f"⚠️ Kanji annotation failed for {card.kanji}: {e}")
        return {"on_readings": "", "kun_readings": "", "meanings_en": "", "jlpt_level": ""}


def generate_kanji_quiz(card: Any, distractors: List[Dict[str, Any]], model: Any) -> List[Dict[str, Any]]:
    """Generate a 3-question quiz for a Top Kanji card.

    Question types:
      1. meaning_match — identify the English meaning of the kanji
      2. reading_match — identify a correct on/kun reading
      3. compound_word — given a real compound word containing this kanji, pick its meaning
    """
    if model is None:
        raise ValueError("AI model required")

    system_prompt = """You are a Japanese kanji quiz generator. Generate exactly 3 distinct multiple-choice questions
for the provided kanji. Each question must have exactly 4 answer choices (A, B, C, D).

Return ONLY valid JSON in this format:
{
  "questions": [
    {
      "question_type": "meaning_match",
      "question": "...",
      "choices": [{"key": "A", "text": "..."}, {"key": "B", "text": "..."}, {"key": "C", "text": "..."}, {"key": "D", "text": "..."}],
      "correct_key": "B"
    },
    ... (2 more questions, 3 total)
  ]
}

RULES:
- Randomize the position of the correct answer for each question.
- Make distractors plausible but clearly wrong.
- All questions must be comprehensible by an English-speaking Japanese learner."""

    distractor_info = "\n".join(
        f"- {d.get('kanji','?')}: meanings={d.get('meanings_en','?')}, on={d.get('on_readings','?')}, kun={d.get('kun_readings','?')}"
        for d in distractors
    )

    prompt = f"""Target kanji: {card.kanji}
Meanings: {card.meanings_en}
On readings: {card.on_readings}
Kun readings: {card.kun_readings}

Distractors:
{distractor_info}

Generate exactly 3 questions, one for each type:

1. meaning_match: Ask "What does the kanji 「{card.kanji}」 mean?"
   Choices are English meanings. Use the correct meaning and 3 distractor meanings.

2. reading_match: Ask "What is a reading of 「{card.kanji}」?"
   Choices are readings in hiragana/katakana. Include one correct reading and 3 plausible wrong readings from the distractors.

3. compound_word: Think of a real, common Japanese compound word that contains 「{card.kanji}」.
   Ask "What does the word 「<compound>」 mean?" where <compound> is a 2-3 character word using 「{card.kanji}」.
   Choices are English meanings. The correct answer is the compound word's meaning.
   The compound word MUST be a real word that Japanese speakers actually use.

Return ONLY valid JSON with a "questions" array containing exactly 3 objects."""

    try:
        response = model.prompt(prompt, system=system_prompt)
        raw = response.text().strip()
        if raw.startswith("```"):
            lines = raw.split("\n")
            lines = [l for l in lines if not l.strip().startswith("```")]
            raw = "\n".join(lines)
        result = json.loads(raw)
        questions = result.get("questions", [])
        if len(questions) < 3:
            print(f"⚠️ Expected 3 kanji questions, got {len(questions)}. Padding.")
            while len(questions) < 3:
                questions.append(_kanji_fallback_question(card, distractors))
        return questions[:3]
    except Exception as e:
        print(f"❌ Kanji quiz generation failed: {e}")
        fallback = _kanji_fallback_question(card, distractors)
        return [fallback] * 3


def generate_kanji_mc_exercise(card: Any, distractors: List[Dict[str, Any]], model: Any) -> Dict[str, Any]:
    """Legacy wrapper for single question generation."""
    quiz = generate_kanji_quiz(card, distractors, model)
    return quiz[0] if quiz else _kanji_fallback_question(card, distractors)


def _kanji_fallback_question(card: Any, distractors: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Deterministic fallback for kanji MC question."""
    choices = [{"key": "A", "text": card.meanings_en or "unknown"}]
    for i, d in enumerate(distractors[:3]):
        choices.append({"key": chr(ord("B") + i), "text": d.get("meanings_en", f"meaning {i+1}")})
    correct_text = choices[0]["text"]
    random.shuffle(choices)
    correct_key = "A"
    for i, c in enumerate(choices):
        c["key"] = chr(ord("A") + i)
        if c["text"] == correct_text:
            correct_key = c["key"]
    return {"question": f"What does 「{card.kanji}」 mean?", "question_type": "meaning_match",
            "choices": choices, "correct_key": correct_key}


def generate_kanji_lesson(card: Any, model: Any) -> Dict[str, Any]:
    """Generate an AI lesson for a kanji character."""
    if model is None:
        raise ValueError("AI model required")
    system_prompt = "You are a Japanese kanji teacher. Return ONLY valid JSON."
    prompt = f"""Generate a lesson for the kanji 「{card.kanji}」 for an English native language speaker who is learning Japanese.
On readings: {card.on_readings}
Kun readings: {card.kun_readings}
Meanings: {card.meanings_en}

Return JSON:
{{
  "kanji": "{card.kanji}",
  "meanings": "{card.meanings_en}",
  "on_readings": "{card.on_readings}",
  "kun_readings": "{card.kun_readings}",
  "explanation": "Clear explanation of this kanji (2-3 sentences)",
  "formation": "Describe the radical composition or mnemonics",
  "examples": [{{"japanese": "word using this kanji", "reading": "reading", "english": "meaning"}}],
  "tips": "Tips for remembering this kanji"
}}"""
    try:
        response = model.prompt(prompt, system=system_prompt)
        raw = response.text().strip()
        if raw.startswith("```"):
            lines = raw.split("\n")
            lines = [l for l in lines if not l.strip().startswith("```")]
            raw = "\n".join(lines)
        lesson = json.loads(raw)
        lesson.setdefault("kanji", card.kanji)
        lesson.setdefault("meanings", card.meanings_en)
        return lesson
    except Exception as e:
        print(f"❌ Kanji lesson failed: {e}")
        return {"kanji": card.kanji, "meanings": card.meanings_en, "on_readings": card.on_readings,
                "kun_readings": card.kun_readings,
                "explanation": f"{card.kanji} means '{card.meanings_en}'.",
                "formation": "", "examples": [], "tips": f"Ranked #{card.rank} by frequency."}


def generate_kanji_feedback(card: Any, is_correct: bool, selected_text: str, correct_text: str, model: Any) -> str:
    """Brief AI feedback after a kanji MC answer."""
    if model is None:
        return f"{'Correct!' if is_correct else 'Incorrect.'} 「{card.kanji}」 means '{card.meanings_en}'."
    try:
        prompt = f"The student {'correctly' if is_correct else 'incorrectly'} answered about kanji 「{card.kanji}」 ({card.meanings_en}). " + \
                 (f"They selected '{selected_text}' but correct was '{correct_text}'. " if not is_correct else "") + \
                 "Give 1-2 sentence feedback **predominantly in English**."
        response = model.prompt(prompt, system="You are a Japanese kanji tutor. Be encouraging and concise.")
        return response.text().strip()
    except Exception:
        return f"{'Correct!' if is_correct else 'Incorrect.'} 「{card.kanji}」 means '{card.meanings_en}'."


# ======================================================================
# Top Words — LLM Annotation + MC Questions
# ======================================================================

WORD_QUESTION_TYPES = ["meaning_match", "reading_match", "word_from_meaning", "fill_in_blank", "usage_context"]
WORD_BEGINNER_TYPES = ["meaning_match", "word_from_meaning"]
WORD_ADVANCED_TYPES = ["reading_match", "fill_in_blank", "usage_context"]


def annotate_word_with_llm(card: Any, model: Any) -> Dict[str, str]:
    """Call LLM to populate a word card's annotations. Returns the annotation dict."""
    system_prompt = "You are a Japanese dictionary. Return ONLY valid JSON."
    prompt = f"""Given the Japanese word 「{card.lemma}」 (frequency rank #{card.rank} — this is one of the most commonly used words in Japanese):

Provide its PRIMARY, most common meaning as used in everyday Japanese.
IGNORE rare/archaic/homophone meanings. Focus on what a beginner needs to know.

Return JSON:
{{
  "reading": "the most common reading in hiragana",
  "meanings_en": "2-4 most common English meanings, comma-separated",
  "pos_tags": "part of speech (e.g. verb, noun, adjective, particle, adverb)",
  "jlpt_level": "N5/N4/N3/N2/N1 or empty"
}}"""

    try:
        response = model.prompt(prompt, system=system_prompt)
        raw = response.text().strip()
        if raw.startswith("```"):
            lines = raw.split("\n")
            lines = [l for l in lines if not l.strip().startswith("```")]
            raw = "\n".join(lines)
        data = json.loads(raw)
        return {
            "reading": str(data.get("reading", "")),
            "meanings_en": str(data.get("meanings_en", "")),
            "pos_tags": str(data.get("pos_tags", "")),
            "jlpt_level": str(data.get("jlpt_level", "")),
        }
    except Exception as e:
        print(f"⚠️ Word annotation failed for {card.lemma}: {e}")
        return {"reading": "", "meanings_en": "", "pos_tags": "", "jlpt_level": ""}


def generate_word_quiz(card: Any, distractors: List[Dict[str, Any]], model: Any) -> List[Dict[str, Any]]:
    """Generate a 3-question quiz for a Top Word card.

    Question types:
      1. meaning_match — identify the English meaning of the word
      2. reading_match — identify the hiragana reading (for kanji words)
         OR sentence_translation (for kana-only words where reading is obvious)
      3. sentence_translation — translate a Japanese sentence containing this word
    """
    if model is None:
        raise ValueError("AI model required")

    system_prompt = """You are a Japanese vocabulary quiz generator. Generate exactly 3 distinct multiple-choice questions
for the provided word. Each question must have exactly 4 answer choices (A, B, C, D).

Return ONLY valid JSON in this format:
{
  "questions": [
    {
      "question_type": "meaning_match",
      "question": "...",
      "choices": [{"key": "A", "text": "..."}, {"key": "B", "text": "..."}, {"key": "C", "text": "..."}, {"key": "D", "text": "..."}],
      "correct_key": "B"
    },
    ... (2 more questions, 3 total)
  ]
}

RULES:
- Randomize the position of the correct answer for each question.
- Make distractors plausible but clearly wrong.
- All questions must be comprehensible by an English-speaking Japanese learner.
- Sentences must be natural Japanese that a native speaker would actually say."""

    distractor_info = "\n".join(
        f"- {d.get('lemma','?')}: reading={d.get('reading','?')}, meanings={d.get('meanings_en','?')}"
        for d in distractors
    )

    # Determine if word contains kanji (reading question is useful) or is kana-only
    has_kanji = any('\u4e00' <= ch <= '\u9fff' for ch in (card.lemma or ''))

    if has_kanji:
        q2_instruction = f"""2. reading_match: Ask "What is the reading (hiragana) of 「{card.lemma}」?"
   Choices are hiragana readings. Include the correct reading ({card.reading}) and 3 plausible wrong readings from distractors."""
    else:
        q2_instruction = f"""2. sentence_translation: Create a SHORT, natural Japanese sentence using 「{card.lemma}」.
   Ask "What does this sentence mean?" Choices are English translations.
   Only one translation should be correct. Make this a DIFFERENT sentence from question 3."""

    prompt = f"""Target word: {card.lemma}
Reading: {card.reading}
Meanings: {card.meanings_en}
POS: {card.pos_tags}
Contains kanji: {"yes" if has_kanji else "no (kana-only word)"}

Distractors:
{distractor_info}

Generate exactly 3 questions, one for each type:

1. meaning_match: Ask "What does 「{card.lemma}」 mean?"
   Choices are English meanings. Use the correct meaning and 3 distractor meanings.

{q2_instruction}

3. sentence_translation: Create a SHORT, natural Japanese sentence (8-15 words) using 「{card.lemma}」.
   Ask "Choose the correct English translation of this sentence:"
   Show the Japanese sentence in the question. Choices are 4 English translations — only one correct.
   The sentence should be natural and help the learner understand how the word is used in context.

Return ONLY valid JSON with a "questions" array containing exactly 3 objects."""

    try:
        response = model.prompt(prompt, system=system_prompt)
        raw = response.text().strip()
        if raw.startswith("```"):
            lines = raw.split("\n")
            lines = [l for l in lines if not l.strip().startswith("```")]
            raw = "\n".join(lines)
        result = json.loads(raw)
        questions = result.get("questions", [])
        if len(questions) < 3:
            print(f"⚠️ Expected 3 word questions, got {len(questions)}. Padding.")
            while len(questions) < 3:
                questions.append(_word_fallback_question(card, distractors))
        return questions[:3]
    except Exception as e:
        print(f"❌ Word quiz generation failed: {e}")
        fallback = _word_fallback_question(card, distractors)
        return [fallback] * 3


def generate_word_mc_exercise(card: Any, distractors: List[Dict[str, Any]], model: Any) -> Dict[str, Any]:
    """Legacy wrapper for single question generation."""
    quiz = generate_word_quiz(card, distractors, model)
    return quiz[0] if quiz else _word_fallback_question(card, distractors)


def _word_fallback_question(card: Any, distractors: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Deterministic fallback for word MC question."""
    choices = [{"key": "A", "text": card.meanings_en or "unknown"}]
    for i, d in enumerate(distractors[:3]):
        choices.append({"key": chr(ord("B") + i), "text": d.get("meanings_en", f"meaning {i+1}")})
    correct_text = choices[0]["text"]
    random.shuffle(choices)
    correct_key = "A"
    for i, c in enumerate(choices):
        c["key"] = chr(ord("A") + i)
        if c["text"] == correct_text:
            correct_key = c["key"]
    return {"question": f"What does 「{card.lemma}」 mean?", "question_type": "meaning_match",
            "choices": choices, "correct_key": correct_key}


def generate_word_lesson(card: Any, model: Any) -> Dict[str, Any]:
    """Generate an AI lesson for a Japanese word."""
    if model is None:
        raise ValueError("AI model required")
    system_prompt = "You are a Japanese vocabulary teacher. Return ONLY valid JSON."
    prompt = f"""Generate a lesson for the word 「{card.lemma}」 for an English native language speaker who is learning Japanese.
Reading: {card.reading}
Meanings: {card.meanings_en}
POS: {card.pos_tags}

Return JSON:
{{
  "word": "{card.lemma}",
  "reading": "{card.reading}",
  "meanings": "{card.meanings_en}",
  "explanation": "Clear explanation of this word and when to use it (2-3 sentences)",
  "formation": "If it has kanji, explain the kanji composition",
  "examples": [{{"japanese": "example sentence", "english": "translation"}}],
  "tips": "Tips for remembering or using this word"
}}"""
    try:
        response = model.prompt(prompt, system=system_prompt)
        raw = response.text().strip()
        if raw.startswith("```"):
            lines = raw.split("\n")
            lines = [l for l in lines if not l.strip().startswith("```")]
            raw = "\n".join(lines)
        lesson = json.loads(raw)
        lesson.setdefault("word", card.lemma)
        lesson.setdefault("reading", card.reading)
        lesson.setdefault("meanings", card.meanings_en)
        return lesson
    except Exception as e:
        print(f"❌ Word lesson failed: {e}")
        return {"word": card.lemma, "reading": card.reading, "meanings": card.meanings_en,
                "explanation": f"{card.lemma} means '{card.meanings_en}'.",
                "formation": "", "examples": [], "tips": f"Ranked #{card.rank} by frequency."}


def generate_word_feedback(card: Any, is_correct: bool, selected_text: str, correct_text: str, model: Any) -> str:
    """Brief AI feedback after a word MC answer."""
    if model is None:
        return f"{'Correct!' if is_correct else 'Incorrect.'} 「{card.lemma}」 means '{card.meanings_en}'."
    try:
        prompt = f"The student {'correctly' if is_correct else 'incorrectly'} answered about 「{card.lemma}」 ({card.meanings_en}). " + \
                 (f"They selected '{selected_text}' but correct was '{correct_text}'. " if not is_correct else "") + \
                 "Give 1-2 sentence feedback **predominantly in English**."
        response = model.prompt(prompt, system="You are a Japanese vocabulary tutor. Be encouraging and concise.")
        return response.text().strip()
    except Exception:
        return f"{'Correct!' if is_correct else 'Incorrect.'} 「{card.lemma}」 means '{card.meanings_en}'."