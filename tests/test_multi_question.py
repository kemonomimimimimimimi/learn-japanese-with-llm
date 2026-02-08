import pytest
from unittest.mock import MagicMock, patch
from llm_learn_japanese import exercises, db

class MockResponse:
    def __init__(self, text_content):
        self.text_content = text_content
    def text(self):
        return self.text_content

@pytest.fixture
def mock_model():
    model = MagicMock()
    # Mock response for Bunpro quiz (3 questions)
    bunpro_json = """
    {
        "questions": [
            {"question_type": "meaning_match", "question": "What does へ mean?", "choices": [{"key":"A","text":"to, toward"},{"key":"B","text":"from"},{"key":"C","text":"at"},{"key":"D","text":"in"}], "correct_key": "A"},
            {"question_type": "sentence_translation", "question": "Translate: 学校へ行きます。", "choices": [{"key":"A","text":"I go to school."},{"key":"B","text":"I come from school."},{"key":"C","text":"I am at school."},{"key":"D","text":"I study at school."}], "correct_key": "A"},
            {"question_type": "correct_usage", "question": "Which sentence correctly uses へ?", "choices": [{"key":"A","text":"日本へ行きたい (I want to go to Japan)"},{"key":"B","text":"水へ飲む (I drink water)"},{"key":"C","text":"本へ読む (I read a book)"},{"key":"D","text":"友達へなる (I become a friend)"}], "correct_key": "A"}
        ]
    }
    """
    model.prompt.return_value = MockResponse(bunpro_json)
    return model

def test_generate_bunpro_quiz(mock_model):
    card = MagicMock()
    card.grammar = "test_grammar"
    card.meaning = "test_meaning"
    card.jlpt = "N5"
    card.register = "casual"
    
    distractors = [{"grammar": "d1", "meaning": "m1"}, {"grammar": "d2", "meaning": "m2"}]
    
    quiz = exercises.generate_bunpro_quiz(card, distractors, mock_model)
    
    assert len(quiz) == 3
    assert quiz[0]["question_type"] == "meaning_match"
    assert quiz[1]["question_type"] == "sentence_translation"
    assert quiz[2]["question_type"] == "correct_usage"

def test_generate_kanji_quiz(mock_model):
    # Mock response for Kanji quiz (3 questions)
    kanji_json = """
    {
        "questions": [
            {"question_type": "meaning_match", "question": "What does 日 mean?", "choices": [{"key":"A","text":"day"},{"key":"B","text":"moon"},{"key":"C","text":"fire"},{"key":"D","text":"water"}], "correct_key": "A"},
            {"question_type": "reading_match", "question": "What is a reading of 日?", "choices": [{"key":"A","text":"げつ"},{"key":"B","text":"にち"},{"key":"C","text":"か"},{"key":"D","text":"すい"}], "correct_key": "B"},
            {"question_type": "compound_word", "question": "What does 日本 mean?", "choices": [{"key":"A","text":"sunrise"},{"key":"B","text":"Sunday"},{"key":"C","text":"Japan"},{"key":"D","text":"diary"}], "correct_key": "C"}
        ]
    }
    """
    mock_model.prompt.return_value = MockResponse(kanji_json)
    
    card = MagicMock()
    card.kanji = "日"
    card.meanings_en = "day, sun"
    card.on_readings = "ニチ, ジツ"
    card.kun_readings = "ひ, か"
    
    distractors = []
    
    quiz = exercises.generate_kanji_quiz(card, distractors, mock_model)
    
    assert len(quiz) == 3
    assert quiz[0]["question_type"] == "meaning_match"
    assert quiz[1]["question_type"] == "reading_match"
    assert quiz[2]["question_type"] == "compound_word"

def test_generate_word_quiz(mock_model):
    # Mock response for Word quiz (3 questions)
    word_json = """
    {
        "questions": [
            {"question_type": "meaning_match", "question": "What does 食べる mean?", "choices": [{"key":"A","text":"to eat"},{"key":"B","text":"to drink"},{"key":"C","text":"to run"},{"key":"D","text":"to sleep"}], "correct_key": "A"},
            {"question_type": "reading_match", "question": "What is the reading of 食べる?", "choices": [{"key":"A","text":"のべる"},{"key":"B","text":"たべる"},{"key":"C","text":"くべる"},{"key":"D","text":"しべる"}], "correct_key": "B"},
            {"question_type": "sentence_translation", "question": "Choose the correct English translation: 昨日レストランで食べました。", "choices": [{"key":"A","text":"I went to a restaurant yesterday."},{"key":"B","text":"I ate at a restaurant yesterday."},{"key":"C","text":"I cooked at a restaurant yesterday."},{"key":"D","text":"I worked at a restaurant yesterday."}], "correct_key": "B"}
        ]
    }
    """
    mock_model.prompt.return_value = MockResponse(word_json)
    
    card = MagicMock()
    card.lemma = "食べる"
    card.reading = "たべる"
    card.meanings_en = "to eat"
    card.pos_tags = "verb"
    
    distractors = []
    
    quiz = exercises.generate_word_quiz(card, distractors, mock_model)
    
    assert len(quiz) == 3
    assert quiz[0]["question_type"] == "meaning_match"
    assert quiz[1]["question_type"] == "reading_match"
    assert quiz[2]["question_type"] == "sentence_translation"

def test_generate_word_quiz_kana_only(mock_model):
    """For kana-only words, question 2 should be sentence_translation instead of reading_match."""
    word_json = """
    {
        "questions": [
            {"question_type": "meaning_match", "question": "What does けど mean?", "choices": [{"key":"A","text":"but"},{"key":"B","text":"and"},{"key":"C","text":"or"},{"key":"D","text":"so"}], "correct_key": "A"},
            {"question_type": "sentence_translation", "question": "Translate: 行きたいけど時間がない。", "choices": [{"key":"A","text":"I want to go and I have time."},{"key":"B","text":"I want to go but I don't have time."},{"key":"C","text":"I don't want to go because I have time."},{"key":"D","text":"I will go when I have time."}], "correct_key": "B"},
            {"question_type": "sentence_translation", "question": "Translate: 高いけど買った。", "choices": [{"key":"A","text":"It was cheap so I bought it."},{"key":"B","text":"It was expensive but I bought it."},{"key":"C","text":"It was expensive so I didn't buy it."},{"key":"D","text":"It was cheap but I didn't buy it."}], "correct_key": "B"}
        ]
    }
    """
    mock_model.prompt.return_value = MockResponse(word_json)

    card = MagicMock()
    card.lemma = "けど"
    card.reading = "けど"
    card.meanings_en = "but, though, however"
    card.pos_tags = "conjunction"

    distractors = []

    quiz = exercises.generate_word_quiz(card, distractors, mock_model)

    assert len(quiz) == 3
    assert quiz[0]["question_type"] == "meaning_match"
    # For kana-only words, no reading_match
    question_types = [q["question_type"] for q in quiz]
    assert "reading_match" not in question_types
