# Multi-Question Review Plan

## Objective
Enhance the review process for Bunpro Grammar, Top Kanji, and Top Words by asking 4 distinct questions per item during a single review session. Optimize the generation process using parallel API calls.

## Question Types

### Bunpro Grammar
1.  **Meaning Match:** Identify the correct English meaning of the grammar point.
2.  **Fill-in-the-blank:** Select the correct grammar form to complete a Japanese sentence.
3.  **Translation (J->E):** Choose the correct English translation for a Japanese sentence using the grammar point.
4.  **Usage/Context:** Identify the correct usage or nuance (e.g., choose the correct sentence among 4 options, or identify the situation where the grammar is appropriate).

### Top Kanji
1.  **Meaning Match:** Identify the English meaning of the Kanji.
2.  **Reading Match:** Identify the correct reading (Onyomi or Kunyomi) of the Kanji.
3.  **Word Completion:** Select the correct Kanji to complete a compound word.
4.  **Context Sentence:** Choose the correct sentence that uses a word containing this Kanji (or identify the correct meaning of the Kanji in a specific sentence).

### Top Words
1.  **Meaning Match:** Identify the English meaning of the word.
2.  **Reading Match:** Identify the correct reading (Hiragana) of the word.
3.  **Fill-in-the-blank:** Select the correct word to complete a Japanese sentence.
4.  **Usage/Context:** Choose the correct English translation of a sentence using the word, or identify the correct usage context.

## Data Structure (JSON Response from LLM)

The LLM will return a JSON object containing a list of 4 questions.

```json
{
  "questions": [
    {
      "question_type": "meaning_match",
      "question": "What does 「...」 mean?",
      "choices": [
        {"key": "A", "text": "..."},
        {"key": "B", "text": "..."},
        {"key": "C", "text": "..."},
        {"key": "D", "text": "..."}
      ],
      "correct_key": "B"
    },
    ...
  ]
}
```

## Implementation Plan

### 1. Backend: `llm_learn_japanese/exercises.py`
-   **Status:** `generate_bunpro_quiz`, `generate_kanji_quiz`, and `generate_word_quiz` appear to be implemented to return 4 questions.
-   **Action:** Verify these functions are robust and correctly integrated.

### 2. Backend: `app.py` - Parallel Generation
-   **Current State:** `api_bunpro_batch` loops sequentially, making 10 separate blocking calls to the LLM. This is slow.
-   **Optimization:** Use `concurrent.futures.ThreadPoolExecutor` to run the quiz generation for the 10 cards in parallel.
-   **Why Parallel vs Single Batch Prompt:**
    -   **Reliability:** Generating 40 questions in one JSON response is prone to syntax errors and hallucination.
    -   **Speed:** Parallel calls are nearly as fast as a single call (limited by the slowest response).
    -   **Isolation:** If one card fails, the others still succeed.

### 3. Frontend: Templates (`bunpro.html`, `mc_study.html`)
-   **Action:** Ensure the JavaScript correctly handles the `questions` array in the card object.
-   **UI Flow:**
    -   Display Question 1/4.
    -   User answers -> Immediate feedback.
    -   "Next" button loads Question 2/4 (without reloading page/card).
    -   After Question 4/4, calculate aggregate score (0-5).
    -   Submit aggregate score to backend.

## Files to Modify
-   `app.py`: Implement `ThreadPoolExecutor` in `api_*_batch` endpoints.
-   `templates/mc_study.html`: Update JS to cycle through the list of questions for a single card.
-   `templates/bunpro.html`: Update JS similarly.

## Next Steps
1.  Approve this plan.
2.  Modify `app.py` for parallel execution.
3.  Update frontend templates to support multi-question flow.
