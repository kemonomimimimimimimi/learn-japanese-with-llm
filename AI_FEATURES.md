# AI-Powered Features Documentation

## Overview

This Japanese Learning Application now includes comprehensive AI integration for dynamic exercise generation and intelligent answer evaluation. The system uses language models to create personalized, context-aware exercises and provides semantic evaluation of student responses.

## Key Components

### 1. Dynamic Exercise Generation (`llm_learn_japanese/exercises.py`)

#### `generate_exercise(aspect, model=None)`

The core exercise generation function that creates dynamic, AI-powered exercises based on the learning aspect and user's progress.

**Features:**
- **Adaptive Difficulty**: Adjusts difficulty based on the student's success count
  - Beginner (< 3 successful reviews)
  - Intermediate (3-10 successful reviews)
  - Advanced (> 10 successful reviews)
- **Varied Exercise Types**: Generates different types of exercises for each aspect
- **Context-Aware**: Uses full content context to create meaningful exercises
- **Fallback Support**: Returns template-based exercises when AI is unavailable

**Exercise Types by Content:**

#### Vocabulary
- **Meaning Aspect**: Direct translation, multiple choice, fill-in-blank, context sentences
- **Reading Aspect**: Write reading, choose correct reading, reading from context
- **Usage Aspect**: Create sentences, complete dialogues, appropriate context

#### Kanji
- **Meaning Aspect**: Kanji to meaning, meaning to kanji, compound meanings
- **Onyomi Aspect**: Write onyomi, identify onyomi words, onyomi compounds
- **Kunyomi Aspect**: Write kunyomi, identify kunyomi words, kunyomi usage
- **Usage Aspect**: Kanji in sentences, compound creation, reading choice

#### Grammar
- **Explanation Aspect**: Explain usage, identify patterns, error correction
- **Usage Aspect**: Create examples, transform sentences, complete with grammar

#### Phrases
- **Meaning Aspect**: Translate phrase, situational usage, cultural context
- **Usage Aspect**: Dialogue completion, appropriate response, role-play scenarios

#### Idioms
- **Meaning Aspect**: Literal vs figurative, explain meaning, match situations
- **Usage Aspect**: Use in context, identify appropriate situations, create dialogues

### 2. AI-Powered Answer Evaluation (`llm_learn_japanese/db.py`)

#### `evaluate_answer_with_ai(aspect, user_answer, model=None)`

Intelligent evaluation system that understands semantic meaning and provides constructive feedback.

**Features:**
- **Semantic Understanding**: Evaluates based on meaning, not just exact matches
- **Partial Credit**: Awards appropriate scores for partially correct answers
- **Synonym Recognition**: Accepts alternative correct forms and synonyms
- **Contextual Evaluation**: Considers the type of question when evaluating
- **Detailed Feedback**: Provides constructive, encouraging feedback

**Scoring System:**
- **0**: No answer or completely wrong
- **1**: Very poor understanding
- **2**: Poor understanding with major errors
- **3**: Acceptable with some errors
- **4**: Good understanding with minor issues
- **5**: Perfect or excellent answer

### 3. Context Enrichment Functions

#### Helper Functions

**`_get_aspect_context(aspect)`**: Retrieves full context for an aspect including parent content details
**`_calculate_difficulty_level(aspect)`**: Determines appropriate difficulty based on performance history
**`_get_exercise_types_for_aspect(aspect_type, parent_type)`**: Returns appropriate exercise types for the given aspect
**`_build_generation_prompt(context, exercise_type, difficulty)`**: Creates detailed prompts for AI exercise generation

## Usage Examples

### Using AI-Powered Exercise Generation

```python
import llm
from llm_learn_japanese import exercises, db

# Get an LLM model
model = llm.get_model("gpt-5-2025-08-07")

# Get next card aspect
aspect = db.get_next_card()

# Generate AI-powered exercise
exercise = exercises.generate_exercise(aspect, model=model)
print(exercise)
# Output: "Create a sentence using the word 勉強 that shows you understand its meaning."
```

### Using AI-Powered Answer Evaluation

```python
# Evaluate a user's answer
user_answer = "I need to 勉強 for my test tomorrow"
quality_score, feedback = db.evaluate_answer_with_ai(
    aspect,
    user_answer,
    model=model
)

print(f"Score: {quality_score}/5")
print(f"Feedback: {feedback}")
# Output:
# Score: 4/5
# Feedback: Good job! You correctly used 勉強 in a natural context. Consider adding する to make it grammatically complete (勉強する).
```

### Fallback Behavior

When no AI model is available, the system gracefully falls back to template-based generation and rule-based evaluation:

```python
# Without model - uses templates
exercise = exercises.generate_exercise(aspect, model=None)
# Returns: "What is the English meaning of: 勉強?"

# Evaluation without model - uses simple matching
score, feedback = db.evaluate_answer_with_ai(aspect, "study", model=None)
# Returns moderate score with manual review suggestion
```

## CLI Integration

The AI features are fully integrated with the command-line interface:

```bash
# Study with AI-powered exercises and evaluation
llm jp-next-card --model gpt-5-2025-08-07

# Manual mode (just show exercise, no auto-evaluation)
llm jp-next-card --model gpt-5-2025-08-07 --manual

# Add content from images using vision models
llm jp-add-image manga_page.jpg --model gpt-5-2025-08-07-vision

# Process Discord chat logs
llm jp-add-discord chat_log.txt --model gpt-5-2025-08-07
```

## Benefits of AI Integration

### 1. **Dynamic Content Generation**
- Never see the exact same question twice
- Exercises adapt to your learning level
- Creative and engaging question formats

### 2. **Intelligent Evaluation**
- Understands synonyms and alternative answers
- Provides partial credit for partially correct responses
- Gives constructive, personalized feedback

### 3. **Personalized Learning**
- Difficulty adjusts based on performance
- Exercise types vary to maintain engagement
- Context-aware questions that build understanding

### 4. **Comprehensive Coverage**
- Multiple exercise types per aspect
- Different angles of testing the same concept
- Progressive difficulty scaling

### 5. **Graceful Degradation**
- Works without AI using templates
- Maintains core functionality even offline
- Smooth transition between AI and non-AI modes

## Technical Architecture

### Exercise Generation Flow
1. Retrieve aspect and full context from database
2. Calculate appropriate difficulty level
3. Select random exercise type for variety
4. Build detailed AI prompt with context
5. Generate exercise using language model
6. Fallback to template if generation fails

### Answer Evaluation Flow
1. Retrieve full context including expected answers
2. Build evaluation prompt with scoring guidelines
3. Submit to AI for semantic evaluation
4. Parse JSON response for score and feedback
5. Fallback to rule-based evaluation if needed

### Database Integration
- Aspect success counts track performance
- Context retrieval supports all content types
- Scheduling algorithm (FSRS) remains independent
- Progress tracking updated based on AI scores

## Configuration and Requirements

### Required Dependencies
- `llm` library for model access
- SQLAlchemy for database operations
- Language model API access (OpenAI, Anthropic, etc.)

### Recommended Models
- **Exercise Generation**: GPT-5, Claude, or similar
- **Answer Evaluation**: Any model with JSON output support
- **Vision Tasks**: GPT-5, Claude-Vision
- **Text Processing**: Any capable language model

## Future Enhancements

### Planned Features
1. **Adaptive Learning Paths**: AI suggests what to study next
2. **Weakness Detection**: Identify and focus on problem areas
3. **Custom Exercise Requests**: Generate specific exercise types on demand
4. **Multi-modal Exercises**: Include images and audio in exercises
5. **Conversation Practice**: AI-powered dialogue simulations

### Optimization Opportunities
1. **Response Caching**: Cache similar exercises for efficiency
2. **Batch Generation**: Pre-generate exercises during idle time
3. **Model Fine-tuning**: Train specialized models for Japanese learning
4. **Local Models**: Support for offline AI using local models

## Troubleshooting

### Common Issues

**Issue**: AI generates inappropriate difficulty
**Solution**: Check success_count in database, reset if needed

**Issue**: Evaluation seems too harsh/lenient
**Solution**: Adjust scoring guidelines in system prompt

**Issue**: Exercises are repetitive
**Solution**: Ensure variety in exercise type selection

**Issue**: AI responses are slow
**Solution**: Consider using faster models or implementing caching

## Best Practices

1. **Use Appropriate Models**: Choose models based on task complexity
2. **Monitor Costs**: AI calls can add up, use --manual mode when appropriate
3. **Review AI Feedback**: Occasionally verify AI evaluations manually
4. **Combine Methods**: Use both AI and template exercises for variety
5. **Regular Updates**: Keep prompts and exercise types current

## Conclusion

The AI integration transforms this Japanese learning application from a simple flashcard system into an intelligent, adaptive learning platform. The combination of dynamic exercise generation and semantic answer evaluation provides a more engaging and effective learning experience while maintaining reliability through graceful fallbacks.