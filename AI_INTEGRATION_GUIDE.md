# AI Integration Guide for Japanese Learning App

## Current Architecture

The Japanese Learning App currently operates as a **curses-based terminal application** without active AI integration. However, the codebase has been fully engineered to support AI-powered features when language models become available.

## Implemented AI-Ready Features

### 1. Exercise Generation System (`llm_learn_japanese/exercises.py`)

The `generate_exercise()` function is designed to work in two modes:

**Without AI (Current Mode):**
```python
question = exercises.generate_exercise(aspect, model=None)
# Returns template-based questions from the database
```

**With AI (Future Mode):**
```python
question = exercises.generate_exercise(aspect, model=ai_model)
# Generates dynamic, context-aware exercises based on:
# - User's performance history (success_count)
# - Content type and aspect
# - Difficulty level (beginner/intermediate/advanced)
# - Varied exercise types
```

### 2. Answer Evaluation System (`llm_learn_japanese/db.py`)

The `evaluate_answer_with_ai()` function provides intelligent scoring:

**Without AI (Current Mode):**
```python
score, feedback = db.evaluate_answer_with_ai(aspect, user_answer, model=None)
# Uses simple string matching and basic rules
```

**With AI (Future Mode):**
```python
score, feedback = db.evaluate_answer_with_ai(aspect, user_answer, model=ai_model)
# Provides semantic evaluation with:
# - Understanding of synonyms
# - Partial credit for partially correct answers
# - Constructive, personalized feedback
# - Context-aware scoring
```

## How to Add AI Integration

To enable AI features in the curses interface, you would need to:

### Step 1: Add AI Model Configuration

Create a configuration system in `main.py`:

```python
class JapaneseApp:
    def __init__(self, stdscr: Any, ai_model=None) -> None:
        self.stdscr = stdscr
        self.ai_model = ai_model  # Store AI model if available
        # ... rest of initialization
```

### Step 2: Update Study Session

Modify the study session to use AI when available:

```python
def show_study_session(self) -> None:
    # ... existing code ...

    # Generate exercise with AI if available
    try:
        question = exercises.generate_exercise(aspect, model=self.ai_model)
    except Exception as e:
        question = f"What is the {aspect.aspect_type} for this item?"

    # ... later in the code ...

    # Evaluate with AI if available
    score, feedback = db.evaluate_answer_with_ai(
        aspect, user_answer, model=self.ai_model
    )
```

### Step 3: Add Model Selection Menu

Add an option to configure AI model in the interface:

```python
def configure_ai_model(self) -> None:
    """Configure AI model for enhanced features."""
    model_choice = self.get_user_input("Enter model name (e.g., gpt-5-2025-08-07, claude):")
    if model_choice:
        try:
            # Initialize your AI model here
            # self.ai_model = initialize_model(model_choice)
            self.show_message(f"✅ AI model configured: {model_choice}", 3)
        except Exception as e:
            self.show_message(f"❌ Failed to configure model: {e}", 4)
```

## Benefits of the Current Architecture

### 1. **Graceful Degradation**
- The app works perfectly without AI
- Template-based questions provide consistent learning
- No dependency on external AI services for basic functionality

### 2. **Future-Ready**
- All AI integration points are already in place
- Exercise generation supports 40+ different exercise types
- Evaluation system ready for semantic understanding

### 3. **Flexible Design**
- Easy to switch between AI and non-AI modes
- Can enable AI for specific features while keeping others template-based
- Supports multiple AI models through abstraction

## Exercise Types Ready for AI

The system includes comprehensive exercise type mappings for:

### Vocabulary (per aspect)
- **Meaning**: direct translation, multiple choice, fill-in-blank, context sentences
- **Reading**: write reading, choose correct reading, reading from context
- **Usage**: create sentences, complete dialogues, appropriate context

### Kanji (per aspect)
- **Meaning**: kanji to meaning, meaning to kanji, compound meanings
- **Onyomi**: write onyomi, identify onyomi words, onyomi compounds
- **Kunyomi**: write kunyomi, identify kunyomi words, kunyomi usage
- **Usage**: kanji in sentences, compound creation, reading choice

### Grammar (per aspect)
- **Explanation**: explain usage, identify patterns, error correction
- **Usage**: create examples, transform sentences, complete with grammar

### Phrases & Idioms
- Various contextual and cultural exercises
- Dialogue completion and role-play scenarios
- Situational usage and appropriateness checks

## Testing the AI Features

The test suite (`tests/test_app.py`) includes tests for both modes:

```python
# Test without AI (current behavior)
exercise = exercises.generate_exercise(aspect, model=None)
assert exercise == aspect.prompt_template  # Uses template

# Test with mock AI (future behavior)
mock_model = MockModel()
exercise = exercises.generate_exercise(aspect, model=mock_model)
# Returns AI-generated content
```

## Performance Considerations

When AI is eventually integrated:

1. **Caching**: Consider caching generated exercises for similar contexts
2. **Batch Generation**: Pre-generate exercises during idle time
3. **Fallback Strategy**: Always fall back to templates if AI fails
4. **Response Time**: Add loading indicators for AI operations
5. **Cost Management**: Track AI API usage and provide options to limit

## Summary

The Japanese Learning App is fully prepared for AI integration while currently operating successfully without it. The architecture allows for:

- **Immediate use** as a template-based learning system
- **Future enhancement** with AI-powered dynamic content
- **Gradual adoption** of AI features as they become available
- **Consistent experience** whether using AI or templates

The codebase demonstrates best practices in:
- Separation of concerns
- Graceful degradation
- Future-proofing
- Testable architecture
- Clear abstraction layers

This design ensures the app remains functional and valuable today while being ready for advanced AI features tomorrow.