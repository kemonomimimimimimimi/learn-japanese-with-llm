from . import db
from typing import Any

try:
    import llm  # type: ignore
    hookimpl = llm.hookimpl  # type: ignore
except ImportError:
    import pluggy
    hookimpl = pluggy.HookimplMarker("llm")

@hookimpl  # type: ignore[misc]
def register_commands(cli: Any) -> None:
    import click

    @cli.command("jp-init-db")  # type: ignore[misc]
    def init_db() -> None:
        """Initialize the Japanese learning database."""
        db.init_db()
        click.echo("Database initialized.")

    from typing import Any as _Any
    @cli.command("jp-save-message")  # type: ignore[misc]
    @click.argument("message")
    def save_message(message: str) -> None:
        """Save a message into the database."""
        db.save_message(message)
        click.echo("Message saved.")

    @cli.command("jp-add-card")  # type: ignore[misc]
    @click.argument("content")
    def add_card(content: str) -> None:
        """Add vocabulary with atomic aspects for comprehensive learning."""
        is_new = db.add_card(content)
        if is_new:
            click.echo(f"Vocabulary card '{content}' added with atomic aspects (meaning, reading, usage).")
        else:
            click.echo(f"Vocabulary card '{content}' already exists (skipped).")

    @cli.command("jp-add-kanji")  # type: ignore[misc]
    @click.argument("character")
    @click.option("--onyomi", default="", help="Chinese reading")
    @click.option("--kunyomi", default="", help="Japanese reading")
    @click.option("--meanings", default="", help="Meanings")
    def add_kanji(character: str, onyomi: str, kunyomi: str, meanings: str) -> None:
        """Add kanji with atomic aspects (meaning, onyomi, kunyomi, usage)."""
        is_new = db.add_kanji(character, onyomi, kunyomi, meanings)
        if is_new:
            click.echo(f"Kanji '{character}' added with atomic aspects.")
        else:
            click.echo(f"Kanji '{character}' already exists (skipped).")

    @cli.command("jp-add-grammar")  # type: ignore[misc]
    @click.argument("point")
    @click.option("--explanation", default="", help="Grammar explanation")
    @click.option("--example", default="", help="Example sentence")
    def add_grammar(point: str, explanation: str, example: str) -> None:
        """Add grammar point with atomic aspects (explanation, usage)."""
        is_new = db.add_grammar(point, explanation, example)
        if is_new:
            click.echo(f"Grammar point '{point}' added with atomic aspects.")
        else:
            click.echo(f"Grammar point '{point}' already exists (skipped).")

    @cli.command("jp-add-phrase")  # type: ignore[misc]
    @click.argument("phrase")
    @click.option("--meaning", default="", help="English meaning")
    def add_phrase(phrase: str, meaning: str) -> None:
        """Add phrase with atomic aspects (meaning, usage)."""
        is_new = db.add_phrase(phrase, meaning)
        if is_new:
            click.echo(f"Phrase '{phrase}' added with atomic aspects.")
        else:
            click.echo(f"Phrase '{phrase}' already exists (skipped).")

    @cli.command("jp-add-idiom")  # type: ignore[misc]
    @click.argument("idiom")
    @click.option("--meaning", default="", help="English meaning")
    @click.option("--example", default="", help="Example usage")
    def add_idiom(idiom: str, meaning: str, example: str) -> None:
        """Add idiom with atomic aspects (meaning, usage)."""
        is_new = db.add_idiom(idiom, meaning, example)
        if is_new:
            click.echo(f"Idiom '{idiom}' added with atomic aspects.")
        else:
            click.echo(f"Idiom '{idiom}' already exists (skipped).")

    @cli.command("jp-next-card")  # type: ignore[misc]
    @click.option("--model", default="gpt-5-2025-08-07", help="LLM model name to use for generating exercises and evaluation")
    @click.option("--manual", is_flag=True, help="Manual mode - don't prompt for answer, just show question")
    def next_card(model: str, manual: bool) -> None:
        """Fetch the next aspect using FSRS algorithm, generate an exercise, and evaluate your answer."""
        from . import exercises
        import llm  # type: ignore
        aspect = db.get_next_card()
        if aspect:
            # Load model if available
            llm_model = None
            try:
                if model and model != "default":
                    llm_model = llm.get_model(model)
            except Exception:
                llm_model = None

            question = exercises.generate_exercise(aspect, model=llm_model)
            click.echo(f"Exercise (Aspect: {aspect.aspect_type}):")
            click.echo(f"{question}")

            if manual:
                # Manual mode - just show the question
                click.echo(f"\nAspect ID: {aspect.id}")
                click.echo("Use 'llm jp-review-card <aspect_id> <quality>' to record your performance (quality: 0-5)")
            else:
                # Interactive mode - prompt for answer and auto-evaluate
                click.echo("")
                user_answer = click.prompt("Your answer", type=str)

                try:
                    # Use LLM to evaluate the answer
                    quality_score, feedback = db.evaluate_answer_with_ai(aspect, user_answer, model=llm_model)

                    # Display results
                    quality_labels = {
                        0: "❌ Incorrect",
                        1: "💔 Very Poor",
                        2: "😟 Poor",
                        3: "😐 Acceptable",
                        4: "😊 Good",
                        5: "🎉 Perfect"
                    }

                    click.echo(f"\n{quality_labels.get(quality_score, '❓ Unknown')} (Quality: {quality_score}/5)")
                    click.echo(f"Feedback: {feedback}")

                    # Automatically update the card with the quality score
                    db.review_card(aspect.id, quality_score)

                    if quality_score >= 3:
                        click.echo("📅 Card rescheduled for later review.")
                    else:
                        click.echo("🔄 Card will be reviewed again soon.")

                except Exception as e:
                    click.echo(f"\n⚠️  Could not auto-evaluate answer: {e}")
                    click.echo(f"Your answer was: '{user_answer}'")
                    click.echo(f"Please use 'llm jp-review-card {aspect.id} <quality>' to record manually (quality: 0-5)")
        else:
            click.echo("🎉 No aspects are due for review! All caught up!")

    @cli.command("jp-review-card")  # type: ignore[misc]
    @click.argument("aspect_id", type=int)
    @click.argument("quality", type=int)
    def review_card(aspect_id: int, quality: int) -> None:
        """Review a card aspect and update FSRS scheduling."""
        if quality < 0 or quality > 5:
            click.echo("Quality must be between 0-5 (0=forgot, 3=remembered with effort, 5=easy)")
            return

        db.review_card(aspect_id, quality)
        if quality >= 3:
            click.echo(f"Good! Aspect {aspect_id} scheduled for later review.")
        else:
            click.echo(f"That's okay! Aspect {aspect_id} will be reviewed again soon.")

    @cli.command("jp-progress")  # type: ignore[misc]
    @click.argument("user")
    def show_progress(user: str) -> None:
        """Show learning progress for a user."""
        progress = db.get_progress(user)
        if progress:
            accuracy = (progress["correct_answers"] / progress["total_reviews"] * 100) if progress["total_reviews"] > 0 else 0
            click.echo(f"Progress for {user}:")
            click.echo(f"  Total reviews: {progress['total_reviews']}")
            click.echo(f"  Correct answers: {progress['correct_answers']}")
            click.echo(f"  Accuracy: {accuracy:.1f}%")
            click.echo(f"  Last updated: {progress['last_updated']}")
        else:
            click.echo(f"No progress found for user '{user}'")

    @cli.command("jp-add-image")  # type: ignore[misc]
    @click.argument("image_path")
    @click.option("--model", default="gpt-5-2025-08-07", help="LLM vision model to use for image parsing")
    def add_image(image_path: str, model: str) -> None:
        """Add new cards by analyzing Japanese text from an image (e.g. photo, manga)."""
        import llm, json  # type: ignore
        try:
            vision_model = llm.get_model(model)
        except Exception:
            click.echo("Model not available for image parsing.")
            return

        prompt = f"Extract Japanese text and structure it as JSON with keys: vocabulary, kanji, grammar, phrases, idioms. Each should be a list of entries with fields as appropriate (word, readings, meanings, etc.)."
        response = vision_model.prompt(prompt, image=image_path)
        try:
            structured = json.loads(response.text())
        except Exception:
            click.echo("Could not parse structured JSON from model output.")
            return

        # Process all structured content types with duplicate tracking
        vocab_new = vocab_dup = kanji_new = kanji_dup = grammar_new = grammar_dup = phrase_new = phrase_dup = idiom_new = idiom_dup = 0

        # Vocabulary
        for item in structured.get("vocabulary", []):
            word = item.get("word", "")
            if word:
                if db.add_card(word):
                    vocab_new += 1
                else:
                    vocab_dup += 1

        # Kanji
        for item in structured.get("kanji", []):
            character = item.get("character", "")
            if character:
                if db.add_kanji(
                    character=character,
                    onyomi=item.get("onyomi", ""),
                    kunyomi=item.get("kunyomi", ""),
                    meanings=item.get("meanings", "")
                ):
                    kanji_new += 1
                else:
                    kanji_dup += 1

        # Grammar
        for item in structured.get("grammar", []):
            point = item.get("point", "")
            if point:
                if db.add_grammar(
                    point=point,
                    explanation=item.get("explanation", ""),
                    example=item.get("example", "")
                ):
                    grammar_new += 1
                else:
                    grammar_dup += 1

        # Phrases
        for item in structured.get("phrases", []):
            phrase = item.get("phrase", "")
            if phrase:
                if db.add_phrase(
                    phrase=phrase,
                    meaning=item.get("meaning", "")
                ):
                    phrase_new += 1
                else:
                    phrase_dup += 1

        # Idioms
        for item in structured.get("idioms", []):
            idiom = item.get("idiom", "")
            if idiom:
                if db.add_idiom(
                    idiom=idiom,
                    meaning=item.get("meaning", ""),
                    example=item.get("example", "")
                ):
                    idiom_new += 1
                else:
                    idiom_dup += 1

        total_new = vocab_new + kanji_new + grammar_new + phrase_new + idiom_new
        total_dup = vocab_dup + kanji_dup + grammar_dup + phrase_dup + idiom_dup

        click.echo(f"Image processed successfully!")
        click.echo(f"Added: {vocab_new} vocabulary, {kanji_new} kanji, {grammar_new} grammar, {phrase_new} phrases, {idiom_new} idioms")
        if total_dup > 0:
            click.echo(f"Skipped duplicates: {vocab_dup} vocabulary, {kanji_dup} kanji, {grammar_dup} grammar, {phrase_dup} phrases, {idiom_dup} idioms")

    @cli.command("jp-add-discord")  # type: ignore[misc]
    @click.argument("chat_log_path")
    @click.option("--model", default="gpt-5-2025-08-07", help="LLM model to use for parsing Discord logs")
    def add_discord(chat_log_path: str, model: str) -> None:
        """Parse Discord chat logs and extract Japanese learning content."""
        import llm, json  # type: ignore
        from . import structured

        try:
            with open(chat_log_path, 'r', encoding='utf-8') as f:
                chat_content = f.read()
        except FileNotFoundError:
            click.echo(f"Chat log file not found: {chat_log_path}")
            return
        except Exception as e:
            click.echo(f"Error reading chat log: {e}")
            return

        try:
            llm_model = llm.get_model(model)
        except Exception:
            click.echo(f"Model '{model}' not available for Discord log parsing.")
            return

        response = llm_model.prompt(structured.DISCORD_CHAT_PROMPT, system=chat_content)
        try:
            parsed_content = json.loads(response.text())
        except Exception:
            click.echo("Could not parse structured JSON from model output.")
            return

        # Process all structured content types with duplicate tracking
        vocab_new = vocab_dup = kanji_new = kanji_dup = grammar_new = grammar_dup = phrase_new = phrase_dup = idiom_new = idiom_dup = 0

        # Vocabulary
        for item in parsed_content.get("vocabulary", []):
            word = item.get("word", "")
            if word:
                if db.add_card(word):
                    vocab_new += 1
                else:
                    vocab_dup += 1

        # Kanji
        for item in parsed_content.get("kanji", []):
            character = item.get("character", "")
            if character:
                if db.add_kanji(
                    character=character,
                    onyomi=item.get("onyomi", ""),
                    kunyomi=item.get("kunyomi", ""),
                    meanings=item.get("meanings", "")
                ):
                    kanji_new += 1
                else:
                    kanji_dup += 1

        # Grammar
        for item in parsed_content.get("grammar", []):
            point = item.get("point", "")
            if point:
                if db.add_grammar(
                    point=point,
                    explanation=item.get("explanation", ""),
                    example=item.get("example", "")
                ):
                    grammar_new += 1
                else:
                    grammar_dup += 1

        # Phrases
        for item in parsed_content.get("phrases", []):
            phrase = item.get("phrase", "")
            if phrase:
                if db.add_phrase(
                    phrase=phrase,
                    meaning=item.get("meaning", "")
                ):
                    phrase_new += 1
                else:
                    phrase_dup += 1

        # Idioms
        for item in parsed_content.get("idioms", []):
            idiom = item.get("idiom", "")
            if idiom:
                if db.add_idiom(
                    idiom=idiom,
                    meaning=item.get("meaning", ""),
                    example=item.get("example", "")
                ):
                    idiom_new += 1
                else:
                    idiom_dup += 1

        total_new = vocab_new + kanji_new + grammar_new + phrase_new + idiom_new
        total_dup = vocab_dup + kanji_dup + grammar_dup + phrase_dup + idiom_dup

        click.echo(f"Discord chat log processed successfully!")
        click.echo(f"Added: {vocab_new} vocabulary, {kanji_new} kanji, {grammar_new} grammar, {phrase_new} phrases, {idiom_new} idioms")
        if total_dup > 0:
            click.echo(f"Skipped duplicates: {vocab_dup} vocabulary, {kanji_dup} kanji, {grammar_dup} grammar, {phrase_dup} phrases, {idiom_dup} idioms")