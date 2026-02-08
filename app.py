#!/usr/bin/env python3
"""
Japanese Learning App - Flask Web Application
Server-side rendered web interface for learning Japanese with spaced repetition.
Requires OpenAI API for intelligent exercise generation and evaluation.
"""

import os
import sys
import json
import base64
import traceback
import argparse
import concurrent.futures
from datetime import datetime
from typing import List, Optional, Dict, Any, Union
from pathlib import Path

from flask import Flask, render_template, request, redirect, url_for, session, flash, jsonify
from werkzeug.utils import secure_filename

# Check for test mode
TEST_MODE = os.environ.get("TEST_MODE", "0") == "1"
MAX_COMPLETION_TOKENS = 32768


try:
    import openai
    from openai import OpenAI
except ImportError:
    if not TEST_MODE:
        print("Error: OpenAI library is required")
        print("Please install it with: pip install openai")
        sys.exit(1)
    else:
        OpenAI = None  # type: ignore

# Global variables for AI clients
client = None
ai_model = None
study_ai_model = None

# Add the current directory to the path so we can import our modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from llm_learn_japanese import db, exercises
from llm_learn_japanese.db import CardAspect, Progress

# Check for debug mode
DEBUG = os.environ.get("DEBUG", "0") == "1"

class OpenAIModel:
    """Wrapper for OpenAI API to match expected interface."""
    def __init__(self, client: Any, model_name: str = "gpt-5.2"):
        self.client = client
        self.model_name = model_name

    def prompt(self, prompt_text: str, system: str = "") -> Any:
        """Send prompt to OpenAI and return response."""
        messages: List[Dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt_text})

        # Log the API call details
        if DEBUG:
            print(f"🤖 OpenAI API Call Details:")
            print(f"   Model: {self.model_name}")
            print(f"   System prompt length: {len(system) if system else 0} characters")
            print(f"   User prompt length: {len(prompt_text)} characters")

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,  # type: ignore
                temperature=1.0,
                max_completion_tokens=MAX_COMPLETION_TOKENS,
            )

            content = response.choices[0].message.content or ""

            # Log the response details
            if DEBUG:
                print(f"✅ OpenAI API Response:")
                print(f"   Response length: {len(content)} characters")
                print(f"   Usage: {response.usage}")

            # Return object with text() method to match expected interface
            class Response:
                def __init__(self, content: str) -> None:
                    self.content = content
                def text(self) -> str:
                    return self.content

            return Response(content)

        except Exception as e:
            if DEBUG:
                print(f"❌ OpenAI API call failed: {str(e)}")
            raise

def init_ai(api_key: Optional[str] = None,
            base_url: Optional[str] = None,
            model_name: str = "gpt-5.2",
            study_model_name: Optional[str] = None) -> None:
    """Initialize the OpenAI client and models."""
    global client, ai_model, study_ai_model

    if TEST_MODE:
        return

    if not api_key:
        api_key = os.environ.get("OPENAI_API_KEY")

    if not api_key:
        print("Warning: No API key provided. AI features will be disabled.")
        return

    if OpenAI is None:
        print("Error: OpenAI library is required but not installed.")
        return

    try:
        client_kwargs = {"api_key": api_key}
        if base_url:
            client_kwargs["base_url"] = base_url

        client = OpenAI(**client_kwargs)

        ai_model = OpenAIModel(client, model_name=model_name)
        study_ai_model = OpenAIModel(client, model_name=study_model_name or model_name)

        print(f"✅ AI initialized with model: {model_name}")
        if study_model_name and study_model_name != model_name:
            print(f"✅ Study AI initialized with model: {study_model_name}")

    except Exception as e:
        print(f"❌ Failed to initialize AI: {e}")

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'dev-secret-key-change-in-production')
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Try to initialize with environment variables by default
if not TEST_MODE:
    init_ai()

# Configure upload folder
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'webp'}

def allowed_file(filename: str) -> bool:
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.before_request
def initialize_app() -> None:
    """Initialize the database if needed."""
    if not hasattr(app, '_database_initialized'):
        try:
            if not db.is_db_initialized():
                db.init_db()
                print("✅ Database initialized on startup")
        except Exception as e:
            print(f"❌ Database startup check failed: {str(e)}")
        setattr(app, "_database_initialized", True)

@app.before_request
def ensure_username() -> None:
    """Ensure username is set in session."""
    if 'username' not in session:
        session['username'] = 'default_user'

@app.route('/')
def index() -> Any:
    """Main menu page."""
    # Get some stats for the dashboard
    try:
        progress = db.get_progress(session['username'])
        # Count cards due for review
        card = db.get_next_card()
        has_cards_due = card is not None
    except Exception as e:
        progress = None
        has_cards_due = False
        if DEBUG:
            print(f"Error getting stats: {e}")

    return render_template('index.html', 
                         username=session['username'], 
                         progress=progress, 
                         has_cards_due=has_cards_due)

@app.route('/study')
def study_session() -> Any:
    """Study session page."""
    return render_template('study.html')

@app.route('/api/next_card')
def api_next_card() -> Any:
    """API endpoint to get the next card for study."""
    try:
        exclude_id = request.args.get("exclude_id", type=int)
        aspect = db.get_next_card(exclude_id=exclude_id)
        if not aspect:
            return jsonify({'status': 'no_cards', 'message': 'No cards due for review!'})
        
        # Generate AI exercise
        study_model = study_ai_model or ai_model
        if study_model is None:
            raise ValueError("AI study model is not configured. Please ensure OpenAI credentials are set.")
        question = exercises.generate_exercise(aspect, model=study_model)
        
        return jsonify({
            'status': 'success',
            'card': {
                'id': aspect.id,
                'aspect_type': aspect.aspect_type.replace('_', ' ').title(),
                'question': question
            }
        })
    except Exception as e:
        if DEBUG:
            print(f"Error getting next card: {e}")
            traceback.print_exc()
        return jsonify({'status': 'error', 'message': f'Error: {str(e)}'})

@app.route('/api/submit_answer', methods=['POST'])
def api_submit_answer() -> Any:
    """API endpoint to submit an answer for evaluation."""
    try:
        data = request.get_json()
        card_id = int(data['card_id'])
        user_answer = data['answer']
        
        # Get the card aspect
        session_db = db.get_session()
        aspect = session_db.query(CardAspect).get(card_id)
        session_db.close()
        
        if not aspect:
            return jsonify({'status': 'error', 'message': 'Card not found'})
        
        # AI evaluation
        asked_question = data.get('question') or data.get('asked_question')
        study_model = study_ai_model or ai_model
        if study_model is None:
            raise ValueError("AI study model is not configured. Please ensure OpenAI credentials are set.")
        score, feedback = db.evaluate_answer_with_ai(
            aspect,
            user_answer,
            model=study_model,
            asked_question=asked_question
        )
        
        # Update progress and card
        is_correct = score >= 3
        db.update_progress(session['username'], is_correct)
        db.review_card(aspect.id, score)
        
        return jsonify({
            'status': 'success',
            'score': score,
            'feedback': feedback,
            'is_correct': is_correct
        })
        
    except Exception as e:
        if DEBUG:
            print(f"Error submitting answer: {e}")
            traceback.print_exc()
        return jsonify({'status': 'error', 'message': f'Error: {str(e)}'})

@app.route('/api/chat', methods=['POST'])
def api_chat() -> Any:
    """API endpoint for multi-turn chat with AI tutor."""
    try:
        data = request.get_json()
        card_id = int(data['card_id'])
        user_message = data['message']
        chat_history = data.get('chat_history', [])

        # Get the card aspect
        session_db = db.get_session()
        aspect = session_db.query(CardAspect).get(card_id)
        session_db.close()

        if not aspect:
            return jsonify({'status': 'error', 'message': 'Card not found'})

        # Build system prompt for tutoring
        system_prompt = f"""You are a helpful Japanese language tutor. The student is currently learning about:

Topic: {aspect.aspect_type.replace('_', ' ').title()}
Current Question: {aspect.prompt_template}

Your role is to:
1. Answer questions about this topic in a clear, educational manner.
2. Provide additional context, examples, or explanations when helpful.
3. Encourage the student and provide supportive feedback.
4. Stay focused on Japanese language learning, but make sure your sentences are able to be understand by an English speaker learning Japanese.

Keep responses concise but informative (5 sentences max). Use simple English explanations for complex Japanese concepts."""

        # Build conversation context
        conversation = []
        for role, msg in chat_history[-6:]:  # Last 6 messages for context
            conversation.append(f"{'Student' if role == 'user' else 'Tutor'}: {msg}")

        conversation_context = "\n".join(conversation)

        prompt = f"""Previous conversation:
{conversation_context}

Student's latest question: {user_message}

Please respond as a helpful Japanese tutor to the student's question."""

        response = ai_model.prompt(prompt, system=system_prompt)
        tutor_response = response.text().strip()

        return jsonify({
            'status': 'success',
            'response': tutor_response
        })

    except Exception as e:
        if DEBUG:
            print(f"Error in chat: {e}")
            traceback.print_exc()
        return jsonify({'status': 'error', 'message': f'Error: {str(e)}'})

@app.route('/add')
def add_content() -> Any:
    """Add content page."""
    return render_template('add_content.html')

@app.route('/add/vocabulary', methods=['GET', 'POST'])
def add_vocabulary() -> Any:
    """Add vocabulary page."""
    if request.method == 'POST':
        word = request.form['word'].strip()
        if word:
            if db.add_card(word):
                flash(f'✅ Added vocabulary: {word}', 'success')
            else:
                flash(f'⚠️ Vocabulary already exists: {word}', 'warning')
        return redirect(url_for('add_vocabulary'))
    
    return render_template('add_vocabulary.html')

@app.route('/add/kanji', methods=['GET', 'POST'])
def add_kanji() -> Any:
    """Add kanji page."""
    if request.method == 'POST':
        character = request.form['character'].strip()
        onyomi = request.form.get('onyomi', '').strip()
        kunyomi = request.form.get('kunyomi', '').strip()
        meanings = request.form.get('meanings', '').strip()
        
        if character:
            if db.add_kanji(character, onyomi, kunyomi, meanings):
                flash(f'✅ Added kanji: {character}', 'success')
            else:
                flash(f'⚠️ Kanji already exists: {character}', 'warning')
        return redirect(url_for('add_kanji'))
    
    return render_template('add_kanji.html')

@app.route('/add/grammar', methods=['GET', 'POST'])
def add_grammar() -> Any:
    """Add grammar page."""
    if request.method == 'POST':
        point = request.form['point'].strip()
        explanation = request.form.get('explanation', '').strip()
        example = request.form.get('example', '').strip()
        
        if point:
            if db.add_grammar(point, explanation, example):
                flash(f'✅ Added grammar: {point}', 'success')
            else:
                flash(f'⚠️ Grammar already exists: {point}', 'warning')
        return redirect(url_for('add_grammar'))
    
    return render_template('add_grammar.html')

@app.route('/conjugations')
def view_conjugations() -> Any:
    """View list of verb conjugations."""
    session_db = db.get_session()
    from llm_learn_japanese.db import VerbConjugation
    conjugations = session_db.query(VerbConjugation).order_by(VerbConjugation.category, VerbConjugation.label).all()
    session_db.close()
    return render_template('verb_conjugations.html', conjugations=conjugations)

@app.route('/study/conjugation/<int:conj_id>')
def study_conjugation(conj_id: int) -> Any:
    """Interactive study page for a specific verb conjugation."""
    session_db = db.get_session()
    from llm_learn_japanese.db import VerbConjugation
    conj = session_db.get(VerbConjugation, conj_id)
    session_db.close()
    if not conj:
        flash("❌ Conjugation not found", "error")
        return redirect(url_for("view_conjugations"))
    return render_template('study_conjugation.html', conjugation=conj)


@app.route('/api/next_conjugation')
def api_next_conjugation() -> Any:
    """API endpoint to fetch the next due verb conjugation based on SRS schedule."""
    try:
        session_db = db.get_session()
        from llm_learn_japanese.db import VerbConjugation
        import datetime as dt
        now = dt.datetime.now(dt.timezone.utc)
        conj = (session_db.query(VerbConjugation)
                .filter(VerbConjugation.next_review <= now)
                .order_by(VerbConjugation.next_review)
                .first())
        session_db.close()
        if not conj:
            return jsonify({'status': 'no_due', 'message': 'No conjugations due for review'})
        return redirect(url_for('study_conjugation', conj_id=conj.id))
    except Exception as e:
        if DEBUG:
            traceback.print_exc()
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/add/phrase', methods=['GET', 'POST'])
def add_phrase() -> Any:
    """Add phrase page."""
    if request.method == 'POST':
        phrase = request.form['phrase'].strip()
        meaning = request.form.get('meaning', '').strip()
        
        if phrase:
            if db.add_phrase(phrase, meaning):
                flash(f'✅ Added phrase: {phrase}', 'success')
            else:
                flash(f'⚠️ Phrase already exists: {phrase}', 'warning')
        return redirect(url_for('add_phrase'))
    
    return render_template('add_phrase.html')

@app.route('/add/idiom', methods=['GET', 'POST'])
def add_idiom() -> Any:
    """Add idiom page."""
    if request.method == 'POST':
        idiom = request.form['idiom'].strip()
        meaning = request.form.get('meaning', '').strip()
        example = request.form.get('example', '').strip()
        
        if idiom:
            if db.add_idiom(idiom, meaning, example):
                flash(f'✅ Added idiom: {idiom}', 'success')
            else:
                flash(f'⚠️ Idiom already exists: {idiom}', 'warning')
        return redirect(url_for('add_idiom'))
    
    return render_template('add_idiom.html')

@app.route('/progress')
def view_progress() -> Any:
    """View progress page with advanced analytics."""
    username = session['username']
    global_progress = db.get_progress(username)
    # Grab last 1-2 days of daily progress so we can show today's card
    daily_progress = db.get_daily_progress(username, days=1)
    today_progress = daily_progress[-1] if daily_progress else None
    # Count due review cards for today (only cards the user has actually seen before: success_count > 0)
    from sqlalchemy import func
    session_db = db.get_session()
    due_count = (session_db.query(db.CardAspect.parent_type, db.CardAspect.parent_id)
                 .filter(db.CardAspect.success_count > 0)
                 .filter(db.CardAspect.next_review <= datetime.utcnow())
                 .group_by(db.CardAspect.parent_type, db.CardAspect.parent_id)
                 .count())
    session_db.close()

    # Compute advanced analytics
    try:
        analytics = db.get_analytics_data(username)
    except Exception as e:
        if DEBUG:
            traceback.print_exc()
        analytics = {}

    return render_template('progress.html',
                           username=username,
                           progress=global_progress,
                           today_progress=today_progress,
                           due_count=due_count,
                           analytics=analytics)

@app.route('/progress_data')
def progress_data() -> Any:
    """Return daily progress data as JSON for charts."""
    data = db.get_daily_progress(session['username'], days=30)
    return jsonify(data)

@app.route('/api/analytics')
def api_analytics() -> Any:
    """Return advanced analytics data as JSON."""
    try:
        analytics = db.get_analytics_data(session['username'])
        return jsonify(analytics)
    except Exception as e:
        if DEBUG:
            traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route('/settings', methods=['GET', 'POST'])
def settings() -> Any:
    """Settings page."""
    from llm_learn_japanese.db import UserSettings, get_session

    if request.method == 'POST':
        username = request.form['username'].strip()
        max_new_per_day = int(request.form.get('max_new_per_day', 20))
        if username:
            session['username'] = username
            flash(f'✅ Username set to: {username}', 'success')
        # Save user settings
        db_sess = get_session()
        settings = db_sess.query(UserSettings).filter_by(user=username).first()
        if not settings:
            settings = UserSettings(user=username, max_new_per_day=max_new_per_day)
            db_sess.add(settings)
        else:
            settings.max_new_per_day = max_new_per_day
        db_sess.commit()
        db_sess.close()
        flash(f'✅ Max new cards/day set to {max_new_per_day}', 'success')
        return redirect(url_for('settings'))
    
    # GET: show current values
    db_sess = get_session()
    settings = db_sess.query(UserSettings).filter_by(user=session['username']).first()
    db_sess.close()
    max_new = settings.max_new_per_day if settings else 20

    return render_template('settings.html', username=session['username'], max_new_per_day=max_new)

@app.route('/process_image', methods=['GET', 'POST'])
def process_image() -> Any:
    """Process image file page."""
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file selected', 'error')
            return redirect(request.url)
        
        file = request.files['file']
        if file.filename == '':
            flash('No file selected', 'error')
            return redirect(request.url)
        
        if file and isinstance(file.filename, str) and file.filename != "" and allowed_file(file.filename):
            try:
                # Read and encode image
                image_data = base64.b64encode(file.read()).decode('utf-8')
                
                # Create vision model prompt
                from llm_learn_japanese import structured
                prompt = structured.PROMPT_INSTRUCTIONS
                
                # Use OpenAI vision model
                messages = [
                    {"role": "system", "content": "You are a Japanese learning assistant."},
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{image_data}"
                                }
                            }
                        ]
                    }
                ]
                
                response = client.chat.completions.create(
                    model="gpt-5-2025-08-07",
                    messages=messages,  # type: ignore
                    max_completion_tokens=MAX_COMPLETION_TOKENS,
                    temperature=1.0,
                )
                
                content = response.choices[0].message.content or "{}"
                
                try:
                    structured_content = json.loads(content)
                except json.JSONDecodeError:
                    # Try to extract JSON from response
                    import re
                    json_match = re.search(r'\{.*\}', content, re.DOTALL)
                    if json_match:
                        try:
                            structured_content = json.loads(json_match.group())
                        except json.JSONDecodeError:
                            structured_content = {"vocabulary": [], "kanji": [], "grammar": [], "phrases": [], "idioms": []}
                    else:
                        structured_content = {"vocabulary": [], "kanji": [], "grammar": [], "phrases": [], "idioms": []}
                
                # Process extracted content
                results = process_structured_content(structured_content)
                
                flash(f'✅ Image processed! Added: {results["total_new"]} items, Skipped: {results["total_dup"]} duplicates', 'success')
                return render_template('process_image.html', results=results)
                
            except Exception as e:
                flash(f'❌ Image processing failed: {str(e)}', 'error')
                if DEBUG:
                    traceback.print_exc()
        else:
            flash('Invalid file type. Please upload an image file.', 'error')
    
    return render_template('process_image.html')

def process_renpy_chunks(chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Convert parsed RenPy dialogue chunks into a structured format compatible with the DB processor.
    Each chunk is turned into a 'phrases' batch for consistency with existing pipeline.
    """
    structured_content: Dict[str, Any] = {"vocabulary": [], "kanji": [], "grammar": [], "phrases": [], "idioms": []}
    for chunk in chunks:
        text_block = "\n".join([f'{d["speaker"]}: {d["text"]}' for d in chunk["dialogues"]])
        structured_content["phrases"].append({
            "phrase": text_block,
            "meaning": f"RenPy dialogue chunk {chunk['chunk_id']}"
        })
    return structured_content

def parse_renpy_file(content: str, chunk_size: int = 20) -> List[Dict[str, Any]]:
    """
    Parse a RenPy file and extract dialogue into chunks of ~20 lines.
    Non-dialogue lines (scene, play, jump, comments) are ignored.
    Returns a list of chunks, each containing structured dialogue.
    """
    dialogues = []
    for line_num, line in enumerate(content.splitlines(), 1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue

        # Skip RenPy commands (scene, play, jump, show, hide, with, etc.)
        renpy_commands = ['scene ', 'play ', 'jump ', 'show ', 'hide ', 'with ', 'stop ', 'window ']
        if any(line.startswith(cmd) for cmd in renpy_commands):
            continue

        # Match dialogue in the form: character "dialogue" or character 'dialogue'
        if '"' in line or "'" in line or "「" in line:
            # Find the first space to separate speaker from dialogue
            space_pos = line.find(' ')
            if space_pos > 0:
                speaker = line[:space_pos].strip()
                speech = line[space_pos + 1:].strip()

                # Check if this looks like dialogue (starts with quote)
                if (speech.startswith('"') and speech.endswith('"')) or \
                   (speech.startswith("'") and speech.endswith("'")) or \
                   (speech.startswith("「") and speech.endswith("」")) or \
                   (speech.startswith('"') or speech.startswith("'") or speech.startswith("「")):
                    dialogues.append({
                        "speaker": speaker,
                        "text": speech,
                        "line_number": line_num
                    })

    # Chunk dialogues into groups of ~chunk_size
    chunks: List[Dict[str, Any]] = []
    for i in range(0, len(dialogues), chunk_size):
        chunk = dialogues[i:i+chunk_size]
        chunks.append({
            "chunk_id": len(chunks) + 1,
            "dialogues": chunk,
            "start_line": chunk[0]["line_number"] if chunk else 0,
            "end_line": chunk[-1]["line_number"] if chunk else 0
        })
    return chunks

def process_structured_content(structured_content: Dict[str, Any]) -> Dict[str, int]:
    """Process structured content and add to database."""
    results = {
        'vocab_new': 0, 'vocab_dup': 0,
        'kanji_new': 0, 'kanji_dup': 0,
        'grammar_new': 0, 'grammar_dup': 0,
        'phrase_new': 0, 'phrase_dup': 0,
        'idiom_new': 0, 'idiom_dup': 0
    }
    
    # Vocabulary
    for item in structured_content.get("vocabulary", []):
        word = item.get("word", "") if isinstance(item, dict) else ""
        if word and word.strip():
            if db.add_card(word):
                results['vocab_new'] += 1
            else:
                results['vocab_dup'] += 1
    
    # Kanji
    for item in structured_content.get("kanji", []):
        character = item.get("character", "") if isinstance(item, dict) else ""
        if character and character.strip():
            if db.add_kanji(
                character=character,
                onyomi=item.get("onyomi", ""),
                kunyomi=item.get("kunyomi", ""),
                meanings=item.get("meanings", "")
            ):
                results['kanji_new'] += 1
            else:
                results['kanji_dup'] += 1
    
    # Grammar
    for item in structured_content.get("grammar", []):
        point = item.get("point", "") if isinstance(item, dict) else ""
        if point and point.strip():
            if db.add_grammar(
                point=point,
                explanation=item.get("explanation", ""),
                example=item.get("example", "")
            ):
                results['grammar_new'] += 1
            else:
                results['grammar_dup'] += 1
    
    # Phrases
    for item in structured_content.get("phrases", []):
        phrase = item.get("phrase", "") if isinstance(item, dict) else ""
        if phrase and phrase.strip():
            if db.add_phrase(
                phrase=phrase,
                meaning=item.get("meaning", "")
            ):
                results['phrase_new'] += 1
            else:
                results['phrase_dup'] += 1
    
    # Idioms
    for item in structured_content.get("idioms", []):
        idiom = item.get("idiom", "") if isinstance(item, dict) else ""
        if idiom and idiom.strip():
            if db.add_idiom(
                idiom=idiom,
                meaning=item.get("meaning", ""),
                example=item.get("example", "")
            ):
                results['idiom_new'] += 1
            else:
                results['idiom_dup'] += 1
    
    results['total_new'] = results['vocab_new'] + results['kanji_new'] + results['grammar_new'] + results['phrase_new'] + results['idiom_new']
    results['total_dup'] = results['vocab_dup'] + results['kanji_dup'] + results['grammar_dup'] + results['phrase_dup'] + results['idiom_dup']
    
    return results

@app.route('/process_discord', methods=['GET', 'POST'])
def process_discord() -> Any:
    """Process Discord log file page."""
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file selected', 'error')
            return redirect(request.url)

        file = request.files['file']
        if file.filename == '':
            flash('No file selected', 'error')
            return redirect(request.url)

        if file and isinstance(file.filename, str) and file.filename.lower().endswith(('.txt', '.log')):
            try:
                # Read file content
                chat_content = file.read().decode('utf-8')

                # Use AI to extract Japanese content
                from llm_learn_japanese import structured

                if DEBUG:
                    print(f"🤖 Processing Discord chat with AI:")
                    print(f"   Chat content length: {len(chat_content)} characters")
                    print(f"   Content preview: {chat_content[:200]}...")

                parsed_content: Dict[str, Any] = {"vocabulary": [], "kanji": [], "grammar": [], "phrases": [], "idioms": []}
                try:
                    response = ai_model.prompt(
                        chat_content,
                        system=structured.DISCORD_CHAT_PROMPT
                    )
                    response_text = response.text()
                    if DEBUG:
                        print(f"📤 Discord AI response length: {len(response_text)} characters")
                        print(f"📤 Discord AI response preview: {response_text[:300]}...")

                    parsed_content = json.loads(response_text)
                    if DEBUG:
                        print(f"✅ Discord JSON parsing successful")
                        print(f"📊 Parsed structure keys: {list(parsed_content.keys()) if isinstance(parsed_content, dict) else 'Not a dict'}")
                except json.JSONDecodeError as e:
                    if DEBUG:
                        print(f"❌ Discord JSON parsing failed: {e}")
                    parsed_content = {"vocabulary": [], "kanji": [], "grammar": [], "phrases": [], "idioms": []}
                except Exception as e:
                    if DEBUG:
                        print(f"❌ Discord processing failed: {e}")
                    flash(f'⚠️ Discord log processing failed: {str(e)}', 'warning')

                # Process extracted content
                results = process_structured_content(parsed_content)

                flash(f'✅ Discord log processed! Added: {results["total_new"]} items, Skipped: {results["total_dup"]} duplicates', 'success')
                return render_template('process_discord.html', results=results)

            except Exception as e:
                flash(f'❌ Discord log processing failed: {str(e)}', 'error')
                if DEBUG:
                    traceback.print_exc()
        else:
            flash('Invalid file type. Please upload a text or log file.', 'error')

    return render_template('process_discord.html')

@app.route('/process_renpy', methods=['GET', 'POST'])
def process_renpy() -> Any:
    """Process RenPy script file page."""
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file selected', 'error')
            return redirect(request.url)

        file = request.files['file']
        if file.filename == '':
            flash('No file selected', 'error')
            return redirect(request.url)

        if file and isinstance(file.filename, str) and file.filename.lower().endswith(('.rpy', '.renpy', '.txt')):
            try:
                # Read file content
                renpy_content = file.read().decode('utf-8')

                # Parse RenPy dialogues into chunks
                chunks = parse_renpy_file(renpy_content)

                # Use AI to extract Japanese content from each chunk
                from llm_learn_japanese import structured

                if DEBUG:
                    print(f"🤖 Processing RenPy script with AI:")
                    print(f"   Content length: {len(renpy_content)} characters")
                    print(f"   Content preview: {renpy_content[:200]}...")
                    print(f"   Number of chunks: {len(chunks)}")

                # Track overall results
                total_results = {
                    'vocab_new': 0, 'vocab_dup': 0,
                    'kanji_new': 0, 'kanji_dup': 0,
                    'grammar_new': 0, 'grammar_dup': 0,
                    'phrase_new': 0, 'phrase_dup': 0,
                    'idiom_new': 0, 'idiom_dup': 0
                }

                # Enhanced prompt for more concise responses
                concise_prompt = """Extract Japanese content and return ONLY essential items in valid JSON format.

FOCUS ON:
- Key vocabulary words (no character names unless Japanese)
- Important kanji characters
- Grammar patterns actually used
- Meaningful phrases
- Idioms only if present

Be CONCISE - only include significant learning content.

JSON format:
{"vocabulary": [{"word": "...", "reading": "...", "meaning": "..."}], "kanji": [{"character": "...", "onyomi": "...", "kunyomi": "...", "meanings": "..."}], "grammar": [{"point": "...", "explanation": "...", "example": "..."}], "phrases": [{"phrase": "...", "meaning": "..."}], "idioms": [{"idiom": "...", "meaning": "...", "example": "..."}]}"""

                # Process each chunk individually with incremental database saves
                for i, chunk in enumerate(chunks, 1):
                    max_retries = 2
                    retry_count = 0
                    chunk_success = False

                    while retry_count <= max_retries and not chunk_success:
                        try:
                            # Convert chunk dialogues to text
                            chunk_text = "\n".join([f'{d["speaker"]}: {d["text"]}' for d in chunk["dialogues"]])

                            if DEBUG:
                                print(f"🔄 Processing chunk {i}/{len(chunks)} (attempt {retry_count + 1}):")
                                print(f"   Chunk size: {len(chunk_text)} characters")
                                print(f"   Lines {chunk['start_line']}-{chunk['end_line']}")

                            # Send chunk to AI with concise prompt
                            response = ai_model.prompt(
                                chunk_text,
                                system=concise_prompt
                            )
                            response_text = response.text().strip()

                            if DEBUG:
                                print(f"📤 Chunk {i} AI response length: {len(response_text)} characters")
                                print(f"📤 Chunk {i} AI response preview: {response_text[:200]}...")

                            # Skip empty responses
                            if not response_text:
                                if DEBUG:
                                    print(f"⚠️ Chunk {i} returned empty response, retrying...")
                                retry_count += 1
                                continue

                            try:
                                chunk_structured = json.loads(response_text)
                                if DEBUG:
                                    print(f"✅ Chunk {i} JSON parsing successful")
                                    print(f"📊 Chunk {i} structure keys: {list(chunk_structured.keys()) if isinstance(chunk_structured, dict) else 'Not a dict'}")

                                # Immediately process this chunk's content to database
                                chunk_results = process_structured_content(chunk_structured)

                                # Add to total results
                                for key in total_results:
                                    total_results[key] += chunk_results.get(key, 0)

                                if DEBUG:
                                    print(f"💾 Chunk {i} saved to database: {chunk_results['total_new']} new, {chunk_results['total_dup']} duplicates")

                                chunk_success = True

                            except json.JSONDecodeError as e:
                                if DEBUG:
                                    print(f"❌ Chunk {i} JSON parsing failed: {e}")

                                # Try to extract JSON from response
                                import re
                                json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
                                if json_match:
                                    try:
                                        chunk_structured = json.loads(json_match.group())
                                        # Process this chunk immediately
                                        chunk_results = process_structured_content(chunk_structured)

                                        # Add to total results
                                        for key in total_results:
                                            total_results[key] += chunk_results.get(key, 0)

                                        if DEBUG:
                                            print(f"💾 Chunk {i} extracted and saved: {chunk_results['total_new']} new, {chunk_results['total_dup']} duplicates")

                                        chunk_success = True
                                    except json.JSONDecodeError:
                                        if DEBUG:
                                            print(f"❌ Chunk {i} JSON extraction also failed, retrying...")
                                        retry_count += 1
                                else:
                                    if DEBUG:
                                        print(f"❌ Chunk {i} no JSON found in response, retrying...")
                                    retry_count += 1

                        except Exception as e:
                            if DEBUG:
                                print(f"❌ Chunk {i} processing failed: {e}")
                            retry_count += 1

                    if not chunk_success:
                        if DEBUG:
                            print(f"❌ Chunk {i} failed after {max_retries + 1} attempts, skipping...")

                # Calculate final totals
                total_results['total_new'] = total_results['vocab_new'] + total_results['kanji_new'] + total_results['grammar_new'] + total_results['phrase_new'] + total_results['idiom_new']
                total_results['total_dup'] = total_results['vocab_dup'] + total_results['kanji_dup'] + total_results['grammar_dup'] + total_results['phrase_dup'] + total_results['idiom_dup']

                if DEBUG:
                    print(f"📊 Final totals:")
                    print(f"   Total new: {total_results['total_new']}")
                    print(f"   Total duplicates: {total_results['total_dup']}")

                flash(f'✅ RenPy file processed! Added: {total_results["total_new"]} items, Skipped: {total_results["total_dup"]} duplicates', 'success')
                return render_template('process_renpy.html', results=total_results, chunks=chunks)

            except Exception as e:
                flash(f'❌ RenPy file processing failed: {str(e)}', 'error')
                if DEBUG:
                    traceback.print_exc()
        else:
            flash('Invalid file type. Please upload a RenPy (.rpy) or text file.', 'error')
    return render_template('process_renpy.html')

@app.route('/process_raw_text', methods=['GET', 'POST'])
def process_raw_text() -> Any:
    """Process raw text input page."""
    if request.method == 'POST':
        raw_text = request.form.get('raw_text', '').strip()
        if not raw_text:
            flash('⚠️ Please enter some text to process.', 'warning')
            return redirect(request.url)

        try:
            from llm_learn_japanese import structured

            if DEBUG:
                print(f"🤖 Processing raw text input with AI:")
                print(f"   Content length: {len(raw_text)} characters")
                print(f"   Preview: {raw_text[:200]}...")

            parsed_content: Dict[str, Any] = {"vocabulary": [], "kanji": [], "grammar": [], "phrases": [], "idioms": []}
            try:
                response = ai_model.prompt(
                    raw_text,
                    system=structured.RAW_TEXT_PROMPT
                )
                response_text = response.text()
                if DEBUG:
                    print(f"📤 Raw text AI response length: {len(response_text)} characters")
                    print(f"📤 Raw text AI response preview: {response_text[:300]}...")

                parsed_content = json.loads(response_text)
                if DEBUG:
                    print(f"✅ Raw text JSON parsing successful")
            except json.JSONDecodeError as e:
                if DEBUG:
                    print(f"❌ Raw text JSON parsing failed: {e}")
                parsed_content = {"vocabulary": [], "kanji": [], "grammar": [], "phrases": [], "idioms": []}
            except Exception as e:
                if DEBUG:
                    print(f"❌ Raw text processing failed: {e}")
                flash(f'⚠️ Raw text processing failed: {str(e)}', 'warning')

            results = process_structured_content(parsed_content)

            flash(f'✅ Raw text processed! Added: {results["total_new"]} items, Skipped: {results["total_dup"]} duplicates', 'success')
            return render_template('process_raw_text.html', results=results, raw_text=raw_text)

        except Exception as e:
            flash(f'❌ Raw text processing failed: {str(e)}', 'error')
            if DEBUG:
                traceback.print_exc()

    return render_template('process_raw_text.html')

@app.route('/manual_review', methods=['GET', 'POST'])
def manual_review() -> Any:
    """Manual card review page."""
    if request.method == 'POST':
        try:
            aspect_id = int(request.form['aspect_id'])
            quality = int(request.form['quality'])
            
            if quality < 0 or quality > 5:
                flash('❌ Quality must be between 0-5', 'error')
                return redirect(url_for('manual_review'))
            
            db.review_card(aspect_id, quality)
            
            if quality >= 3:
                flash(f'✅ Good! Aspect {aspect_id} scheduled for later review.', 'success')
            else:
                flash(f'✅ That\'s okay! Aspect {aspect_id} will be reviewed again soon.', 'info')
                
        except ValueError:
            flash('❌ Please enter valid numbers for aspect ID and quality', 'error')
        except Exception as e:
            flash(f'❌ Review failed: {str(e)}', 'error')
        
        return redirect(url_for('manual_review'))
    
    return render_template('manual_review.html')

@app.route('/save_message', methods=['GET', 'POST'])
def save_message() -> Any:
    """Save message page."""
    if request.method == 'POST':
        message = request.form['message'].strip()
        if message:
            try:
                db.save_message(message)
                flash('✅ Message saved successfully', 'success')
            except Exception as e:
                flash(f'❌ Failed to save message: {str(e)}', 'error')
        return redirect(url_for('save_message'))
    
    return render_template('save_message.html')

@app.route('/ai_status')
def ai_status() -> Any:
    """AI status page."""
    return render_template('ai_status.html')


# ----------------------------------------------------------------------
# Bunpro Grammar Routes
# ----------------------------------------------------------------------

@app.route('/bunpro')
def bunpro_study() -> Any:
    """Bunpro Grammar study page."""
    # Auto-import CSV on first visit if table is empty
    try:
        progress = db.get_bunpro_progress()
        if progress['total'] == 0:
            csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    'data', 'bunpro_jlptplus_usage_ranked_heuristic.csv')
            if os.path.exists(csv_path):
                db.import_bunpro_csv(csv_path)
                print("✅ Auto-imported Bunpro CSV on first visit")
    except Exception as e:
        print(f"⚠️ Bunpro auto-import check: {e}")
        # Table might not exist yet; create it
        try:
            db.Base.metadata.create_all(bind=db.engine)
            csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    'data', 'bunpro_jlptplus_usage_ranked_heuristic.csv')
            if os.path.exists(csv_path):
                db.import_bunpro_csv(csv_path)
                print("✅ Created bunpro_grammar table and imported CSV")
        except Exception as e2:
            print(f"❌ Bunpro table creation failed: {e2}")

    return render_template('bunpro.html', batch_size=app.config.get('BATCH_SIZE', 5))


@app.route('/api/bunpro/batch')
def api_bunpro_batch() -> Any:
    """Batch API: generate multiple Bunpro MC questions at once using parallel execution.

    Query params:
        count: number of questions to generate (default from config, max 15)
    """
    default_count = app.config.get('BATCH_SIZE', 5)
    count = min(request.args.get('count', default_count, type=int), 15)
    exclude_param = request.args.get('exclude_ids', '', type=str)
    client_exclude: set = set(int(x) for x in exclude_param.split(',') if x.strip()) if exclude_param else set()

    # 1. Fetch all cards first (sequential DB access is fast)
    cards_to_process = []
    seen_ids: set = set(client_exclude)
    pending_new_count = 0

    for _ in range(count):
        card = db.get_next_bunpro_card(exclude_ids=seen_ids, pending_new_count=pending_new_count)
        if not card:
            break
        if card.id in seen_ids:
            continue
        seen_ids.add(card.id)
        if card.success_count <= 0:
            pending_new_count += 1

        # Get distractors (DB access)
        distractors = db.get_bunpro_distractors(card, count=3)
        cards_to_process.append((card, distractors))

    if not cards_to_process:
        return jsonify({'status': 'no_cards', 'message': 'No grammar cards available', 'cards': []})

    # 2. Generate quizzes in parallel
    study_model = study_ai_model or ai_model
    if study_model is None:
        return jsonify({'status': 'error', 'message': 'AI model not configured'})

    def generate_quiz_task(card_tuple):
        card, distractors = card_tuple
        try:
            quiz_questions = exercises.generate_bunpro_quiz(card, distractors, model=study_model)
            return {
                'id': card.id,
                'grammar': card.grammar,
                'meaning': card.meaning,
                'jlpt': card.jlpt,
                'rank': card.rank,
                'url': card.url,
                'is_new': card.success_count <= 0,
                'questions': quiz_questions,
            }
        except Exception as e:
            if DEBUG:
                print(f"⚠️ Error generating Bunpro question for {card.grammar}: {e}")
            return None

    cards_data = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        future_to_card = {executor.submit(generate_quiz_task, c): c for c in cards_to_process}
        for future in concurrent.futures.as_completed(future_to_card):
            result = future.result()
            if result:
                cards_data.append(result)

    if not cards_data:
        return jsonify({'status': 'error', 'message': 'Failed to generate questions', 'cards': []})

    return jsonify({'status': 'success', 'cards': cards_data})


@app.route('/api/bunpro/answer', methods=['POST'])
def api_bunpro_answer() -> Any:
    """Submit a Bunpro quiz result (aggregate of 3 questions)."""
    try:
        data = request.get_json()
        card_id = int(data['card_id'])

        # Manual recall override: user picks quality directly (Anki-style)
        if 'manual_quality' in data:
            quality = max(1, min(5, int(data['manual_quality'])))
            is_correct = quality >= 3
            feedback = f"Manual override: quality {quality}"

        # Support both old single-answer format and new multi-answer format
        elif 'correct_count' in data:
            # New format
            correct_count = int(data['correct_count'])
            total_count = int(data.get('total_count', 3))

            # Calculate quality score (0-5) for SM-2 algorithm
            # 3/3 = 5 (Perfect), 2/3 = 4 (Good), 1/3 = 2 (Fail), 0/3 = 1 (Fail)
            if correct_count == total_count:
                quality = 5
            elif correct_count >= total_count - 1:
                quality = 4
            elif correct_count >= total_count * 0.5:
                quality = 3
            elif correct_count > 0:
                quality = 2
            else:
                quality = 1

            is_correct = quality >= 3
            feedback = f"You got {correct_count} out of {total_count} correct."

        else:
            # Legacy format (single question)
            selected_key = data['selected_key']
            correct_key = data['correct_key']
            is_correct = selected_key == correct_key
            quality = 5 if is_correct else 1
            feedback = "Correct!" if is_correct else "Incorrect."

        # Update SRS scheduling
        db.review_bunpro_card(card_id, quality, user=session.get('username', 'default_user'))

        # Update progress
        db.update_progress(session.get('username', 'default_user'), is_correct)

        return jsonify({
            'status': 'success',
            'is_correct': is_correct,
            'score': quality,
            'feedback': feedback,
        })

    except Exception as e:
        if DEBUG:
            print(f"Error in bunpro answer: {e}")
            traceback.print_exc()
        return jsonify({'status': 'error', 'message': str(e)})


@app.route('/api/bunpro/lesson', methods=['POST'])
def api_bunpro_lesson() -> Any:
    """Generate an AI lesson for a Bunpro grammar point."""
    try:
        data = request.get_json()
        card_id = int(data['card_id'])

        session_db = db.get_session()
        card = session_db.get(db.BunproGrammar, card_id)
        session_db.close()

        if not card:
            return jsonify({'status': 'error', 'message': 'Card not found'})

        # Mark lesson as viewed
        db.mark_bunpro_lesson_viewed(card_id)

        # Generate lesson
        study_model = study_ai_model or ai_model
        if study_model is None:
            raise ValueError("AI model not configured")

        lesson = exercises.generate_bunpro_lesson(card, model=study_model)

        return jsonify({'status': 'success', 'lesson': lesson})

    except Exception as e:
        if DEBUG:
            print(f"Error generating bunpro lesson: {e}")
            traceback.print_exc()
        return jsonify({'status': 'error', 'message': str(e)})


@app.route('/api/bunpro/chat', methods=['POST'])
def api_bunpro_chat() -> Any:
    """Chat with AI tutor about a Bunpro grammar point."""
    try:
        data = request.get_json()
        card_id = int(data['card_id'])
        user_message = data['message']
        chat_history = data.get('chat_history', [])

        session_db = db.get_session()
        card = session_db.get(db.BunproGrammar, card_id)
        session_db.close()

        if not card:
            return jsonify({'status': 'error', 'message': 'Card not found'})

        system_prompt = f"""You are a helpful Japanese grammar tutor. The student is studying:

Grammar point: 「{card.grammar}」
Meaning: {card.meaning}
JLPT Level: {card.jlpt}
Register: {card.register}

Your role is to:
1. Answer questions about this grammar point clearly
2. Provide examples and comparisons with similar grammar
3. Be encouraging and supportive
4. Keep responses concise (3-4 sentences max)
5. Use English explanations accessible to a learner"""

        conversation = []
        for role, msg in chat_history[-6:]:
            conversation.append(f"{'Student' if role == 'user' else 'Tutor'}: {msg}")

        prompt = f"""Previous conversation:
{chr(10).join(conversation)}

Student's question: {user_message}

Respond helpfully about 「{card.grammar}」."""

        chat_model = ai_model  # Use lighter model for chat
        if chat_model is None:
            raise ValueError("AI model not configured")

        response = chat_model.prompt(prompt, system=system_prompt)
        tutor_response = response.text().strip()

        return jsonify({'status': 'success', 'response': tutor_response})

    except Exception as e:
        if DEBUG:
            print(f"Error in bunpro chat: {e}")
            traceback.print_exc()
        return jsonify({'status': 'error', 'message': str(e)})


@app.route('/api/bunpro/progress')
def api_bunpro_progress() -> Any:
    """Get Bunpro Grammar progress stats."""
    try:
        progress = db.get_bunpro_progress()
        return jsonify({'status': 'success', 'progress': progress})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})


# ----------------------------------------------------------------------
# Top Kanji Routes
# ----------------------------------------------------------------------

def _ensure_top_kanji_imported() -> None:
    """Auto-import kanji CSV on first visit if table is empty."""
    try:
        progress = db.get_top_kanji_progress()
        if progress['total'] == 0:
            csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'top_10000_kanji.csv')
            if os.path.exists(csv_path):
                db.Base.metadata.create_all(bind=db.engine)
                db.import_top_kanji_csv(csv_path)
    except Exception:
        try:
            db.Base.metadata.create_all(bind=db.engine)
            csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'top_10000_kanji.csv')
            if os.path.exists(csv_path):
                db.import_top_kanji_csv(csv_path)
        except Exception as e2:
            print(f"❌ Kanji import failed: {e2}")


@app.route('/kanji-study')
def kanji_study() -> Any:
    """Top Kanji study page."""
    _ensure_top_kanji_imported()
    return render_template('mc_study.html',
        title='Top Kanji', icon='漢', subtitle='Learn the most frequent kanji with multiple-choice quizzes',
        batch_url=url_for('api_kanji_batch'), answer_url=url_for('api_kanji_answer'),
        lesson_url=url_for('api_kanji_lesson'), chat_url=url_for('api_kanji_chat'),
        new_session_url=url_for('kanji_study'), display_field='kanji',
        batch_size=app.config.get('BATCH_SIZE', 5))


@app.route('/api/kanji/batch')
def api_kanji_batch() -> Any:
    """Batch generate kanji MC questions using parallel execution. Handles lazy annotation."""
    default_count = app.config.get('BATCH_SIZE', 5)
    count = min(request.args.get('count', default_count, type=int), 15)
    exclude_param = request.args.get('exclude_ids', '', type=str)
    client_exclude: set = set(int(x) for x in exclude_param.split(',') if x.strip()) if exclude_param else set()

    # 1. Fetch cards and handle lazy annotation (sequential because annotation writes to DB)
    cards_to_process = []
    seen_ids: set = set(client_exclude)
    pending_new_count = 0
    study_model = study_ai_model or ai_model

    if not study_model:
        return jsonify({'status': 'error', 'message': 'AI model not configured'})

    for _ in range(count):
        try:
            card = db.get_next_top_kanji(exclude_ids=seen_ids, pending_new_count=pending_new_count)
            if not card or card.id in seen_ids:
                break
            seen_ids.add(card.id)
            if card.success_count <= 0:
                pending_new_count += 1

            # Lazy LLM annotation (must be done before parallel step if writing to DB)
            if not card.annotated:
                annotations = exercises.annotate_kanji_with_llm(card, study_model)
                db.annotate_top_kanji(card.id, **annotations)
                # Refresh card object
                card.on_readings = annotations['on_readings']
                card.kun_readings = annotations['kun_readings']
                card.meanings_en = annotations['meanings_en']
                card.jlpt_level = annotations['jlpt_level']
                card.annotated = True

            if not card.annotated or not card.meanings_en:
                continue

            distractors = db.get_kanji_distractors(card, count=3)
            cards_to_process.append((card, distractors))

        except Exception as e:
            if DEBUG:
                print(f"⚠️ Error fetching kanji card: {e}")
            continue

    if not cards_to_process:
        return jsonify({'status': 'no_cards', 'message': 'No kanji cards available', 'cards': []})

    # 2. Generate quizzes in parallel
    def generate_kanji_task(card_tuple):
        card, distractors = card_tuple
        try:
            quiz_questions = exercises.generate_kanji_quiz(card, distractors, model=study_model)
            return {
                'id': card.id, 'kanji': card.kanji, 'display_text': card.kanji,
                'meanings_en': card.meanings_en, 'on_readings': card.on_readings,
                'kun_readings': card.kun_readings, 'jlpt_level': card.jlpt_level,
                'rank': card.rank, 'is_new': card.success_count <= 0,
                'questions': quiz_questions,
            }
        except Exception as e:
            if DEBUG:
                print(f"⚠️ Kanji question error for {card.kanji}: {e}")
            return None

    cards_data = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        future_to_card = {executor.submit(generate_kanji_task, c): c for c in cards_to_process}
        for future in concurrent.futures.as_completed(future_to_card):
            result = future.result()
            if result:
                cards_data.append(result)

    if not cards_data:
        return jsonify({'status': 'error', 'message': 'Failed to generate questions', 'cards': []})

    return jsonify({'status': 'success', 'cards': cards_data})


@app.route('/api/kanji/answer', methods=['POST'])
def api_kanji_answer() -> Any:
    """Submit a kanji quiz result (aggregate of 3 questions)."""
    try:
        data = request.get_json()
        card_id = int(data['card_id'])

        # Manual recall override: user picks quality directly (Anki-style)
        if 'manual_quality' in data:
            quality = max(1, min(5, int(data['manual_quality'])))
            is_correct = quality >= 3
            feedback = f"Manual override: quality {quality}"

        elif 'correct_count' in data:
            correct_count = int(data['correct_count'])
            total_count = int(data.get('total_count', 3))

            # Calculate quality score (0-5) for SM-2 algorithm
            # 3/3 = 5 (Perfect), 2/3 = 4 (Good), 1/3 = 2 (Fail), 0/3 = 1 (Fail)
            if correct_count == total_count:
                quality = 5
            elif correct_count >= total_count - 1:
                quality = 4
            elif correct_count >= total_count * 0.5:
                quality = 3
            elif correct_count > 0:
                quality = 2
            else:
                quality = 1

            is_correct = quality >= 3
            feedback = f"You got {correct_count} out of {total_count} correct."
        else:
            # Legacy
            selected_key = data['selected_key']
            correct_key = data['correct_key']
            is_correct = selected_key == correct_key
            quality = 5 if is_correct else 1
            feedback = "Correct!" if is_correct else "Incorrect."

        db.review_top_kanji(card_id, quality, user=session.get('username', 'default_user'))
        db.update_progress(session.get('username', 'default_user'), is_correct)

        return jsonify({'status': 'success', 'is_correct': is_correct, 'score': quality, 'feedback': feedback})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})


@app.route('/api/kanji/lesson', methods=['POST'])
def api_kanji_lesson() -> Any:
    """Generate a kanji lesson."""
    try:
        data = request.get_json()
        card_id = int(data['card_id'])
        session_db = db.get_session()
        card = session_db.get(db.TopKanji, card_id)
        session_db.close()
        if not card:
            return jsonify({'status': 'error', 'message': 'Card not found'})
        db.mark_top_kanji_lesson_viewed(card_id)
        study_model = study_ai_model or ai_model
        if not study_model:
            raise ValueError("AI model not configured")
        lesson = exercises.generate_kanji_lesson(card, model=study_model)
        return jsonify({'status': 'success', 'lesson': lesson})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})


@app.route('/api/kanji/chat', methods=['POST'])
def api_kanji_chat() -> Any:
    """Chat about a kanji."""
    try:
        data = request.get_json()
        card_id = int(data['card_id'])
        session_db = db.get_session()
        card = session_db.get(db.TopKanji, card_id)
        session_db.close()
        if not card:
            return jsonify({'status': 'error', 'message': 'Card not found'})
        system_prompt = f"You are a Japanese kanji tutor. The student is studying 「{card.kanji}」 (meanings: {card.meanings_en}, on: {card.on_readings}, kun: {card.kun_readings}). Be concise (3-4 sentences)."
        conversation = [f"{'Student' if r == 'user' else 'Tutor'}: {m}" for r, m in data.get('chat_history', [])[-6:]]
        prompt = f"Previous: {chr(10).join(conversation)}\nStudent: {data['message']}\nRespond helpfully."
        chat_model = ai_model
        if not chat_model:
            raise ValueError("AI model not configured")
        response = chat_model.prompt(prompt, system=system_prompt)
        return jsonify({'status': 'success', 'response': response.text().strip()})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})


# ----------------------------------------------------------------------
# Top Words Routes
# ----------------------------------------------------------------------

def _ensure_top_words_imported() -> None:
    """Auto-import words CSV on first visit if table is empty."""
    try:
        progress = db.get_top_word_progress()
        if progress['total'] == 0:
            csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'top_20000_words.csv')
            if os.path.exists(csv_path):
                db.Base.metadata.create_all(bind=db.engine)
                db.import_top_words_csv(csv_path)
    except Exception:
        try:
            db.Base.metadata.create_all(bind=db.engine)
            csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'top_20000_words.csv')
            if os.path.exists(csv_path):
                db.import_top_words_csv(csv_path)
        except Exception as e2:
            print(f"❌ Words import failed: {e2}")


@app.route('/words-study')
def words_study() -> Any:
    """Top Words study page."""
    _ensure_top_words_imported()
    return render_template('mc_study.html',
        title='Top Words', icon='📝', subtitle='Learn the most frequent Japanese words with multiple-choice quizzes',
        batch_url=url_for('api_words_batch'), answer_url=url_for('api_words_answer'),
        lesson_url=url_for('api_words_lesson'), chat_url=url_for('api_words_chat'),
        new_session_url=url_for('words_study'), display_field='lemma',
        batch_size=app.config.get('BATCH_SIZE', 5))


@app.route('/api/words/batch')
def api_words_batch() -> Any:
    """Batch generate word MC questions using parallel execution. Handles lazy annotation."""
    default_count = app.config.get('BATCH_SIZE', 5)
    count = min(request.args.get('count', default_count, type=int), 15)
    exclude_param = request.args.get('exclude_ids', '', type=str)
    client_exclude: set = set(int(x) for x in exclude_param.split(',') if x.strip()) if exclude_param else set()

    # 1. Fetch cards and handle lazy annotation
    cards_to_process = []
    seen_ids: set = set(client_exclude)
    pending_new_count = 0
    study_model = study_ai_model or ai_model

    if not study_model:
        return jsonify({'status': 'error', 'message': 'AI model not configured'})

    for _ in range(count):
        try:
            card = db.get_next_top_word(exclude_ids=seen_ids, pending_new_count=pending_new_count)
            if not card or card.id in seen_ids:
                break
            seen_ids.add(card.id)
            if card.success_count <= 0:
                pending_new_count += 1

            # Lazy LLM annotation
            if not card.annotated:
                annotations = exercises.annotate_word_with_llm(card, study_model)
                db.annotate_top_word(card.id, **annotations)
                card.reading = annotations['reading']
                card.meanings_en = annotations['meanings_en']
                card.pos_tags = annotations['pos_tags']
                card.jlpt_level = annotations['jlpt_level']
                card.annotated = True

            if not card.annotated or not card.meanings_en:
                continue

            distractors = db.get_word_distractors(card, count=3)
            cards_to_process.append((card, distractors))

        except Exception as e:
            if DEBUG:
                print(f"⚠️ Error fetching word card: {e}")
            continue

    if not cards_to_process:
        return jsonify({'status': 'no_cards', 'message': 'No word cards available', 'cards': []})

    # 2. Generate quizzes in parallel
    def generate_word_task(card_tuple):
        card, distractors = card_tuple
        try:
            quiz_questions = exercises.generate_word_quiz(card, distractors, model=study_model)
            return {
                'id': card.id, 'lemma': card.lemma, 'display_text': card.lemma,
                'reading': card.reading, 'meanings_en': card.meanings_en,
                'pos_tags': card.pos_tags, 'jlpt_level': card.jlpt_level,
                'rank': card.rank, 'is_new': card.success_count <= 0,
                'questions': quiz_questions,
            }
        except Exception as e:
            if DEBUG:
                print(f"⚠️ Word question error for {card.lemma}: {e}")
            return None

    cards_data = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        future_to_card = {executor.submit(generate_word_task, c): c for c in cards_to_process}
        for future in concurrent.futures.as_completed(future_to_card):
            result = future.result()
            if result:
                cards_data.append(result)

    if not cards_data:
        return jsonify({'status': 'error', 'message': 'Failed to generate questions', 'cards': []})

    return jsonify({'status': 'success', 'cards': cards_data})


@app.route('/api/words/answer', methods=['POST'])
def api_words_answer() -> Any:
    """Submit a word quiz result (aggregate of 3 questions)."""
    try:
        data = request.get_json()
        card_id = int(data['card_id'])

        # Manual recall override: user picks quality directly (Anki-style)
        if 'manual_quality' in data:
            quality = max(1, min(5, int(data['manual_quality'])))
            is_correct = quality >= 3
            feedback = f"Manual override: quality {quality}"

        elif 'correct_count' in data:
            correct_count = int(data['correct_count'])
            total_count = int(data.get('total_count', 3))

            # Calculate quality score (0-5) for SM-2 algorithm
            # 3/3 = 5 (Perfect), 2/3 = 4 (Good), 1/3 = 2 (Fail), 0/3 = 1 (Fail)
            if correct_count == total_count:
                quality = 5
            elif correct_count >= total_count - 1:
                quality = 4
            elif correct_count >= total_count * 0.5:
                quality = 3
            elif correct_count > 0:
                quality = 2
            else:
                quality = 1

            is_correct = quality >= 3
            feedback = f"You got {correct_count} out of {total_count} correct."
        else:
            # Legacy
            selected_key = data['selected_key']
            correct_key = data['correct_key']
            is_correct = selected_key == correct_key
            quality = 5 if is_correct else 1
            feedback = "Correct!" if is_correct else "Incorrect."

        db.review_top_word(card_id, quality, user=session.get('username', 'default_user'))
        db.update_progress(session.get('username', 'default_user'), is_correct)

        return jsonify({'status': 'success', 'is_correct': is_correct, 'score': quality, 'feedback': feedback})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})


@app.route('/api/words/lesson', methods=['POST'])
def api_words_lesson() -> Any:
    """Generate a word lesson."""
    try:
        data = request.get_json()
        card_id = int(data['card_id'])
        session_db = db.get_session()
        card = session_db.get(db.TopWord, card_id)
        session_db.close()
        if not card:
            return jsonify({'status': 'error', 'message': 'Card not found'})
        db.mark_top_word_lesson_viewed(card_id)
        study_model = study_ai_model or ai_model
        if not study_model:
            raise ValueError("AI model not configured")
        lesson = exercises.generate_word_lesson(card, model=study_model)
        return jsonify({'status': 'success', 'lesson': lesson})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})


@app.route('/api/words/chat', methods=['POST'])
def api_words_chat() -> Any:
    """Chat about a word."""
    try:
        data = request.get_json()
        card_id = int(data['card_id'])
        session_db = db.get_session()
        card = session_db.get(db.TopWord, card_id)
        session_db.close()
        if not card:
            return jsonify({'status': 'error', 'message': 'Card not found'})
        system_prompt = f"You are a Japanese vocabulary tutor. The student is studying 「{card.lemma}」 (reading: {card.reading}, meanings: {card.meanings_en}). Be concise (3-4 sentences)."
        conversation = [f"{'Student' if r == 'user' else 'Tutor'}: {m}" for r, m in data.get('chat_history', [])[-6:]]
        prompt = f"Previous: {chr(10).join(conversation)}\nStudent: {data['message']}\nRespond helpfully."
        chat_model = ai_model
        if not chat_model:
            raise ValueError("AI model not configured")
        response = chat_model.prompt(prompt, system=system_prompt)
        return jsonify({'status': 'success', 'response': response.text().strip()})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})


def get_local_ip():
    """Attempt to determine the local network IP address."""
    import socket
    try:
        # Connect to an external server (doesn't actually send data)
        # to determine the interface used for internet access
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        local_ip = s.getsockname()[0]
        s.close()
        return local_ip
    except Exception:
        return "127.0.0.1"

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Japanese Learning App')
    parser.add_argument('--host', help='Host IP to bind to (default: auto-detect local IP)')
    parser.add_argument('--port', type=int, default=5000, help='Port to bind to (default: 5000)')
    parser.add_argument('--openai-key', help='OpenAI API Key')
    parser.add_argument('--openrouter-key', help='OpenRouter API Key (overrides OpenAI key)')
    parser.add_argument('--model', default='gpt-5.2', help='Main AI model name')
    parser.add_argument('--study-model', help='Study AI model name (defaults to main model)')
    parser.add_argument('--batch-size', type=int, default=5, help='Number of cards to prefetch in batch (default: 5)')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')

    args = parser.parse_args()

    if args.debug:
        DEBUG = True

    app.config['BATCH_SIZE'] = args.batch_size
    print(f"⚙️  Batch size configured to: {app.config['BATCH_SIZE']}")

    # Re-initialize AI if arguments are provided
    if args.openai_key or args.openrouter_key or args.model or args.study_model:
        api_key = args.openrouter_key or args.openai_key
        base_url = "https://openrouter.ai/api/v1" if args.openrouter_key else None

        init_ai(
            api_key=api_key,
            base_url=base_url,
            model_name=args.model,
            study_model_name=args.study_model
        )

    # Initialize database
    try:
        if not db.is_db_initialized():
            db.init_db()
            print("✅ Database initialized")
    except Exception as e:
        print(f"❌ Database initialization failed: {str(e)}")
    
    host = args.host or get_local_ip()
    print(f"🚀 Starting server on http://{host}:{args.port}")
    app.run(debug=DEBUG, host=host, port=args.port)
