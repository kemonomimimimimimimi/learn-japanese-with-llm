# Japanese Learning App

A comprehensive **web-based application** for learning Japanese using **Spaced Repetition** (FSRS) with a modern Flask web interface and AI-powered exercises.

*(Note: the previous terminal-based curses interface in `main.py` has been deprecated and replaced with this web application in `app.py`.)*

This application stores study content in a SQLite database via SQLAlchemy, covering:
- **Vocabulary** with reading, meaning, and usage aspects
- **Kanji** with onyomi, kunyomi, meaning, and usage aspects
- **Grammar** with explanation and usage aspects
- **Phrases** with meaning and usage aspects
- **Idioms** with meaning and usage aspects
- **Progress tracking** with accuracy statistics

---

## Installing `uv`

This project uses **[uv](https://docs.astral.sh/uv/)** — an extremely fast Python package and project manager written in Rust. Install it for your operating system below.

### Windows

**Option A – Standalone installer (PowerShell):**

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

**Option B – winget:**

```powershell
winget install --id=astral-sh.uv -e
```

After installation, restart your terminal so the `uv` command is available on your `PATH`.

### macOS

**Option A – Standalone installer:**

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**Option B – Homebrew:**

```bash
brew install uv
```

### Linux

**Option A – Standalone installer:**

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**Option B – Package managers:**

```bash
# Arch Linux
pacman -S uv

# Alpine
apk add uv
```

After running the install script, follow the printed instructions to add `uv` to your shell's `PATH` (usually by sourcing `~/.local/bin` or restarting your terminal).

> **Verify the installation** on any platform by running:
> ```
> uv --version
> ```

---

## Setting Up the Project

### Prerequisites: Install Git (Windows)

On macOS and Linux, `git` is typically pre-installed. On **Windows**, you need to install it first:

1. Download the installer from **https://git-scm.com/download/win**
2. Run the installer — the defaults are fine for most users
3. When prompted for the default editor, pick your preference (e.g., VS Code)
4. When prompted for PATH, choose **"Git from the command line and also from 3rd-party software"**
5. Finish the installer and restart your terminal

Verify with:

```powershell
git --version
```

### 1. Clone the repository

```bash
git clone <repository-url>
cd learn-japanese-with-llm
```

### 2. Create a virtual environment and install dependencies

`uv` reads [`pyproject.toml`](pyproject.toml) automatically. The commands are the same on every OS; only the virtual-environment *activation* step differs.

```bash
# Create the virtual environment (uses the Python version from pyproject.toml: >=3.12)
uv venv

# Install the project and all its dependencies in editable mode
uv pip install -e .

# (Optional) Install development dependencies (pytest, mypy)
uv pip install -e ".[dev]"
```

### 3. Activate the virtual environment

#### Linux / macOS

```bash
source .venv/bin/activate
```

#### Windows (PowerShell)

```powershell
.venv\Scripts\Activate.ps1
```

#### Windows (cmd.exe)

```cmd
.venv\Scripts\activate.bat
```

> **Tip:** You can skip manual activation entirely by prefixing commands with `uv run`, e.g. `uv run python app.py`. `uv run` automatically uses the project's virtual environment.

### 4. Set your OpenAI API key

The app requires an OpenAI-compatible API key for AI-powered exercise generation and evaluation. Export it as an environment variable **before** starting the server:

#### Linux / macOS

```bash
export OPENAI_API_KEY="sk-..."
```

#### Windows (PowerShell)

```powershell
$env:OPENAI_API_KEY = "sk-..."
```

You can also pass it directly via the `--openai-key` flag (see below).

---

## Running the Application

```bash
python app.py
```

The server starts at **http://\<your-local-ip\>:5000/** by default (it auto-detects your local network IP so you can access it from other devices).

### Command-line options

| Flag | Default | Description |
|---|---|---|
| `--host` | auto-detect | Host IP to bind to |
| `--port` | `5000` | Port to bind to |
| `--openai-key` | `$OPENAI_API_KEY` | OpenAI API key |
| `--openrouter-key` | — | OpenRouter API key (uses OpenRouter base URL) |
| `--model` | `gpt-5.2` | Main AI model name |
| `--study-model` | same as `--model` | Separate model for study sessions |
| `--batch-size` | `5` | Number of cards to prefetch per batch |
| `--debug` | off | Enable Flask debug mode and verbose AI logging |

**Examples:**

```bash
# Run with OpenAI directly
python app.py --openai-key sk-... --model gpt-4o --debug

# Run with OpenRouter (recommended — supports many models)
python app.py --openrouter-key sk-or-v1-... --model google/gemini-3-flash-preview

# Run with OpenRouter using a separate study model
python app.py --openrouter-key sk-or-v1-... --model google/gemini-3-flash-preview --study-model google/gemini-3-flash-preview

# Using uv run (no venv activation needed)
uv run python app.py --openrouter-key sk-or-v1-... --model google/gemini-3-flash-preview

# Custom port
python app.py --openrouter-key sk-or-v1-... --model google/gemini-3-flash-preview --port 8080
```

### Accessing the app in your browser

When the server starts, you'll see output like this:

```
$ OPENAI_API_KEY=sk-proj-key python app.py --debug
✅ AI initialized with model: gpt-5.2
⚙️  Batch size configured to: 5
✅ AI initialized with model: gpt-5.2
🚀 Starting server on http://192.168.1.123:5000
 * Serving Flask app 'app'
 * Debug mode: on
 * Running on http://192.168.1.123:5000
Press CTRL+C to quit
```

Open the URL shown in the `🚀 Starting server on …` line in your web browser:

- **Same machine:** Navigate to **http://127.0.0.1:5000** (or the IP shown)
- **Other devices on your network** (phone, tablet, etc.): Use the IP address printed in the output (e.g., `http://192.168.1.123:5000`). Make sure both devices are on the same Wi-Fi / LAN.

Press **Ctrl+C** in the terminal to stop the server.

---

## Application Overview

[`app.py`](app.py) is a **Flask web application** that serves as the main entry point. It provides a server-rendered UI with Jinja2 templates and a set of JSON APIs consumed by the frontend JavaScript.

### Web Pages & Routes

| Route | Purpose |
|---|---|
| `/` | **Dashboard** – user stats, cards due, navigation hub |
| `/study` | **Study session** – AI-generated Q&A review of due cards |
| `/add` | **Add content hub** – links to all content-type forms |
| `/add/vocabulary` | Add vocabulary (meaning, reading, usage aspects) |
| `/add/kanji` | Add kanji (meaning, onyomi, kunyomi, usage aspects) |
| `/add/grammar` | Add grammar points (explanation, usage aspects) |
| `/add/phrase` | Add phrases (meaning, usage aspects) |
| `/add/idiom` | Add idioms (meaning, usage aspects) |
| `/conjugations` | Browse verb conjugation patterns |
| `/study/conjugation/<id>` | Practice a specific conjugation |
| `/bunpro` | **Bunpro grammar study** – SRS-based grammar drills |
| `/kanji-study` | **Top kanji study** – batch kanji review sessions |
| `/words-study` | **Top words study** – batch vocabulary review sessions |
| `/progress` | **Progress dashboard** – accuracy stats & history |
| `/settings` | User settings (username, daily new-card limit) |
| `/process_image` | Extract Japanese content from images (AI-powered) |
| `/process_discord` | Parse Discord chat logs for learning material |
| `/process_renpy` | Extract Japanese dialogue from Ren'Py game scripts |
| `/process_raw_text` | Process raw Japanese text into study cards |
| `/manual_review` | Manually review and score cards |
| `/save_message` | Save notes and study logs |
| `/ai_status` | Check AI connectivity and model info |

### JSON APIs

The frontend communicates with the server through several API endpoints:

- **`/api/next_card`** – Fetch the next due card aspect for review
- **`/api/submit_answer`** – Submit an answer for AI evaluation and FSRS rescheduling
- **`/api/chat`** – Free-form AI chat during study sessions
- **`/api/bunpro/*`** – Bunpro grammar batch fetch, answer submission, lessons, chat, progress
- **`/api/kanji/*`** – Kanji batch fetch, answer submission, lessons, chat
- **`/api/words/*`** – Words batch fetch, answer submission, lessons, chat
- **`/api/analytics`** – Advanced analytics data (streak, SRS stages, forecast, heatmap, etc.)

### Core Architecture

```
app.py                          # Flask web application (main entry point)
main.py                         # (Deprecated) old curses terminal interface
llm_learn_japanese/
  ├── db.py                     # SQLAlchemy models, DB operations, AI answer evaluation
  ├── scheduler.py              # FSRS spaced repetition algorithm
  ├── exercises.py              # AI-powered exercise generation
  ├── structured.py             # Data structures and prompt templates
  └── plugin.py                 # Plugin utilities
templates/                      # Jinja2 HTML templates
static/css/                     # Stylesheets
migrations/                     # Alembic database migrations
tests/                          # pytest test suite
data/                           # CSV data files for bulk import
plans/                          # Feature planning documents
```

### AI Integration

The app uses the **OpenAI API** (or any OpenAI-compatible provider like OpenRouter) for:

1. **Dynamic exercise generation** – Creates varied, difficulty-adaptive questions for each card aspect (40+ exercise types across all content categories). See [`AI_FEATURES.md`](AI_FEATURES.md) for the full list.
2. **Intelligent answer evaluation** – Semantic scoring (0–5) with synonym recognition, partial credit, and constructive feedback. See [`AI_INTEGRATION_GUIDE.md`](AI_INTEGRATION_GUIDE.md) for architecture details.
3. **Content extraction** – Processes images, Discord logs, Ren'Py scripts, and raw text into structured study cards.
4. **In-session chat** – Contextual AI chat for asking follow-up questions during study.

All AI features **degrade gracefully** — the app falls back to template-based exercises and rule-based evaluation when no API key is configured.

### Spaced Repetition (FSRS)

Each content item is broken into **atomic aspects** (e.g., a kanji card has separate aspects for meaning, onyomi, kunyomi, and usage). The FSRS algorithm schedules each aspect independently based on review quality:

- **Quality 0–2**: Failed recall → short review interval
- **Quality 3**: Remembered with effort → moderate interval increase
- **Quality 4–5**: Easy recall → significant interval increase

---

## Development

### Running Tests

```bash
uv pip install -e ".[dev]"
uv run pytest tests/ -v
```

### Type Checking

```bash
uv run mypy .
```

### Database Migrations

The project uses **Alembic** for schema migrations:

```bash
uv run alembic upgrade head
```

---

## Additional Documentation

- [`AI_FEATURES.md`](AI_FEATURES.md) – Detailed documentation of all AI-powered features, exercise types, and evaluation system
- [`AI_INTEGRATION_GUIDE.md`](AI_INTEGRATION_GUIDE.md) – Technical guide for the AI architecture and integration points
- [`VERB_CONJUGATION_STUDY_GUIDE.md`](VERB_CONJUGATION_STUDY_GUIDE.md) – Comprehensive Japanese verb conjugation checklist

---

## Roadmap

- [x] Core spaced repetition with FSRS
- [x] Web-based interface using Flask
- [x] Support for Vocabulary, Kanji, Grammar, Idioms, Phrases
- [x] Atomic aspects system with independent scheduling
- [x] Duplicate detection
- [x] Progress tracking with statistics
- [x] AI-powered exercise generation and answer evaluation
- [x] Bunpro grammar study integration
- [x] Top kanji and words study modules
- [x] Verb conjugation study
- [x] Image, Discord, Ren'Py, and raw text content extraction
- [ ] Audio content processing
- [ ] Batch import/export functionality
- [x] Advanced analytics and learning insights

## Contributing

Contributions welcome! Please ensure:
- All tests pass (`uv run pytest`)
- Type checking passes (`uv run mypy .`)
- New features include tests
- Documentation is updated

---

## License

MIT License
