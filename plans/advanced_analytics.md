# Advanced Analytics & Learning Insights — Feature Plan

## Overview

Enhance the existing `/progress` page with rich, data-driven analytics derived entirely from the current database schema (no migrations needed). The page will be reorganised into tabbed sections with multiple Chart.js visualisations and computed insight cards.

---

## New Analytics Sections

### 1. Study Streak & Activity Heatmap

**Data source:** `DailyProgress` table

- **Current streak**: Consecutive days (ending today/yesterday) with `cards_reviewed > 0`
- **Longest streak**: All-time longest consecutive run
- **Activity heatmap**: A GitHub-style contribution grid showing the last 90 days of study activity, colour-coded by intensity (0 = grey, low = light green, high = dark green)

### 2. Content Inventory Breakdown

**Data source:** `CardAspect` grouped by `parent_type`

- **Bar chart** showing total items vs studied items per content type (vocabulary, kanji, grammar, phrase, idiom)
- **Summary stat cards**: total content items, total studied, total unstudied, % coverage

### 3. SRS Stage Distribution

**Data source:** `CardAspect.interval` and `CardAspect.success_count`

Bucket every card aspect into maturity stages:
| Stage | Criteria |
|---|---|
| New | `success_count == 0` |
| Learning | `success_count > 0` AND `interval <= 6` |
| Young | `interval` 7–30 |
| Mature | `interval > 30` |

- **Donut chart** showing distribution across stages
- Also broken down by `parent_type` in a stacked bar chart

### 4. Review Forecast

**Data source:** `CardAspect.next_review` for the next 14 days

- **Bar chart** showing how many cards will be due each day for the next 14 days
- Helps users plan study time

### 5. Weakest & Strongest Cards

**Data source:** `CardAspect.ease_factor`, joined to parent content tables

- **Weakest cards**: 10 cards with the lowest `ease_factor` (where `success_count > 0`) — these are the items the user struggles with most
- **Strongest cards**: 10 cards with the highest `interval` — well-mastered items
- Displayed as tables with the item name, type, ease factor, interval, and next review date

### 6. Bunpro / Top Kanji / Top Words Progress

**Data source:** `BunproGrammar`, `TopKanji`, `TopWord` tables

- **3 mini progress bars**: total items vs studied (success_count > 0) for each module
- Only displayed if the respective table has data

### 7. Accuracy Trend (Enhanced Daily Chart)

**Data source:** `DailyProgress` (existing chart, enhanced)

- Keep the existing 30-day line chart but add a **cumulative accuracy** line derived from `Progress.correct_answers / Progress.total_reviews` snapshots — since we only have the global total, we'll compute a **rolling average** from daily data or simply overlay the global accuracy as a reference line

---

## Architecture

```mermaid
flowchart TD
    A[User visits /progress] --> B[view_progress route in app.py]
    B --> C[get_analytics_data in db.py]
    C --> D[Query CardAspect - stage distribution]
    C --> E[Query CardAspect - content breakdown]
    C --> F[Query CardAspect - review forecast]
    C --> G[Query CardAspect - weakest/strongest]
    C --> H[Query DailyProgress - streak + heatmap]
    C --> I[Query Bunpro/Kanji/Words - module progress]
    B --> J[Render progress.html with analytics context]
    J --> K[Chart.js renders charts client-side]
    L[/api/analytics endpoint] --> C
    K -.-> L
```

### Data Flow

- **Server-side**: [`view_progress()`](app.py:466) calls a new [`get_analytics_data()`](llm_learn_japanese/db.py) function that runs all queries in a single DB session and returns a dict
- **Template**: [`progress.html`](templates/progress.html) receives the analytics dict and renders the static portions (stat cards, tables) server-side
- **Client-side**: Chart.js consumes JSON data embedded in the template (or fetched from `/api/analytics`) to render the charts
- **API**: A new `/api/analytics` endpoint returns the full analytics JSON for potential future use (e.g., mobile client)

---

## Files to Modify

| File | Changes |
|---|---|
| [`llm_learn_japanese/db.py`](llm_learn_japanese/db.py) | Add `get_analytics_data(user)` function with all analytics queries |
| [`app.py`](app.py) | Add `/api/analytics` route; update `view_progress()` to pass analytics data |
| [`templates/progress.html`](templates/progress.html) | Complete rewrite with tabbed layout, new chart canvases, stat cards, tables |
| [`static/css/style.css`](static/css/style.css) | Add heatmap grid CSS, analytics stat card styles, chart container spacing |
| `tests/test_analytics.py` | New file — test `get_analytics_data()` and `/api/analytics` endpoint |
| [`README.md`](README.md) | Update roadmap checkbox |

---

## Template Layout Plan

The new progress page will use a **tabbed layout** (Bootstrap nav-tabs):

**Tab 1 — Overview** (default)
- Streak card (current streak 🔥, longest streak)
- Existing 3 stat cards (total reviews, correct, accuracy)
- Today's progress card
- Content inventory bar chart
- SRS stage donut chart

**Tab 2 — Charts & Trends**
- Daily activity chart (enhanced existing, 30 days)
- Review forecast bar chart (next 14 days)
- SRS stages by content type stacked bar chart

**Tab 3 — Activity Heatmap**
- 90-day GitHub-style contribution heatmap

**Tab 4 — Card Details**
- Weakest cards table (10 items)
- Strongest cards table (10 items)
- Bunpro / Top Kanji / Top Words progress bars

---

## Key Design Decisions

1. **No database migration** — all analytics derived from existing `CardAspect`, `DailyProgress`, and module tables
2. **Single DB session** — `get_analytics_data()` opens one session and runs all queries, avoiding N+1 problems
3. **Chart.js only** — already loaded in the current template; no new JS dependencies
4. **Server-rendered with client-side charts** — stat cards and tables rendered in Jinja2, charts rendered via Chart.js from embedded JSON
5. **Graceful degradation** — all sections handle empty data with friendly "no data yet" messages
6. **Heatmap via pure CSS Grid** — no extra library, just a grid of coloured `<div>` cells
