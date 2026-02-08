"""
Tests for the advanced analytics feature (get_analytics_data and /api/analytics).
"""

import os
import tempfile
import datetime
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from typing import Any, Generator

from llm_learn_japanese import db


@pytest.fixture(scope="function")
def temp_db() -> Generator[None, None, None]:
    """Setup transient SQLite DB for testing."""
    fd, path = tempfile.mkstemp()
    os.close(fd)
    db.engine = create_engine(f"sqlite:///{path}")
    db.SessionLocal = sessionmaker(bind=db.engine)
    db.Base.metadata.create_all(bind=db.engine)
    yield
    os.unlink(path)


def _seed_card_aspects(session: Any) -> None:
    """Insert a mix of CardAspects for testing analytics."""
    now = datetime.datetime.now(datetime.timezone.utc)
    aspects = [
        # Vocabulary: 2 items, one studied (mature), one new
        db.CardAspect(parent_type="vocabulary", parent_id=1, aspect_type="meaning",
                      prompt_template="meaning?", success_count=5, interval=45,
                      ease_factor=2.6, repetitions=5,
                      next_review=now + datetime.timedelta(days=45)),
        db.CardAspect(parent_type="vocabulary", parent_id=1, aspect_type="reading",
                      prompt_template="reading?", success_count=3, interval=10,
                      ease_factor=2.3, repetitions=3,
                      next_review=now + datetime.timedelta(days=2)),
        db.CardAspect(parent_type="vocabulary", parent_id=2, aspect_type="meaning",
                      prompt_template="meaning?", success_count=0, interval=1,
                      ease_factor=250, repetitions=0,
                      next_review=now),
        # Kanji: 1 item, studied (learning stage)
        db.CardAspect(parent_type="kanji", parent_id=1, aspect_type="meaning",
                      prompt_template="meaning?", success_count=2, interval=3,
                      ease_factor=2.1, repetitions=2,
                      next_review=now - datetime.timedelta(days=1)),
        db.CardAspect(parent_type="kanji", parent_id=1, aspect_type="onyomi",
                      prompt_template="onyomi?", success_count=1, interval=1,
                      ease_factor=1.5, repetitions=1,
                      next_review=now),
        # Grammar: 1 item, young stage
        db.CardAspect(parent_type="grammar", parent_id=1, aspect_type="explanation",
                      prompt_template="explain?", success_count=4, interval=15,
                      ease_factor=2.4, repetitions=4,
                      next_review=now + datetime.timedelta(days=7)),
    ]
    session.add_all(aspects)
    session.commit()


def _seed_daily_progress(session: Any, user: str) -> None:
    """Insert daily progress rows to test streak and heatmap."""
    today = datetime.date.today()
    rows = [
        db.DailyProgress(user=user, date=today, cards_reviewed=10, new_cards_seen=2),
        db.DailyProgress(user=user, date=today - datetime.timedelta(days=1),
                         cards_reviewed=8, new_cards_seen=1),
        db.DailyProgress(user=user, date=today - datetime.timedelta(days=2),
                         cards_reviewed=5, new_cards_seen=3),
        # Gap on day 3
        db.DailyProgress(user=user, date=today - datetime.timedelta(days=4),
                         cards_reviewed=12, new_cards_seen=0),
        db.DailyProgress(user=user, date=today - datetime.timedelta(days=5),
                         cards_reviewed=7, new_cards_seen=1),
    ]
    session.add_all(rows)
    session.commit()


def _seed_vocabulary(session: Any) -> None:
    """Insert parent vocabulary rows for label lookup."""
    session.add(db.Vocabulary(id=1, word="勉強"))
    session.add(db.Vocabulary(id=2, word="食べる"))
    session.commit()


def _seed_kanji(session: Any) -> None:
    """Insert parent kanji rows for label lookup."""
    session.add(db.Kanji(id=1, character="日"))
    session.commit()


def _seed_grammar(session: Any) -> None:
    """Insert parent grammar rows for label lookup."""
    session.add(db.Grammar(id=1, point="〜てから"))
    session.commit()


# ── Unit tests for get_analytics_data ────────────────────────────────

class TestGetAnalyticsData:
    """Test the get_analytics_data function."""

    def test_empty_db_returns_all_keys(self, temp_db: Any) -> None:
        """Analytics should return all expected keys even on an empty DB."""
        result = db.get_analytics_data("test_user")
        assert "streak" in result
        assert "heatmap" in result
        assert "content_breakdown" in result
        assert "srs_stages" in result
        assert "srs_by_type" in result
        assert "forecast" in result
        assert "weakest" in result
        assert "strongest" in result
        assert "module_progress" in result
        assert "averages" in result

    def test_empty_db_streak_zero(self, temp_db: Any) -> None:
        """Streak should be 0 on empty DB."""
        result = db.get_analytics_data("test_user")
        assert result["streak"]["current"] == 0
        assert result["streak"]["longest"] == 0

    def test_empty_db_srs_stages_all_zero(self, temp_db: Any) -> None:
        """SRS stages should all be 0 on empty DB."""
        result = db.get_analytics_data("test_user")
        for stage in ("new", "learning", "young", "mature"):
            assert result["srs_stages"][stage] == 0

    def test_empty_db_averages_zero(self, temp_db: Any) -> None:
        """Averages should be 0 on empty DB."""
        result = db.get_analytics_data("test_user")
        assert result["averages"]["avg_ease"] == 0
        assert result["averages"]["avg_interval"] == 0
        assert result["averages"]["mature_count"] == 0

    def test_heatmap_length_90(self, temp_db: Any) -> None:
        """Heatmap should always have 90 entries."""
        result = db.get_analytics_data("test_user")
        assert len(result["heatmap"]) == 90

    def test_forecast_length_15(self, temp_db: Any) -> None:
        """Forecast should have 15 entries (today + 14 days)."""
        result = db.get_analytics_data("test_user")
        assert len(result["forecast"]) == 15

    def test_streak_calculation(self, temp_db: Any) -> None:
        """Streak should count consecutive study days ending today."""
        session = db.get_session()
        _seed_daily_progress(session, "streak_user")
        session.close()

        result = db.get_analytics_data("streak_user")
        # Today + yesterday + day before = 3-day streak (gap on day 3)
        assert result["streak"]["current"] == 3
        # Longest: days 0,1,2 = 3 or days 4,5 = 2 → longest = 3
        assert result["streak"]["longest"] == 3

    def test_content_breakdown(self, temp_db: Any) -> None:
        """Content breakdown should show correct total and studied counts."""
        session = db.get_session()
        _seed_card_aspects(session)
        session.close()

        result = db.get_analytics_data("test_user")
        breakdown = {r["type"]: r for r in result["content_breakdown"]}

        # vocabulary: 2 parent_ids total, 1 studied (id=1 has success_count > 0)
        assert breakdown["vocabulary"]["total"] == 2
        assert breakdown["vocabulary"]["studied"] == 1

        # kanji: 1 parent_id, 1 studied
        assert breakdown["kanji"]["total"] == 1
        assert breakdown["kanji"]["studied"] == 1

        # grammar: 1 parent_id, 1 studied
        assert breakdown["grammar"]["total"] == 1
        assert breakdown["grammar"]["studied"] == 1

    def test_srs_stage_distribution(self, temp_db: Any) -> None:
        """SRS stages should bucket cards correctly."""
        session = db.get_session()
        _seed_card_aspects(session)
        session.close()

        result = db.get_analytics_data("test_user")
        stages = result["srs_stages"]

        # new: vocab#2 meaning (success_count=0) → 1
        assert stages["new"] == 1
        # learning: kanji meaning (interval=3), kanji onyomi (interval=1) → 2
        assert stages["learning"] == 2
        # young: vocab reading (interval=10), grammar explanation (interval=15) → 2
        assert stages["young"] == 2
        # mature: vocab meaning (interval=45) → 1
        assert stages["mature"] == 1

    def test_srs_by_type(self, temp_db: Any) -> None:
        """SRS by type should break down stages per content type."""
        session = db.get_session()
        _seed_card_aspects(session)
        session.close()

        result = db.get_analytics_data("test_user")
        by_type = {r["type"]: r for r in result["srs_by_type"]}

        assert by_type["vocabulary"]["new"] == 1
        assert by_type["vocabulary"]["young"] == 1
        assert by_type["vocabulary"]["mature"] == 1
        assert by_type["kanji"]["learning"] == 2

    def test_forecast_counts_due_cards(self, temp_db: Any) -> None:
        """Forecast should count cards due in the next 14 days."""
        session = db.get_session()
        _seed_card_aspects(session)
        session.close()

        result = db.get_analytics_data("test_user")
        forecast = result["forecast"]
        total_forecast = sum(f["count"] for f in forecast)

        # Cards with success_count > 0 and next_review in range:
        # kanji meaning: due yesterday → counts as today
        # kanji onyomi: due now → today
        # vocab reading: due in 2 days
        # grammar explanation: due in 7 days
        # vocab meaning: due in 45 days → not in forecast
        assert total_forecast >= 3  # at least kanji meaning, kanji onyomi, vocab reading

    def test_weakest_cards(self, temp_db: Any) -> None:
        """Weakest cards should return those with lowest ease_factor."""
        session = db.get_session()
        _seed_card_aspects(session)
        _seed_kanji(session)
        session.close()

        result = db.get_analytics_data("test_user")
        weakest = result["weakest"]

        assert len(weakest) > 0
        # The weakest card should be kanji onyomi with ease_factor=1.5
        assert weakest[0]["ease_factor"] == 1.5
        assert weakest[0]["label"] == "日"

    def test_strongest_cards(self, temp_db: Any) -> None:
        """Strongest cards should return those with highest interval."""
        session = db.get_session()
        _seed_card_aspects(session)
        _seed_vocabulary(session)
        session.close()

        result = db.get_analytics_data("test_user")
        strongest = result["strongest"]

        assert len(strongest) > 0
        # The strongest card should be vocab meaning with interval=45
        assert strongest[0]["interval"] == 45
        assert strongest[0]["label"] == "勉強"

    def test_averages(self, temp_db: Any) -> None:
        """Averages should compute correctly from studied cards."""
        session = db.get_session()
        _seed_card_aspects(session)
        session.close()

        result = db.get_analytics_data("test_user")
        avgs = result["averages"]

        # Only studied cards (success_count > 0): 5 cards
        # ease factors: 2.6, 2.3, 2.1, 1.5, 2.4 → avg ≈ 2.18
        assert 2.0 <= avgs["avg_ease"] <= 2.3
        # intervals: 45, 10, 3, 1, 15 → avg ≈ 14.8
        assert 14.0 <= avgs["avg_interval"] <= 15.0
        # mature (interval > 30): 1 card
        assert avgs["mature_count"] == 1

    def test_heatmap_includes_activity(self, temp_db: Any) -> None:
        """Heatmap should reflect daily progress data."""
        session = db.get_session()
        _seed_daily_progress(session, "heatmap_user")
        session.close()

        result = db.get_analytics_data("heatmap_user")
        heatmap = result["heatmap"]

        # The last entry should be today with 10 reviews
        today_iso = datetime.date.today().isoformat()
        today_entry = next((h for h in heatmap if h["date"] == today_iso), None)
        assert today_entry is not None
        assert today_entry["count"] == 10

    def test_module_progress_empty(self, temp_db: Any) -> None:
        """Module progress should show 0/0 when tables are empty."""
        result = db.get_analytics_data("test_user")
        mp = result["module_progress"]
        for key in ("bunpro", "kanji", "words"):
            assert mp[key]["total"] == 0
            assert mp[key]["studied"] == 0

    def test_card_label_lookup(self, temp_db: Any) -> None:
        """Card labels should resolve to parent item names."""
        session = db.get_session()
        _seed_card_aspects(session)
        _seed_vocabulary(session)
        _seed_kanji(session)
        _seed_grammar(session)
        session.close()

        result = db.get_analytics_data("test_user")
        labels = {c["label"] for c in result["weakest"] + result["strongest"]}
        # Should contain the seeded names
        assert "勉強" in labels or "日" in labels or "〜てから" in labels


# ── Integration test for /api/analytics endpoint ─────────────────────

class TestAnalyticsEndpoint:
    """Test the /api/analytics Flask endpoint."""

    @pytest.fixture
    def client(self, temp_db: Any) -> Any:
        """Create a Flask test client."""
        os.environ["TEST_MODE"] = "1"
        # Import app after setting TEST_MODE
        import app as flask_app
        flask_app.app.config["TESTING"] = True
        flask_app.app.config["SECRET_KEY"] = "test-secret"
        with flask_app.app.test_client() as c:
            with flask_app.app.app_context():
                yield c

    def test_api_analytics_returns_200(self, client: Any) -> None:
        """The /api/analytics endpoint should return 200 with JSON."""
        resp = client.get("/api/analytics")
        assert resp.status_code == 200
        data = resp.get_json()
        assert "streak" in data
        assert "heatmap" in data
        assert "srs_stages" in data

    def test_api_analytics_has_forecast(self, client: Any) -> None:
        """The /api/analytics endpoint should include forecast data."""
        resp = client.get("/api/analytics")
        data = resp.get_json()
        assert "forecast" in data
        assert len(data["forecast"]) == 15

    def test_progress_page_loads(self, client: Any) -> None:
        """The /progress page should load with analytics data."""
        resp = client.get("/progress")
        assert resp.status_code == 200
        # Check that the page contains analytics tab markup
        assert b"analyticsTabs" in resp.data
        assert b"Activity Heatmap" in resp.data
