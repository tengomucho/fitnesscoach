import datetime
import json
from unittest.mock import MagicMock, mock_open, patch

import pytest

from fitness_coach import fitness_coach


@pytest.fixture
def mock_summary_data():
    return {
        "totalSteps": 10000,
        "dailyStepGoal": 12000,
        "sleepingSeconds": 28800, # 8 hours
        "activeSeconds": 3600,    # 1 hour
        "minHeartRate": 55,
        "maxHeartRate": 165,
        "bodyBatteryMostRecentValue": 85,
        "timestamp": "2026-01-07T10:00:00"
    }

def test_get_steps(mock_summary_data):
    with patch("fitness_coach.fitness_coach.get_summary", return_value=mock_summary_data):
        assert fitness_coach.get_steps() == 10000

def test_get_daily_step_goal(mock_summary_data):
    with patch("fitness_coach.fitness_coach.get_summary", return_value=mock_summary_data):
        assert fitness_coach.get_daily_step_goal() == 12000

def test_get_goal_progress(mock_summary_data):
    with patch("fitness_coach.fitness_coach.get_summary", return_value=mock_summary_data):
        # (10000 / 12000) * 100 = 83.333...
        assert fitness_coach.get_goal_progress() == pytest.approx(83.333, rel=1e-3)

def test_get_goal_progress_zero_goal():
    mock_data = {"totalSteps": 1000, "dailyStepGoal": 0}
    with patch("fitness_coach.fitness_coach.get_summary", return_value=mock_data):
        assert fitness_coach.get_goal_progress() == 0.0

def test_get_sleeping_minutes(mock_summary_data):
    with patch("fitness_coach.fitness_coach.get_summary", return_value=mock_summary_data):
        assert fitness_coach.get_sleeping_minutes() == 480 # 28800 // 60

def test_get_active_minutes(mock_summary_data):
    with patch("fitness_coach.fitness_coach.get_summary", return_value=mock_summary_data):
        assert fitness_coach.get_active_minutes() == 60 # 3600 // 60

def test_get_heart_rate(mock_summary_data):
    with patch("fitness_coach.fitness_coach.get_summary", return_value=mock_summary_data):
        assert fitness_coach.get_heart_rate() == (55, 165)

def test_get_body_battery_level(mock_summary_data):
    with patch("fitness_coach.fitness_coach.get_summary", return_value=mock_summary_data):
        assert fitness_coach.get_body_battery_level() == 85

@patch("fitness_coach.fitness_coach.os.path.exists")
@patch("fitness_coach.fitness_coach.open", new_callable=mock_open)
@patch("fitness_coach.fitness_coach.datetime")
def test_get_summary_cached(mock_datetime, mock_file, mock_exists):
    # Mock cache exists
    mock_exists.return_value = True

    # Mock current time and timedelta
    now = datetime.datetime(2026, 1, 7, 10, 30)
    mock_datetime.datetime.now.return_value = now
    mock_datetime.datetime.fromisoformat.side_effect = lambda x: datetime.datetime.fromisoformat(x)
    mock_datetime.timedelta = datetime.timedelta

    cached_data = {
        "totalSteps": 5000,
        "timestamp": "2026-01-07T10:15:00" # 15 mins ago, valid
    }

    mock_file.return_value.read.return_value = json.dumps(cached_data)

    # We need to patch json.load because it will be called with the mock_file
    with patch("json.load", return_value=cached_data):
        result = fitness_coach.get_summary()
        assert result["totalSteps"] == 5000
        assert result["timestamp"] == "2026-01-07T10:15:00"

@patch("fitness_coach.fitness_coach.login")
@patch("fitness_coach.fitness_coach.os.path.exists")
@patch("fitness_coach.fitness_coach.open", new_callable=mock_open)
@patch("fitness_coach.fitness_coach.datetime")
def test_get_summary_no_cache(mock_datetime, mock_file, mock_exists, mock_login):
    # Mock cache does not exist
    mock_exists.return_value = False

    # Mock login and garmin API
    mock_garmin = MagicMock()
    mock_login.return_value = mock_garmin
    mock_garmin.get_user_summary.return_value = {"totalSteps": 8000}

    # Mock dates
    today = datetime.date(2026, 1, 7)
    mock_datetime.date.today.return_value = today
    now = datetime.datetime(2026, 1, 7, 10, 30)
    mock_datetime.datetime.now.return_value = now

    result = fitness_coach.get_summary()

    assert result["totalSteps"] == 8000
    assert "timestamp" in result
    mock_garmin.get_user_summary.assert_called_once_with(cdate="2026-01-07")
