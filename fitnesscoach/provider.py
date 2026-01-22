import datetime
import json
import os
from typing import Any

import keyring
import requests
import typer
from garminconnect import (
    Garmin,
    GarminConnectAuthenticationError,
    GarminConnectConnectionError,
    GarthException,
    GarthHTTPError,
)
from rich import print


SERVICE_NAME = "fitness-coach"
CONFIG_DIR = os.path.expanduser(f"~/.{SERVICE_NAME}")
CACHED_DATA = os.path.join(CONFIG_DIR, "cached_data.json")
CACHE_EXPIRATION_MINUTES = 60


class Config:
    def __init__(self):
        self.config_file = os.path.join(CONFIG_DIR, "config.json")
        if not os.path.exists(self.config_file):
            os.makedirs(os.path.dirname(self.config_file), exist_ok=True)
            with open(self.config_file, "w") as f:
                json.dump({}, f)
        with open(self.config_file, "r") as f:
            self.config = json.load(f)

    def get(self, key: str):
        return self.config.get(key, None)

    def set(self, key: str, value):
        self.config[key] = value
        with open(self.config_file, "w") as f:
            json.dump(self.config, f)

    def delete(self, key: str):
        del self.config[key]
        with open(self.config_file, "w") as f:
            json.dump(self.config, f)


# Initialize config
config = Config()


def init_api(email: str | None = None, password: str | None = None) -> Garmin | None:
    """Initialize Garmin API with smart error handling and recovery."""
    # First try to login with stored tokens
    try:
        print(f"Attempting to login using stored tokens from: {CONFIG_DIR}")

        garmin = Garmin()
        garmin.login(CONFIG_DIR)
        print("Successfully logged in using stored tokens")
        return garmin

    except (
        FileNotFoundError,
        GarthHTTPError,
        GarminConnectAuthenticationError,
        GarminConnectConnectionError,
    ):
        print("No valid tokens found. Requesting fresh login credentials.")

    # Loop for credential entry with retry on auth failure
    while True:
        try:
            # Get credentials if not provided
            if not email or not password:
                return None

            print("Logging in with credentials...")
            garmin = Garmin(email=email, password=password, is_cn=False, return_on_mfa=True)
            access_token, refresh_token = garmin.login()

            if access_token == "needs_mfa":
                print("Multi-factor authentication required")

                mfa_code = typer.prompt("MFA one-time code: ").strip()
                print("🔄 Submitting MFA code...")

                try:
                    garmin.resume_login(refresh_token, mfa_code)
                    print("✅ MFA authentication successful!")

                except GarthHTTPError as garth_error:
                    # Handle specific HTTP errors from MFA
                    error_str = str(garth_error)
                    print(f"🔍 Debug: MFA error details: {error_str}")

                    if "429" in error_str and "Too Many Requests" in error_str:
                        print("❌ Too many MFA attempts")
                        print("💡 Please wait 30 minutes before trying again")
                        return None
                    elif "401" in error_str or "403" in error_str:
                        print("❌ Invalid MFA code")
                        print("💡 Please verify your MFA code and try again")
                        continue
                    else:
                        # Other HTTP errors - don't retry
                        print(f"❌ MFA authentication failed: {garth_error}")
                        return None

                except GarthException as garth_error:
                    print(f"❌ MFA authentication failed: {garth_error}")
                    print("💡 Please verify your MFA code and try again")
                    continue

            # Save tokens for future use
            garmin.garth.dump(CONFIG_DIR)
            print(f"Login successful! Tokens saved to: {CONFIG_DIR}")

            return garmin

        except GarminConnectAuthenticationError:
            print("❌ Authentication failed:")
            print("💡 Please check your username and password and try again")
            # Clear the provided credentials to force re-entry
            email = None
            password = None
            continue

        except (
            FileNotFoundError,
            GarthHTTPError,
            GarthException,
            GarminConnectConnectionError,
            requests.exceptions.HTTPError,
        ) as err:
            print(f"❌ Connection error: {err}")
            print("💡 Please check your internet connection and try again")
            return None

        except KeyboardInterrupt:
            print("\nLogin cancelled by user")
            return None


def login() -> Garmin:
    username = config.get("username")
    if username is None:
        username = typer.prompt("Enter your Garmin username")

    # Try to get password from: 1) env var, 2) keyring, 3) prompt
    password = os.environ.get("SERVICE_PASSWORD")
    if password is None:
        try:
            password = keyring.get_password(SERVICE_NAME, username)
        except Exception as e:
            print(f"[yellow]Warning: Keyring unavailable ({e})[/yellow]")
            password = None
    if password is None:
        password = typer.prompt("Enter your Garmin password", hide_input=True)

    garmin = init_api(username, password)
    if garmin is None:
        print("❌ Login failed")
        raise typer.Exit(code=1)

    config.set("username", username)
    try:
        keyring.set_password(SERVICE_NAME, username, password)
    except Exception as e:
        print(f"[yellow]Warning: Could not save password to keyring ({e})[/yellow]")
        if not os.environ.get("SERVICE_PASSWORD"):
            print("[yellow]Tip: Set SERVICE_PASSWORD environment variable to avoid re-entering password[/yellow]")
    return garmin


def get_summary(force_refresh: bool = False) -> dict[str, Any]:
    """Get the summary from the Garmin API.

    Args:
        force_refresh: Whether to force a refresh of the summary (do not use cached data).

    Returns:
        dict: The summary data.
    """
    data = None
    if os.path.exists(CACHED_DATA) and not force_refresh:
        with open(CACHED_DATA, "r") as f:
            cached_data = json.load(f)
            timestamp_str = cached_data.get("timestamp", None)
            if timestamp_str:
                timestamp = datetime.datetime.fromisoformat(timestamp_str)
                # check if timestamp is within the last hour
                last_valid_timestamp = datetime.datetime.now() - datetime.timedelta(minutes=CACHE_EXPIRATION_MINUTES)
                if timestamp > last_valid_timestamp:
                    data = cached_data
    if data is None:
        garmin = login()
        data = garmin.get_user_summary(cdate=datetime.date.today().isoformat())
        data["timestamp"] = datetime.datetime.now().isoformat()
        with open(CACHED_DATA, "w") as f:
            json.dump(data, f)
    return data


def get_steps() -> int:
    """Get the steps from the summary.

    Args:
        None

    Returns:
        int: The number of steps taken today
    """
    summary = get_summary()
    return summary.get("totalSteps", 0)


def get_daily_step_goal() -> int:
    """Get the daily step goal from the summary.

    Args:
        None

    Returns:
        int: The daily step goal
    """
    summary = get_summary()
    return summary.get("dailyStepGoal", 0)


def get_goal_progress() -> float:
    """Get the goal progress of the day from the summary in percentage.

    Args:
        None

    Returns:
        float: The goal progress in percentage
    """
    steps = get_steps()
    daily_step_goal = get_daily_step_goal()
    if daily_step_goal == 0:
        return 0.0
    return (steps / daily_step_goal) * 100


def get_sleeping_minutes() -> int:
    """Get the sleeping minutes of the day from the summary.

    Args:
        None

    Returns:
        int: The number of sleeping minutes
    """
    summary = get_summary()
    sleeping_seconds = summary.get("sleepingSeconds", 0)
    return sleeping_seconds // 60


def get_active_minutes() -> int:
    """Get the active minutes of the day from the summary.

    Args:
        None

    Returns:
        int: The number of active minutes
    """
    summary = get_summary()
    active_seconds = summary.get("activeSeconds", 0)
    return active_seconds // 60


def get_heart_rate() -> tuple[int, int]:
    """Get the minimum and maximum heart rate of the day from the summary.

    Args:
        None

    Returns:
        tuple[int, int]: The minimum and maximum heart rate
    """
    summary = get_summary()
    min_heart_rate = summary.get("minHeartRate", 0)
    max_heart_rate = summary.get("maxHeartRate", 0)
    return min_heart_rate, max_heart_rate


def get_body_battery_level() -> int:
    """Get the body battery level of the day from the summary.

    Args:
        None

    Returns:
        int: The body battery level
    """
    summary = get_summary()
    most_recent_body_battery_level = summary.get("bodyBatteryMostRecentValue", 0)
    return most_recent_body_battery_level
