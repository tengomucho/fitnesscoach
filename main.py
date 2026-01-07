import typer
import keyring
import json
import os
import requests
import datetime

from garminconnect import Garmin, GarthHTTPError, GarthException, GarminConnectAuthenticationError, GarminConnectConnectionError

from rich import print


app = typer.Typer()
SERVICE_NAME = "fitness-coach"
CONFIG_DIR = os.path.expanduser(f"~/.{SERVICE_NAME}")

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
            garmin = Garmin(
                email=email, password=password, is_cn=False, return_on_mfa=True
            )
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


@app.command()
def login() -> Garmin:
    username = config.get("username")
    if username is None:
        username = typer.prompt("Enter your Garmin username")
    password = keyring.get_password(SERVICE_NAME, username)
    if password is None:
        password = typer.prompt("Enter your Garmin password", hide_input=True)

    garmin = init_api(username, password)
    if garmin is None:
        typer.echo("❌ Login failed")
        raise typer.Exit(code=1)

    config.set("username", username)
    keyring.set_password(SERVICE_NAME, username, password)
    print("Successfully logged in")
    return garmin


@app.command()
def summary():
    garmin = login()
    summary = garmin.get_user_summary(cdate=datetime.date.today().isoformat())
    steps = summary.get("totalSteps", 0)
    daily_step_goal = summary.get("dailyStepGoal", 0)
    sleeping_seconds = summary.get("sleepingSeconds", 0)
    active_seconds = summary.get("activeSeconds", 0)


    print(f"Steps: {steps}")
    print(f"Daily Step Goal: {daily_step_goal}")
    print(f"Active Minutes: {active_seconds // 60}")
    print(f"Sleeping Minutes: {sleeping_seconds // 60}")


if __name__ == "__main__":
    app()
