import typer
from rich import print

from . import fitness_coach


app = typer.Typer()

@app.command()
def login():
    """Login to Garmin Connect."""
    fitness_coach.login()
    print("✅ Successfully logged in")

@app.command()
def summary():
    """Show daily summary from Garmin Connect."""
    data = fitness_coach.get_summary()
    steps = data.get("totalSteps", 0)
    daily_step_goal = data.get("dailyStepGoal", 0)
    sleeping_seconds = data.get("sleepingSeconds", 0)
    active_seconds = data.get("activeSeconds", 0)

    print(f"Steps: {steps}")
    print(f"Daily Step Goal: {daily_step_goal}")
    print(f"Active Minutes: {active_seconds // 60}")
    print(f"Sleeping Minutes: {sleeping_seconds // 60}")

if __name__ == "__main__":
    app()

