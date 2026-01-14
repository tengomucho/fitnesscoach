import typer
from rich import print

from . import provider


app = typer.Typer()


@app.command()
def login():
    """Login to Garmin Connect."""
    provider.login()
    print("✅ Successfully logged in")


@app.command()
def summary():
    """Show daily summary from Garmin Connect."""
    data = provider.get_summary()
    steps = data.get("totalSteps", 0)
    daily_step_goal = data.get("dailyStepGoal", 0)
    sleeping_seconds = data.get("sleepingSeconds", 0)
    active_seconds = data.get("activeSeconds", 0)

    print(f"Steps: {steps} ({steps / daily_step_goal * 100:.2f}%)")
    print(f"Daily Step Goal: {daily_step_goal}")
    print(f"Active time: {active_seconds // 3600}h:{active_seconds % 3600 // 60}m")
    print(f"Sleeping time: {sleeping_seconds // 3600}h:{sleeping_seconds % 3600 // 60}m")


if __name__ == "__main__":
    app()
