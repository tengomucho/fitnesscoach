import pytest

from fitnesscoach.coach import FitnessCoachChat


@pytest.fixture(scope="module")
def fitness_coach():
    """
    Create a single FitnessCoachChat instance for all tests.
    Using module scope to avoid reloading the model for each test.
    """
    return FitnessCoachChat(verbose=False)


def test_fitness_coach_initialization_and_ask(fitness_coach):
    """Test that FitnessCoachChat initializes and ask() returns an answer."""
    # Ask a simple question
    answer = fitness_coach.ask("Hello, who are you?")
    print(answer)

    # Assertions
    assert isinstance(answer, str)
    assert len(answer) > 0



def test_ask_with_tool_call_steps(fitness_coach):
    """Test asking about steps triggers get_steps and returns answer."""

    called = False
    def fake_get_steps() -> int:
        """Get the number of steps walked today.

        Args:
            None

        Returns:
            int: The number of steps taken today
        """
        nonlocal called
        called = True
        return 8500

    # Replace the function in the instance's tools_dict
    original_get_steps = fitness_coach.tools_dict['get_steps']
    fitness_coach.tools_dict['get_steps'] = fake_get_steps

    try:
        answer = fitness_coach.ask("How many steps have I taken today?")

        # Verify the fake function was called
        assert called, "fake_get_steps was not called"

        # Assertions
        assert isinstance(answer, str)
        assert len(answer) > 0
    finally:
        # Restore original function to avoid affecting other tests
        fitness_coach.tools_dict['get_steps'] = original_get_steps


def test_ask_with_tool_call_heart_rate(fitness_coach):
    """Test asking about heart rate triggers get_heart_rate."""

    called = False
    def fake_get_heart_rate() -> tuple[int, int]:
        """Get the heart rate range for today.

        Returns:
            tuple[int, int]: Min and max heart rate
        """
        nonlocal called
        called = True
        return (60, 150)

    # Replace the function in the instance's tools_dict
    original_get_heart_rate = fitness_coach.tools_dict['get_heart_rate']
    fitness_coach.tools_dict['get_heart_rate'] = fake_get_heart_rate

    try:
        answer = fitness_coach.ask("What was my heart rate today?")

        # Verify the fake function was called
        assert called, "fake_get_heart_rate was not called"

        # Assertions
        assert isinstance(answer, str)
        assert len(answer) > 0
    finally:
        # Restore original function to avoid affecting other tests
        fitness_coach.tools_dict['get_heart_rate'] = original_get_heart_rate
