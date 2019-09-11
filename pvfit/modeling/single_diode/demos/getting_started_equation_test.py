import os


def test_getting_started_equation():
    """Test that getting started script runs without error."""
    exec(open(os.path.join(os.path.dirname(__file__), "getting_started_equation.py")).read())
