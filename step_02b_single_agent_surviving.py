"""Starts the visualization with parameter values for which all agents except for one fail."""

from taadaptivity.server import build_server
from taadaptivity.model import EXAMPLE_PARAMS

def main():
    """Starts the visualization server."""
    desc = """These parameter values exemplify a case where all agents except for one fail."""
    build_server(**EXAMPLE_PARAMS["SINGLE_AGENT_REMAINING"], description = desc).launch()

if __name__ == "__main__":
    main()
