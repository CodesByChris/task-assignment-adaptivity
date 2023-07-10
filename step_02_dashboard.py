"""Run an interactive ABM visualization."""

from taadaptivity.server import build_server
from taadaptivity.model import EXAMPLE_PARAMS


def main():
    """Starts the visualization server."""
    build_server(**EXAMPLE_PARAMS["REGULAR"]).launch()


if __name__ == "__main__":
    main()
