"""Run an interactive ABM visualization."""

from taadaptivity.server import server_factory
from taadaptivity.model import EXAMPLE_PARAMS


def main():
    """Starts the visualization server."""
    server_factory(**EXAMPLE_PARAMS["REGULAR"]).launch()


if __name__ == "__main__":
    main()
