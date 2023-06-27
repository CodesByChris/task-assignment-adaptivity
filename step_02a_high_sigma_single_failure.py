"""Starts the interactive visualization server."""

from taadaptivity.server import build_server
from taadaptivity.model import EXAMPLE_PARAMS

def main():
    """Starts the visualization server."""
    desc = """
        ...
    """
    build_server(**EXAMPLE_PARAMS["HIGH_SIGMA_SINGLE_FAILURE"], description = desc).launch()

if __name__ == "__main__":
    main()
