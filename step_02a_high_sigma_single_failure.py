"""Starts the interactive visualization server."""

from taadaptivity.server import build_server
from taadaptivity.model import PARAMS_HIGH_SIGMA_SINGLE_FAILURE

def main():
    """Starts the visualization server."""
    desc = """
        ...
    """
    build_server(**PARAMS_HIGH_SIGMA_SINGLE_FAILURE, description = desc).launch()

if __name__ == "__main__":
    main()
