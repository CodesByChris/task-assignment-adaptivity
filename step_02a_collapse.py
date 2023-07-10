"""Starts the visualization with parameter values for which all agents fail."""

from taadaptivity.server import build_server
from taadaptivity.model import EXAMPLE_PARAMS


def main():
    """Starts the visualization server."""
    desc = """
        These parameter values exemplify a case where the entire system collapses. This outcome
        happens because everyone reassigns tasks to the highly performant agents. Eventually, they
        are overburdened with tasks and fail, which releases a vast task load to the rest of the
        system. The agents keep distributing many tasks to the remaining highly performant agents,
        who then also fail. At t=920, the performance of the remaining agents is not large enough to
        handle the workload, and they all fail, leaving the system with no active agents.
    """
    build_server(**EXAMPLE_PARAMS["SYSTEM_COLLAPSES"], description=desc).launch()


if __name__ == "__main__":
    main()
