"""Pytest setup for TaskModel."""

import pytest
from itertools import starmap
from networkx.algorithms.isomorphism import is_isomorphic
from taadaptivity.model import TaskModel

SIMPLE_PARAMS = {"num_agents": 20, "t_new": 10, "loc": 50, "sigma": 3, "performance": 0.01, "init_task_count": 15}

@pytest.fixture
def simple_model():
    return TaskModel(SIMPLE_PARAMS, max_steps = 100)


def test_simple_run(simple_model):
    """Simply run without throwing any errors."""
    simple_model.run_model()


def test_step_counter(simple_model):
    """Validate step counter."""
    step = 0
    while step <= 100:
        assert step == simple_model.schedule.steps
        step += 1
        simple_model.step()


def test_repeatable_datacollection(num_reps = 10):
    """Check that datacollection is equal when re-running the model for the same seed."""
    for seed in range(0, 1000, 200):
        old_model_df = None
        old_agent_df = None
        for _ in range(num_reps):
            abm = TaskModel(SIMPLE_PARAMS, max_steps = 100, seed = seed)
            abm.run_model()
            new_model_df = abm.datacollector.get_model_vars_dataframe()
            new_agent_df = abm.datacollector.get_agent_vars_dataframe()
            if not (old_model_df is None or old_agent_df is None):
                assert old_model_df.equals(new_model_df)
                assert old_agent_df.equals(new_agent_df)
            old_model_df = new_model_df
            old_agent_df = new_agent_df


# Test: exact same datacollection objects when re-run with same seed.
# Test: network generation based on mocking TaskModel
