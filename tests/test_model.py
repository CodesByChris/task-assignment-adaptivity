"""Pytest setup for TaskModel."""

import pytest
from taadaptivity.model import TaskModel

@pytest.fixture
def simple_model():
    params = {"num_agents": 20, "loc": 50, "scale": 3, "performance": 0.01, "init_task_count": 15}
    return TaskModel(params, max_steps=100)


def test_simple_run(simple_model):
    """This test should simply run through without throwing any errors."""
    simple_model.run_model()



# Test: exact same datacollection objects when re-run with same seed.
