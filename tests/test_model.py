"""Pytest setup for TaskModel."""

from unittest.mock import patch
import pytest
import networkx as nx
from taadaptivity.model import TaskAgent, TaskModel
import taadaptivity.model as tm

MODEL_LIST = [tm.PARAMS_REGULAR_I,
              tm.PARAMS_REGULAR_II,
              tm.PARAMS_HIGH_SIGMA_SINGLE_AGENT_SURVIVING,
              tm.PARAMS_HIGH_SIGMA_SYSTEM_RESILIENT,
              tm.PARAMS_HIGH_SIGMA_SINGLE_FAILURE,
              tm.PARAMS_HIGH_SIGMA_SYSTEM_COLLAPSES,
              tm.PARAMS_LARGE_SYSTEM_COLLAPSES]


@pytest.fixture
def simple_model():
    """Return a simple abm instance to experiment with."""
    return tm.TaskModel({"num_agents": 20, "t_new": 10, "loc": 50, "sigma": 3,
                         "performance": 0.01, "init_task_count": 15},
                        max_steps = 100)


@pytest.mark.parametrize("taskmodel_args", MODEL_LIST)
def test_simple_run(taskmodel_args):
    """Simply run without throwing any errors."""
    model = tm.TaskModel(**taskmodel_args)
    model.run_model()


@pytest.mark.parametrize("taskmodel_args", MODEL_LIST)
def test_step_counter(taskmodel_args):
    """Validate step counter."""
    model = tm.TaskModel(**taskmodel_args)
    step = 0
    while step <= 100:
        assert step == model.schedule.steps
        step += 1
        model.step()


@pytest.mark.parametrize("taskmodel_args", MODEL_LIST)
def test_repeatable_datacollection(taskmodel_args, num_reps = 5):
    """Check that datacollection is equal when re-running the model for the same seed."""
    for seed in range(0, 1000, 200):
        old_model_df = None
        old_agent_df = None
        for _ in range(num_reps):
            kwargs = taskmodel_args.copy()
            kwargs["seed"] = seed

            # Run model
            abm = tm.TaskModel(**kwargs)
            abm.run_model()

            # Compare data collectors
            new_model_df = abm.datacollector.get_model_vars_dataframe()
            new_agent_df = abm.datacollector.get_agent_vars_dataframe()
            if not (old_model_df is None or old_agent_df is None):
                assert old_model_df.equals(new_model_df)
                assert old_agent_df.equals(new_agent_df)
            old_model_df = new_model_df
            old_agent_df = new_agent_df


def choose_recipients_circle(self, num_recipients):
    """Choose num_recipients times the next agent."""
    num_agents = len(self.model.schedule.agents)
    next_agent_id = (self.pos + 1) % num_agents
    next_agent = self.model.schedule.agents[next_agent_id]
    return [] if next_agent.has_failed else [next_agent] * num_recipients


@pytest.mark.parametrize("taskmodel_args", MODEL_LIST)
@patch.object(TaskAgent, "_choose_recipients", side_effect = choose_recipients_circle, autospec = True)
def test_network_generation_circular(mock_method, taskmodel_args):
    """Test network generation by generating a circular network."""
    model = TaskModel(**taskmodel_args)
    model.run_model()

    # Test: each agent called in every step
    assert mock_method.call_count == taskmodel_args["max_steps"] * taskmodel_args["params"]["num_agents"]
    # Note: "==" fails if we ever change the implementation of TaskModel.step()
    #     such that it calls step() and advance() only for active agents instead
    #     of all agents. This test will also fail if we change TaskAgent.step()
    #     such that it does not call TaskAgent._choose_recipients() for failed
    #     agents and directly returns an empty list to signal that no recipients
    #     shall be chosen.

    # Test: circular network
    adjacency = nx.to_numpy_array(model.network)
    for row in adjacency:
        assert (row > 0).sum() == 1 or (row == 0).all(), "Agents only have 1 neighbour (or 0 if they never reassigned any task)."

    # Test: network generation
    #     Retrieve all calls from the mock and subtract the respective
    #     number of edges from the adjacency matrix. If the ABM correctly
    #     generated a circular network, the remaining adjacency matrix
    #     should not have any edges left. If agents fail, the remaining
    #     adjacency matrix can have negative entries.
    adjacency = nx.to_numpy_array(model.network)
    num_agents = len(model.schedule.agents)
    for call in mock_method.call_args_list:
        # Schema: call(agents, num_recipients)
        agent, num_edges = call.args
        next_agent_id = (agent.pos + 1) % num_agents
        adjacency[agent.pos, next_agent_id] -= num_edges
    assert (adjacency <= 0).all(), "No circle network was generated."



# Test: system collapse after a few steps
# Test: agent assignee selection
