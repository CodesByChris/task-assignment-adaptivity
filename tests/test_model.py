"""Pytest setup for TaskModel."""

from unittest.mock import patch
import pytest
import networkx as nx
from taadaptivity.model import EXAMPLE_PARAMS, TaskAgent, TaskModel


@pytest.mark.parametrize("taskmodel_args", EXAMPLE_PARAMS.values())
def test_simple_run(taskmodel_args):
    """Simply run without throwing any errors."""
    model = TaskModel(**taskmodel_args)
    model.run_model()


def test_idle_system(steps = 100, num_agents = 30, fitness = 50, performance = 0.01):
    """A system where no agent has any tasks should run indefinitely."""
    model = TaskModel(params = {"num_agents": num_agents, "t_new": 2 * steps, "loc": fitness,
                                "sigma": 0, "performance": performance, "init_task_count": 0},
                      max_steps = steps, seed = None)
    model.run_model()
    assert model.schedule.steps == steps


def test_single_agent_system(steps = 100, t_new = 10, init_task_count = 15,
                             fitness = 50, performance = 0.01):
    """A system with one agent should run as well."""
    model = TaskModel(params = {"num_agents": 1, "t_new": t_new, "loc": fitness,
                                "sigma": 0, "performance": performance,
                                "init_task_count": init_task_count},
                      max_steps = steps, seed = None)
    model.run_model()
    assert model.schedule.steps == steps


def test_system_collapse(num_agents = 40, t_new = 10):
    """Test system collapse by overloading all agents with the new tasks."""
    model = TaskModel(params = {"num_agents": num_agents, "t_new": t_new, "loc": 0.5,
                                "sigma": 0, "performance": 0.01,
                                "init_task_count": 0},
                      max_steps = 2 * t_new, seed = None)
    model.run_model()
    assert model.schedule.steps == t_new, "Model did not stop early when no agents remain."
    assert model.fraction_failed_agents == 1
    assert len(model.active_agents) == 0


@pytest.mark.parametrize("taskmodel_args", EXAMPLE_PARAMS.values())
def test_step_counter(taskmodel_args):
    """Validate step counter."""
    model = TaskModel(**taskmodel_args)
    step = 0
    while step <= 100:
        assert step == model.schedule.steps
        step += 1
        model.step()


@pytest.mark.parametrize("taskmodel_args", EXAMPLE_PARAMS.values())
def test_repeatable_datacollection(taskmodel_args, num_reps = 5):
    """Check that datacollection is equal when re-running the model for the same seed."""
    for seed in range(0, 1000, 200):
        old_model_df = None
        old_agent_df = None
        for _ in range(num_reps):
            kwargs = taskmodel_args.copy()
            kwargs["seed"] = seed

            # Run model
            abm = TaskModel(**kwargs)
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


@pytest.mark.parametrize("taskmodel_args", EXAMPLE_PARAMS.values())
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


def test_fraction_failed_agents():
    """Tests the model property fraction_failed_agents."""

    # Test 1: collapsing system
    model = TaskModel(**EXAMPLE_PARAMS["SYSTEM_COLLAPSES"])
    model.run_model()
    assert model.fraction_failed_agents == 1

    # Test 2: one agent fails every step
    #     After every step, we add a number of tasks to one agent that is large
    #     enough such that it fails. We set a high "performance" so that agents
    #     do not fail from the regular task assignment dynamics.
    num_agents = 40
    fitness = 100
    model = TaskModel(params = {"num_agents": num_agents, "t_new": 2 * num_agents, "loc": fitness,
                                "sigma": 0, "performance": 10, "init_task_count": 0},
                      max_steps = None, seed = 1234)
    for step, agent in enumerate(model.schedule.agents):
        # Test fraction_failed_agents
        assert model.fraction_failed_agents == pytest.approx(step / num_agents)
        assert len(model.active_agents) == num_agents - step

        # Overload one agent
        model.step()
        for _ in range(int(fitness - agent.task_count) + 1):
            agent.add_task(sender = None)
        model._update_failures()  # pylint: disable=protected-access


def test_no_task_addition_failed_agents():
    """Tests whether assigning tasks to a failed agent raises an error."""
    model = TaskModel(**EXAMPLE_PARAMS["SINGLE_AGENT_REMAINING"])
    model.run_model()
    remaining_agent = model.active_agents[0]
    for failed_agent in filter(lambda a: a != remaining_agent, model.schedule.agents):
        try:
            failed_agent.add_task(sender = None)
        except AssertionError:
            pass
        else:
            pytest.fail("No exception raised when adding task to failed agent.")


def test_example_params_system_collapses():
    """Tests if the system has no active agents after the simulation."""
    model = TaskModel(**EXAMPLE_PARAMS["SYSTEM_COLLAPSES"])
    model.run_model()
    assert model.fraction_failed_agents == 1
    assert len(model.active_agents) == 0
    assert sum(agent.has_failed for agent in model.schedule.agents) == len(model.schedule.agents)


def test_example_params_single_agent_remaining():
    """Tests if the system has no active agents after the simulation."""
    model = TaskModel(**EXAMPLE_PARAMS["SINGLE_AGENT_REMAINING"])
    model.run_model()
    assert model.fraction_failed_agents < 1
    assert len(model.active_agents) == 1
    assert sum(not agent.has_failed for agent in model.schedule.agents) == 1


# TODO Test: agent assignee selection
# TODO Test: assigning task to failed agent raises error
# TODO Test: matrix_entropy
