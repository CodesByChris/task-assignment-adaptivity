"""Pytest setup for TaskModel."""
# pylint: disable=protected-access

from collections import defaultdict, Counter
from unittest.mock import patch
import pytest
import networkx as nx
from scipy.stats import entropy
from taadaptivity.model import EXAMPLE_PARAMS, TaskAgent, TaskModel


@pytest.mark.parametrize("taskmodel_args", EXAMPLE_PARAMS.values())
def test_simple_run(taskmodel_args):
    """Simply run without throwing any errors."""
    model = TaskModel(**taskmodel_args)
    model.run_model()


def test_idle_system(steps=100, num_agents=30, fitness=50, performance=0.01):
    """A system where no agent has any tasks should run indefinitely."""
    model = TaskModel(params={"num_agents": num_agents, "t_new": 2 * steps, "loc": fitness,
                              "sigma": 0, "performance": performance, "init_tasks": 0},
                      max_steps=steps, seed=None)
    model.run_model()
    assert model.schedule.steps == steps


def test_single_agent_system(steps=100, t_new=10, init_tasks=15, fitness=50, performance=0.01):
    """A system with one agent should run as well."""
    model = TaskModel(params={"num_agents": 1, "t_new": t_new, "loc": fitness, "sigma": 0,
                              "performance": performance, "init_tasks": init_tasks},
                      max_steps=steps, seed=None)
    model.run_model()
    assert model.schedule.steps == steps


def test_system_collapse(num_agents=40, t_new=10):
    """Test system collapse by overloading all agents with the new tasks."""
    model = TaskModel(params={"num_agents": num_agents, "t_new": t_new, "loc": 0.5, "sigma": 0,
                              "performance": 0.01, "init_tasks": 0},
                      max_steps=2 * t_new, seed=None)
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
def test_no_tasks_lost(taskmodel_args):
    """Test that no agent loses tasks marked for redistribution."""

    # Create model
    model = TaskModel(**taskmodel_args)
    all_agents = model.schedule.agents

    # Track task redistribution
    num_steps = 0
    while num_steps < taskmodel_args["max_steps"]:
        task_assignments = defaultdict(Counter)
        old_network = model.network

        # Track step
        for agent in all_agents:
            agent.step()
            assert agent._num_tasks_to_redistribute == len(agent._recipients)
            task_assignments[agent.pos].update(a.pos for a in agent._recipients)

        # Track advance
        for agent in all_agents:
            agent.advance()
        new_network = model.network

        # Validate network structure
        for source_node, target_node, old_weight in old_network.edges.data("weight"):
            new_weight = new_network.edges[source_node, target_node]["weight"]
            assert new_weight - old_weight == task_assignments[source_node][target_node]

        model._update_failures()
        num_steps += 1


@pytest.mark.parametrize("taskmodel_args", EXAMPLE_PARAMS.values())
def test_repeatable_datacollection(taskmodel_args, num_reps=5):
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
@patch.object(TaskAgent, "_choose_recipients", side_effect=choose_recipients_circle, autospec=True)
def test_network_generation_circular(mock_method, taskmodel_args):
    """Test network generation by generating a circular network."""
    model = TaskModel(**taskmodel_args)
    model.run_model()

    # Test: each agent called in every step
    max_steps = taskmodel_args["max_steps"]
    num_agents = taskmodel_args["params"]["num_agents"]
    assert mock_method.call_count == max_steps * num_agents
    # Note: "==" fails if we ever change the implementation of TaskModel.step()
    #     such that it calls step() and advance() only for active agents instead
    #     of all agents. This test will also fail if we change TaskAgent.step()
    #     such that it does not call TaskAgent._choose_recipients() for failed
    #     agents and directly returns an empty list to signal that no recipients
    #     shall be chosen.

    # Test: circular network
    adjacency = nx.to_numpy_array(model.network)
    for row in adjacency:
        assert (row > 0).sum() == 1 or (row == 0).all(), "Agents cannot have more than 1 neighbour."

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
    #     After every step, we knock out the remaining agent with the smallest
    #     fitness by adding a number of tasks that is large enough such that it
    #     fails. Because this agent has the smallest fitness, no other agent
    #     will fail even when receiving all tasks of the failing agent. We set a
    #     high "performance" so that agents do not fail from the regular task
    #     assignment dynamics.
    num_agents = 40
    mean_fitness = 100
    model = TaskModel({"num_agents": num_agents, "t_new": 2 * num_agents, "loc": mean_fitness,
                       "sigma": 3, "performance": 10, "init_tasks": 0},
                      max_steps=None, seed=1234)
    sorted_agents = sorted(model.schedule.agents, key=lambda a: a.fitness)
    for step, agent in enumerate(sorted_agents):
        # Test fraction_failed_agents
        assert model.fraction_failed_agents == pytest.approx(step / num_agents)
        assert len(model.active_agents) == num_agents - step

        # Overload one agent
        model.step()
        for _ in range(int(agent.fitness - agent.tasks) + 1):
            agent.add_task(sender=None)
        model._update_failures()  # pylint: disable=protected-access


def choose_all_recipients(self, _):
    """Assign one task to every other agent."""
    return [a for a in self.model.schedule.agents if a.unique_id != self.unique_id]


@patch.object(TaskAgent, "_choose_recipients", side_effect=choose_all_recipients, autospec=True)
def test_matrix_entropy_maximum_case(mock_method):
    """Tests whether a network with equal edge counts everywhere yields the highest entropy."""

    # Generate fully-connected network with edge-multiplicity 1000 for all pairs
    num_steps = 1000
    model = TaskModel({"num_agents": 50, "t_new": 10, "loc": 100, "sigma": 0,
                       "performance": 10, "init_tasks": 0},
                      max_steps=num_steps, seed=1234)
    model.run_model()
    assert all(weight == num_steps for _, _, weight in model.network.edges.data("weight"))

    # Test maximal entropy
    num_agents = len(model.schedule.agents)
    num_pairs = num_agents * (num_agents - 1)
    max_entropy = entropy([1 / num_pairs] * num_pairs)
    assert model.matrix_entropy == pytest.approx(max_entropy)


def test_no_task_addition_failed_agents():
    """Tests whether assigning tasks to a failed agent raises an error."""
    model = TaskModel(**EXAMPLE_PARAMS["SYSTEM_COLLAPSES"])
    model.run_model()
    for failed_agent in filter(lambda a: a.has_failed, model.schedule.agents):
        with pytest.raises(AssertionError):
            failed_agent.add_task(sender=None)


def test_last_agent():
    """Tests whether last active agent is handled correctly."""

    # Advance model until one agent remains
    model = TaskModel(params={"num_agents": 10, "t_new": 10, "loc": 50, "sigma": 16,
                              "performance": 0.01, "init_tasks": 15},
                      max_steps=1000, seed=1232)
    remaining_agents = model.active_agents
    while len(remaining_agents) > 1:
        model.step()
        remaining_agents = model.active_agents
        # Note: model._update_failures re-generates the list behind
        #     model.active_agents. Hence, we have to assign it again to
        #     remaining_agents after each step.
    assert len(remaining_agents) == 1, "Model params for which no single agent remains."

    # Test last agent gets all tasks from failed agents
    last_agent = model.active_agents[0]
    prev_fails = [a for a in model.schedule.agents if a.has_failed and a.tasks > a.fitness]
    for agent in model.schedule.agents:
        agent.step()

    assert len(prev_fails) > 0
    assert all(set(a._recipients) == {last_agent} for a in prev_fails), "More than one recipient."
    assert all(a._unsolved_tasks < 1 for a in prev_fails)
    assert all(a._num_tasks_to_redistribute == len(a._recipients) for a in prev_fails)

    # Test last agent re-assigns no tasks
    assert len(last_agent._recipients) == 0


def test_example_params_system_collapses():
    """Tests if the system has no active agents after the simulation."""
    model = TaskModel(**EXAMPLE_PARAMS["SYSTEM_COLLAPSES"])
    model.run_model()
    assert model.fraction_failed_agents == 1
    assert len(model.active_agents) == 0
    assert sum(agent.has_failed for agent in model.schedule.agents) == len(model.schedule.agents)
