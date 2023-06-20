"""Model class."""

from typing import Dict
from mesa import Model
from mesa.space import NetworkGrid
from mesa.time import SimultaneousActivation
from networkx import adjacency_matrix, DiGraph, empty_graph
from numpy.random import default_rng
from scipy.stats import entropy
from .agent import TaskAgent


class TaskModel(Model):
    """System of TaskAgents.

    Attributes:
        max_steps: Number of steps after which to stop model execution when
            using run_model().
        num_agents: Number of agents.
        rng: Numpy RNG instance to use when sampling random numbers.
        schedule: The scheduler.
        grid: The network grid.
        datacollector: The DataCollector.
    """

    def __init__(self, max_steps: int, num_agents: int,
                 fitness_params: Dict[str, float], agent_params: Dict[str, float]):
        """Initialize the instance.

        It initializes a numpy RNG to use within the ABM in the place of Mesa's
        self.random. Numpy has the advantage that it implements the
        hypergeometric distribution that the ABM needs.

        Args:
            max_steps: Number of steps after which to stop model execution when
                using self.run_model().
            num_agents: Number of agents.
            fitness_params: The parameters of the normal distribution from which
                to sample the agents' fitness values. The following names need
                to be specified: "loc" (50), "scale". The numbers in parentheses
                specify the values used in the paper. The paper calls the
                parameters $\\mu$ (="loc") and $\\sigma$ (="scale").
            agent_params: The initial values for the agent's parameters. The
                following names need to be specified: "performance" (0.01),
                "init_task_count" (15). The numbers in parentheses specify the
                values used in the paper. Note that agents have a third
                parameter, the fitness, which is sampled for each agent
                individually upon construction.
        """
        super().__init__()
        self.max_steps = max_steps
        self.num_agents = num_agents

        # Initialize numpy RNG
        self.rng = default_rng(self._seed)

        # Initialize agents and mesa setup
        self.schedule = SimultaneousActivation(self)
        self.grid = NetworkGrid(empty_graph(num_agents, DiGraph))
        for agent_id in range(num_agents):
            # Sample agent's fitness
            curr_agent_params = agent_params.copy()
            curr_agent_params["fitness"] = self.rng.normal(**fitness_params, size=None)

            # Add agent
            agent = TaskAgent(agent_id, self, curr_agent_params)
            self.schedule.add(agent)
            self.grid.place_agent(agent, agent_id)

        # Initialize data collection
        self.initialize_data_collector(
            model_reporters={"Network": "network",
                             "Fraction_Failed": "fraction_failed_agents",
                             "Matrix_Entropy": "matrix_entropy"},
            agent_reporters={"Task_Load": "task_load"}
        )


    def step(self):
        """Advance the ABM by one time step."""
        self.schedule.step()
        if self.schedule.steps == self.max_steps:
            self.running = False


    @property
    def network(self) -> DiGraph:
        """A copy of the grid."""
        return self.grid.G.copy()


    @property
    def fraction_failed_agents(self) -> int:
        """Ratio of failed agents, see Equation (2) in the paper."""
        return sum(agent.has_failed for agent in self.schedule.agents) / self.num_agents


    @property
    def matrix_entropy(self) -> float:
        """Information entropy of the entries in the adjacency matrix, see Equation (8) in paper."""
        return entropy(adjacency_matrix(self.grid.G), axis=None)
