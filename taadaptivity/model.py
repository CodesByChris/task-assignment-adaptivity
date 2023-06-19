"""Model class."""

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
        num_agents: Number of agents.
        rng: Numpy RNG instance to use when sampling random numbers.
    """

    def __init__(self, max_steps: int, num_agents: int, sigma: float, loc: float = 50, performance: float = 0.01, init_task_count: int = 15):
        """..."""
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
            self.schedule.add(agent)
            self.grid.place_agent(agent, agent_id)

        # Initialize data collection
        self.initialize_data_collector(
            model_reporters={"Network": "network",
                             "Failed_Agents": "num_failed_agents",
                             "Matrix_Entropy": "matrix_entropy"}
        )


    def step(self):
        """Advances the ABM by one time step."""
        self.schedule.step()
        if self.schedule.steps == self.max_steps:
            self.running = False


    @property
    def network(self) -> DiGraph:
        """A copy of the grid."""
        return self.grid.G.copy()


    @property
    def num_failed_agents(self) -> int:
        """Number of failed agents."""
        return sum(agent.has_failed for agent in self.schedule.agents)


    @property
    def matrix_entropy(self) -> float:
        """Unnormalized information entropy of the adjacency-matrix entries."""
        return entropy(adjacency_matrix(self.grid.G), axis=None)
