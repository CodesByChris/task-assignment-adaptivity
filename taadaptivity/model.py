"""Model class."""

from mesa import Model
from mesa.space import NetworkGrid
from mesa.time import SimultaneousActivation
from networkx import adjacency_matrix, DiGraph, empty_graph
from scipy.stats import entropy
from .agent import TaskAgent


class TaskModel(Model):
    """System of TaskAgents.

    Attributes:
        num_agents: Number of agents.
    """

    def __init__(self, max_steps: int, num_agents: int, sigma: float, loc: float = 50, performance: float = 0.01, init_task_count: int = 15):
        """..."""
        super().__init__()
        self.max_steps = max_steps
        self.num_agents = num_agents
        self.sigma = sigma

        # Initialize agents and mesa setup
        self.schedule = SimultaneousActivation(self)
        self.grid = NetworkGrid(empty_graph(num_agents, DiGraph))
        for agent_id in range(num_agents):
            # A negative fitness is possible but the agent fails immediately,
            # because the number of its tasks are counts, i.e. at least 0.
            fitness = self.random.normalvariate(mu=loc, sigma=sigma)
            agent = TaskAgent(agent_id, self, fitness, performance, init_task_count)
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
