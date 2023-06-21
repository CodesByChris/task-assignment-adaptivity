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
        t_new: Number of time steps after which one new task arrives at each
            agent. The paper uses 10, see p.4.
        rng: Numpy RNG instance to use when sampling random numbers.
        schedule: The scheduler.
        grid: The network grid.
        datacollector: The DataCollector.
    """

    def __init__(self, params: Dict[str, float], max_steps: int, seed: int | None = None):
        """Initialize the instance.

        It initializes a numpy RNG to use within the ABM in the place of Mesa's
        self.random. Numpy has the advantage that it implements the
        hypergeometric distribution that the ABM needs.

        Args:
            max_steps: Number of steps after which to stop model execution when
                using self.run_model().
            params: Represents the parameters of the model. It is a dict with
                the following entries:
                1. "num_agents": Number of agents.
                2. "t_new": Time difference after which external tasks arrive.
                   The paper calls this parameter $T_{new}$ and sets it to 10.
                3. "loc": The location parameter of the normal distribution from
                   which to sample the agents' fitness values. The paper calls
                   this parameter $\\mu$ and sets it to 50.
                4. "scale": The standard deviation parameter of the normal
                   distribution from which to sample the agents' fitness values.
                   The paper calls this parameter $\\sigma$.
                5. "performance": The first agent parameter. It is the rate at
                   which the agents solve tasks. All agents receive the same
                   value. The paper calls this parameter $\\tau_i$ and sets it
                   to 0.01.
                6. "init_task_count": The second agent parameter. It is the
                   number of tasks each agent has at the beginning of the
                   simulation. The paper sets it to 15.
                Note that agents have a third parameter, the fitness
                $\\theta_i$, which is sampled for each agent individually upon
                construction.
            seed: Random seed for the random number generator. Note that Mesa
                uses the same seed to its default RNG in self.random, but which
                is not used here. See __new__ in Mesa's Model class.
        """
        super().__init__()
        self.max_steps = max_steps
        self.t_new = params["t_new"]

        # Initialize numpy RNG
        self.rng = default_rng(seed)

        # Initialize agents and mesa setup
        #     Name ".G" required in server.py. pylint: disable-next=invalid-name
        self.G = empty_graph(params["num_agents"], DiGraph)
        self.grid = NetworkGrid(self.G)
        self.schedule = SimultaneousActivation(self)
        for agent_id in range(params["num_agents"]):
            # Sample agent's fitness
            curr_agent_params = {
                "performance": params["performance"],
                "init_task_count": params["init_task_count"],
                "fitness": self.rng.normal(loc=params["loc"], scale=params["scale"], size=None),
            }

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
        # Advance agents
        curr_step = self.schedule.steps
        self.schedule.step()

        # Add new tasks
        if curr_step > 0 and curr_step % self.t_new == 0:
            for agent in self.schedule.agents:
                if not agent.has_failed:
                    agent.add_task(sender = None)

        # Collect data and check completion
        self.datacollector.collect(self)
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
        if self.G.number_of_edges() == 0:
            # We consider an empty network as fully deterministic, i.e. no randomness
            return 0

        adjacency = adjacency_matrix(self.G).toarray()  # entropy() does not handle sparsearrays
        probs = adjacency / adjacency.sum()
        return entropy(probs, axis=None)
