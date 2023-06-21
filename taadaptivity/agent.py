"""Agent class."""

from __future__ import annotations
from typing import Dict, TYPE_CHECKING
from mesa import Agent
from numpy import exp, floor, zeros
if TYPE_CHECKING:
    from .model import TaskModel


class TaskAgent(Agent):
    """Worker who solves and redistributes tasks.

    Attributes:
        task_count: Number of tasks the agent has to solve; $x_i(t)$ in paper.
        fitness: The maximum number of tasks the agent can solve; $\\theta_i$ in
            paper.
        performance: The rate at which the agent solves tasks; $\\tau_i$ in paper.
        has_failed: Boolean representing whether the agent has failed.
    """

    def __init__(self,
                 unique_id: int,
                 model: TaskModel,
                 agent_params: Dict[str, float]):
        """Initialize the instance.

        Args:
            unique_id: Agent's ID.
            model: Reference to the model containing the agent.
            agent_params: The initial values for the agent's parameters. The
                following names need to be specified: "fitness", "performance",
                "init_task_count".
        """
        super().__init__(unique_id, model)
        self.fitness = agent_params["fitness"]
        self.performance = agent_params["performance"]
        self.task_count = agent_params["init_task_count"]
        self.has_failed = self.task_count > self.fitness
        self._recipients = None          # used for simultaneous update
        self._unsolved_task_count = None  # used for simultaneous update
        self._num_tasks_to_redistribute = None  # used for simultaneous update


    def add_task(self, sender: TaskAgent):
        """Add one task to the agent's task count and update the network."""
        assert not self.has_failed, "Attempted to assign a task to a failed agent."
        self.task_count += 1

        # Update network
        network = self.model.G
        if not network.has_edge(sender.pos, self.pos):
            network.add_edge(sender.pos, self.pos, weight = 0)
        network.edges[sender.pos, self.pos]["weight"] += 1


    def step(self):
        """Stage changes: (i) how many tasks to solve, and (ii) to whom to redistribute tasks.

        This method is the first step in a simultaneous update of all agents.
        """
        self._split_solve_redistribute_tasks()
        self._solve_tasks()
        self._recipients = [self._choose_recipient()
                            for _ in range(self._num_tasks_to_redistribute)]
        # self._recipients is a list because the same recipient can be chosen multiple times.


    def advance(self):
        """Apply changes staged by step().

        This method is the second step in a simultaneous update of all agents.
        It also takes care to switch the agent to failed if its task load
        exceeds the threshold (i.e. self.fitness).
        """
        self._redistribute_tasks()
        self.task_count += self._unsolved_task_count

        if self.task_count > self.fitness:
            self.has_failed = True


    @property
    def task_load(self) -> float:
        """Ratio of the agent's task solving capacity currently in use.

        If the agent has failed, this ratio is defined as 1 to represent its
        inability to solve tasks.
        """
        if self.has_failed or self.fitness <= 0:
            return 1
        return self.task_count / self.fitness


    def _split_solve_redistribute_tasks(self):
        """Distribute tasks with probability p_i. Remaining tasks are left to solve."""
        if self.has_failed:
            # Agent has failed in previous time-step --> Redistribute all tasks
            # p_i = 1
            nbad = 0
            ngood = floor(self.task_count)
        else:
            # p_i = (c + self.x/self.theta)/(1+c)
            nbad = self.fitness - floor(self.task_count)
            ngood = floor(self.task_count)

        # p_i without replacement
        hgeom_params = {"ngood": ngood, "nbad": nbad, "nsample": floor(self.task_count)}
        self._num_tasks_to_redistribute = self.model.rng.hypergeometric(**hgeom_params)
        self.task_count -= self._num_tasks_to_redistribute
        self._unsolved_task_count = self.task_count
        self.task_count = 0


    def _solve_tasks(self):
        """Solve the tasks for the current time step, see Equation (3) in paper."""
        self._unsolved_task_count = self._unsolved_task_count * exp(-self.performance)


    def _choose_recipient(self) -> TaskAgent:
        """Sample a recipent for a task. see Equation (6) in paper.

        Returns:
            The sampled recipent.
        """
        # Compute interaction probabilities: (1) recipient fitness, (2) previous interactions
        num_agents = self.model.num_agents
        grid = self.model.grid
        recipient_probs = zeros(num_agents)
        for j in range(num_agents):
            agent_j = grid.get_cell_list_contents([j])[0]
            if agent_j is not self and not agent_j.has_failed:
                num_interactions = grid.G.number_of_edges(self.pos, j)
                fitness_j = agent_j.fitness
                recipient_probs[j] = fitness_j * (num_interactions + 1)
        recipient_probs /= sum(recipient_probs)

        # Get recipient
        recipient_id = self.model.rng.choice(num_agents, p=recipient_probs)
        return grid.get_cell_list_contents([recipient_id])[0]


    def _redistribute_tasks(self):
        """Redistribute the tasks for the current time step."""
        assert isinstance(self._recipients, list), "Recipients not staged yet."
        for recipient in self._recipients:
            recipient.add_task(self)
        self._recipients = None
