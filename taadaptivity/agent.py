"""Agent class."""

from __future__ import annotations
from typing import Dict, List, TYPE_CHECKING
from mesa import Agent
from numpy import exp, floor, zeros  # pylint: disable=no-name-in-module
if TYPE_CHECKING:
    from .model import TaskModel


class TaskAgent(Agent):
    """Worker who solves and redistributes tasks.

    Attributes:
        tasks: Number of tasks of the agent; $x_i(t)$ in paper.
        fitness: Max number of tasks the agent can hold; $\\theta_i$ in paper.
        performance: Rate at which the agent solves tasks; $\\tau_i$ in paper.
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
            agent_params: The initial values for the agent's parameters. The following names need to
                be specified: "fitness", "performance", "init_tasks".
        """
        super().__init__(unique_id, model)
        self.fitness = agent_params["fitness"]
        self.performance = agent_params["performance"]
        self.tasks = agent_params["init_tasks"]
        self.has_failed = False
        self.determine_failure()
        self._recipients = None  # used for simultaneous update
        self._num_tasks_to_redistribute = None  # used for simultaneous update
        self._unsolved_tasks = None  # used for simultaneous update

    def add_task(self, sender: TaskAgent | None):
        """Add one task to the agent's task count and update the network."""
        assert not self.has_failed, "Attempted to assign a task to a failed agent."
        self.tasks += 1
        if sender:
            self.model.G.edges[sender.pos, self.pos]["weight"] += 1

    def count_assignments(self, recipient: TaskAgent) -> int:
        """Counts how often self has assigned tasks to recipient in the past.

        Args:
            recipient: The past recipient.

        Returns:
            The number of past task assignments from self to recipient.
        """
        return self.model.G.edges[self.pos, recipient.pos]["weight"]

    def determine_failure(self):
        """Marks the agent as failed if its tasks exceeds its fitness."""
        if not self.has_failed and self.tasks > self.fitness:
            self.has_failed = True

    def step(self):
        """Stage changes: (i) how many tasks to solve, and (ii) to whom to redistribute tasks.

        This method is the first step in a simultaneous update of all agents.
        """
        self._split_solve_redistribute_tasks()
        self._recipients = self._choose_recipients(self._num_tasks_to_redistribute)

    def advance(self):
        """Apply changes staged by step().

        This method is the second step in a simultaneous update of all agents.
        """
        self._redistribute_tasks()
        self._solve_tasks()

    @property
    def task_load(self) -> float:
        """Ratio of the agent's task solving capacity currently in use.

        If the agent has failed, this ratio is defined as 1 to represent its inability to solve
        tasks.
        """
        if self.has_failed or self.fitness <= 0:
            return 1
        return self.tasks / self.fitness

    def _split_solve_redistribute_tasks(self):
        """Distribute tasks with probability p_i. Remaining tasks are left to solve."""
        active_agents = self.model.active_agents
        if len(active_agents) == 0 or (len(active_agents) == 1 and self == active_agents[0]):
            # Special case: no redistribution because no other active agent exists
            self._num_tasks_to_redistribute = 0
            self._unsolved_tasks = 0 if self.has_failed else self.tasks
            self.tasks = 0
            return

        # Determine hypergeometric params
        if self.has_failed:
            # Agent has failed in previous time-step --> Redistribute all tasks
            # p_i = 1
            ngood = floor(self.tasks)
            nbad = 0
        else:
            # p_i = (c + self.tasks/self.fitness)/(1+c)
            ngood = floor(self.tasks)
            nbad = self.fitness - ngood

        # p_i without replacement
        hgeom_params = {"ngood": ngood, "nbad": nbad, "nsample": ngood}
        self._num_tasks_to_redistribute = self.model.rng.hypergeometric(**hgeom_params)
        self._unsolved_tasks = self.tasks - self._num_tasks_to_redistribute
        self.tasks = 0

    def _solve_tasks(self):
        """Solve the tasks for the current time step, see solution to Equation (3) in paper."""
        self._unsolved_tasks = self._unsolved_tasks * exp(-self.performance)
        self.tasks += self._unsolved_tasks
        self._unsolved_tasks = None

    def _choose_recipients(self, num_recipients: int) -> List[TaskAgent]:
        """Sample recipents for tasks. see Equation (6) in paper.

        Recipients can be sampled multiple times, in which case they appear in the returned list
        more than once.

        Args:
            num_recipients: The number of recipients to sample.

        Returns:
            The sampled recipents.
        """
        if not num_recipients:
            return []

        # Compute interaction probabilities: (1) recipient fitness, (2) previous interactions
        num_agents = len(self.model.active_agents)
        probs = zeros(num_agents)
        for id_j, agent_j in enumerate(self.model.active_agents):
            if agent_j is not self:
                probs[id_j] = agent_j.fitness * (self.count_assignments(agent_j) + 1)
        probs /= sum(probs)

        # Get recipients
        ids = self.model.rng.choice(num_agents, size=num_recipients, replace=True, p=probs)
        return [self.model.active_agents[rec_id] for rec_id in ids]

    def _redistribute_tasks(self):
        """Redistribute the tasks for the current time step."""
        assert isinstance(self._recipients, list), "Recipients not staged yet."
        for recipient in self._recipients:
            recipient.add_task(self)
        self._recipients = None
        self._num_tasks_to_redistribute = None
