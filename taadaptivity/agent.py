"""Agent class."""

from __future__ import annotations
from typing import Dict, TYPE_CHECKING
from mesa import Agent
from numpy import exp, floor, zeros  # pylint: disable=no-name-in-module
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


    def add_task(self, sender: TaskAgent | None):
        """Add one task to the agent's task count and update the network."""
        assert not self.has_failed, "Attempted to assign a task to a failed agent."
        self.task_count += 1
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
        """Marks the agent as failed if its task_count exceeds its fitness."""
        if not self.has_failed and self.task_count > self.fitness:
            self.has_failed = True


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
        """
        self._redistribute_tasks()
        self.task_count += self._unsolved_task_count


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
            ngood = floor(self.task_count)
            nbad = 0
        else:
            # p_i = (c + self.task_count/self.fitness)/(1+c)
            ngood = floor(self.task_count)
            nbad = self.fitness - ngood

        # p_i without replacement
        hgeom_params = {"ngood": ngood, "nbad": nbad, "nsample": ngood}
        self._num_tasks_to_redistribute = self.model.rng.hypergeometric(**hgeom_params)
        self._unsolved_task_count = self.task_count - self._num_tasks_to_redistribute
        self.task_count = 0


    def _solve_tasks(self):
        """Solve the tasks for the current time step, see solution to Equation (3) in paper."""
        self._unsolved_task_count = self._unsolved_task_count * exp(-self.performance)


    def _choose_recipient(self) -> TaskAgent:
        """Sample a recipent for a task. see Equation (6) in paper.

        Returns:
            The sampled recipent.
        """
        # Compute interaction probabilities: (1) recipient fitness, (2) previous interactions
        num_agents = len(self.model.schedule.agents)
        probs = zeros(num_agents)
        for agent_j in self.model.schedule.agents:
            if agent_j is not self and not agent_j.has_failed:
                probs[agent_j.pos] = agent_j.fitness * (self.count_assignments(agent_j) + 1)
        probs /= sum(probs)

        # Get recipient
        recipient_pos = self.model.rng.choice(num_agents, p = probs)
        return self.model.grid.get_cell_list_contents([recipient_pos])[0]


    def _redistribute_tasks(self):
        """Redistribute the tasks for the current time step."""
        assert isinstance(self._recipients, list), "Recipients not staged yet."
        for recipient in self._recipients:
            recipient.add_task(self)
        self._recipients = None
