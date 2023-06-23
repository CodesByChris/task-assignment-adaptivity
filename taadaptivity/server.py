"""Interactive visualization server for the task assignment ABM."""

from typing import List
from networkx import adjacency_matrix, DiGraph
from matplotlib.cm import Greys, Reds  # pylint: disable=no-name-in-module
from matplotlib.colors import to_hex
from mesa.visualization.modules import ChartModule, NetworkModule, PieChartModule
from mesa.visualization.ModularVisualization import ModularServer
from mesa.visualization.UserParam import NumberInput, Slider
from scipy.stats import entropy
from .model import TaskModel


# Network Rendering Function
def rescale(values: List[float] | float, lower_in: float | None, upper_in: float | None,
            lower_out: float, upper_out: float) -> List[float] | float:
    """Rescales the provided values into the interval [lower_out, upper_out].

    The rescaling maps the following interval bounds:
    - lower_in --> lower_out
    - upper_in --> upper_out
    and all input values are assumed to be in [lower_in, upper_in].

    Args:
        values: The values to rescale. They are supposed to be in the interval
            [lower_in, upper_in]. Alternatively, values can be a single float,
            in which case this value is treated as a list of one element.
            Moreover, then also the return value is a single float instead of a
            list with one element.
        lower_in: Lower bound of the source interval. If None, rescale takes
            `min(values)`. However, you can manually set lower_in to a smaller
            value if needed. This can be useful, for example, when rescale is
            called on multiple ranges, and you want to use a global minimum
            value across all calls.
        upper_in: Same as lower_in but for the upper bound of the source
            interval. If None, rescale takes `max(values)`.
        lower_out: Lower bound of the target interval.
        upper_out: Upper bound of the target interval.

    Returns:
        The rescaled values.

    Examples:
        >>> rescale([1, 2, 3], None, None, 0, 1)
        [0.0, 0.5, 1.0]
        >>> rescale([1, 2, 3], lower_in = 0, upper_in = 4, lower_out = 0, upper_out = 1)
        [0.25, 0.5, 0.75]
        >>> rescale([1, 2, 3], None, None, 10, 30)
        [10.0, 20.0, 30.0]
        >>> rescale([1, 2, 3], lower_in = 0, upper_in = 4, lower_out = 10, upper_out = 30)
        [15.0, 20.0, 25.0]
        >>> rescale(2, lower_in = 0, upper_in = 4, lower_out = 10, upper_out = 30)
        20.0
    """
    # Case handling
    is_single_value = isinstance(values, (int, float))
    if is_single_value:
        values = [values]
    if lower_in is None:
        lower_in = min(values)
    if upper_in is None:
        upper_in = max(values)

    # Rescale values
    unit_values = [(v - lower_in) / (upper_in - lower_in) for v in values]  # rescaled into [0, 1]
    scaled_values = [u * (upper_out - lower_out) + lower_out for u in unit_values]
    return scaled_values[0] if is_single_value else scaled_values


def network_portrayal(G: DiGraph, min_size = 2, max_size = 15):  # pylint: disable=invalid-name
    """Returns a plotting layout specifying how to plot the nodes and edges in G.

    Args:
        G: The network whose plotting layout shall be computed. ModularServer
            passes this argument automatically as abm.G, where abm is an
            instance of TaskModel.
        min_size: Minimum node size in plot.
        max_size: Maximum node size in plot.

    Returns:
        The layout as a dict with two components:
        1. "nodes": A list whose entries correspond to the nodes in G and whose
           values are dicts specifying the plotting style of the respective
           nodes.
        2. "edges": Same as "nodes" but for the edges.
    """
    portrayal = {}

    # Node renderer
    agents = [ags[0] for _, ags in G.nodes(data = "agent")]
    min_fitness = min(a.fitness for a in agents)
    max_fitness = max(a.fitness for a in agents)
    portrayal["nodes"] = []
    for agent_id in G.nodes:
        agent = G.nodes[agent_id]["agent"][0]
        portrayal["nodes"].append({
            "id": agent_id,
            "size": rescale(agent.fitness, min_fitness, max_fitness, min_size, max_size),
            "color": "black" if agent.has_failed else to_hex(Reds(agent.task_load)),
        })

    # Edge renderer
    adj_matrix = adjacency_matrix(G)
    max_adj = adj_matrix.max()

    portrayal["edges"] = []
    for edge_id, (source, target) in enumerate(G.edges):
        num_edges = int(adj_matrix[source, target])  # int() converts an np.int64 to Python int
        opacity = 0 if num_edges == 0 else rescale(num_edges, 0, max_adj, 0, 1)
        portrayal["edges"].append({
            "id": edge_id,
            "source": source,
            "target": target,
            "color": to_hex((0, 0, 0, opacity), keep_alpha = True),
            "alpha": 0.5,
            "width": 2,
        })

    return portrayal



# Plots
network_plot = NetworkModule(network_portrayal)
line_plot = ChartModule([{"Label": "Matrix_Entropy", "Color": "blue"},  # lock-in strength
                         {"Label": "Fraction_Failed", "Color": "red"}],
                         data_collector_name = "datacollector")
pie_plot = PieChartModule([{"Label": "Fraction_Active", "Color": "green"},
                           {"Label": "Fraction_Failed", "Color": "red"}],
                          data_collector_name = "datacollector")


# Input Widgets: Sliders and NumberInputs
model_params = {
    "seed": NumberInput("Random seed", value = 1234, description = "Random seed"),
    "num_agents": Slider("Number of agents", value = 50, min_value = 0, max_value = 100,
                         description = "Number of agents."),
    "t_new": Slider("Task arrival lag", value = 10, min_value = 0, max_value = 50,
                    description = "Number of steps after which each agent receives one task."),
    "init_task_count": Slider("Initial task count", value = 15, min_value = 0, max_value = 50,
                              description = "Task count of each agent at step 0."),
    "performance": Slider("Performance", value = 0.01, min_value = 0, max_value = 1, step = 0.01,
                          description="Agents' performance."),
    "sigma": Slider("sigma", value = 8.5, min_value = 0, step = 0.1, max_value = 50,
                    description = "Standard deviation of the agents' fitness."),
    "loc": 50,
    "max_steps": 1000,
}


# Server
class TaskModelViz(TaskModel):
    """Helper for server.py."""
    def __init__(self, *args, **kwargs):
        """Connect slider parameters to TaskModel.__ini__ and collect data for PieChartModule."""

        # Adjust __init__ arguments
        kwargs["params"] = {
            "num_agents": kwargs.pop("num_agents"),
            "t_new": kwargs.pop("t_new"),
            "sigma": kwargs.pop("sigma"),
            "loc": kwargs.pop("loc"),
            "performance": kwargs.pop("performance"),
            "init_task_count": kwargs.pop("init_task_count"),
        }
        super().__init__(*args, **kwargs)

        # Collect additional variables
        self.initialize_data_collector(
            model_reporters={"Network": "network",
                             "Fraction_Failed": "fraction_failed_agents",
                             "Fraction_Active": lambda model: 1 - model.fraction_failed_agents,
                             "Matrix_Entropy": "relative_entropy"},
            agent_reporters={"Task_Load": "task_load"}
        )

    @property
    def relative_entropy(self):
        """Normalize matrix entropy into [0, 1]."""
        # Maximum entropy: all edges equally likely except for selfloops, which have probability 0.
        num_free_entries = self.G.number_of_nodes() * (self.G.number_of_nodes() - 1)
        equal_probs = [1 / num_free_entries] * num_free_entries
        return self.matrix_entropy / entropy(equal_probs)


server = ModularServer(
    TaskModelViz,
    visualization_elements = [network_plot, line_plot, pie_plot],
    name = "Task Assignment Model",
    model_params = model_params,
    port = None
)
