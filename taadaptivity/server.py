"""Interactive visualization server for the task assignment ABM."""

from typing import Optional, TypedDict
from networkx import adjacency_matrix, DiGraph
from matplotlib.cm import Reds  # pylint: disable=no-name-in-module
from matplotlib.colors import to_hex
from mesa.visualization.modules import ChartModule, NetworkModule
from mesa.visualization.ModularVisualization import ModularServer
from mesa.visualization.UserParam import NumberInput, Slider
from scipy.stats import entropy
from .model import ModelParams, TaskModel


# Types
NodeLayout = TypedDict("NodeLayout", {"id": int, "size": float, "color": str})
EdgeLayout = TypedDict("EdgeLayout", {"id": int, "source": int, "target": int,
                                      "color": str, "width": float})
Portrayal = TypedDict("Portrayal", {"nodes": list[NodeLayout], "edges": list[EdgeLayout]})


# Network Rendering Function
def rescale(value: float, lower_in: float, upper_in: float,
            lower_out: float, upper_out: float) -> float:
    """Rescales the provided value from the interval [lower_in, upper_in] to [lower_out, upper_out].

    Args:
        value: The value to rescale. It is supposed to be in the interval [lower_in, upper_in].
        lower_in: Lower bound of the source interval.
        upper_in: Same as lower_in but for the upper bound of the source interval.
        lower_out: Lower bound of the target interval.
        upper_out: Upper bound of the target interval.

    Returns:
        The rescaled values.

    Examples:
        >>> rescale(2, lower_in = 0, upper_in = 4, lower_out = 10, upper_out = 30)
        20.0
    """
    unit_value = (value - lower_in) / (upper_in - lower_in)  # rescaled into [0, 1]
    return unit_value * (upper_out - lower_out) + lower_out


# pylint: disable-next=invalid-name
def network_portrayal(G: DiGraph, min_size=2, max_size=15) -> Portrayal:
    """Returns a plotting layout specifying how to plot the nodes and edges in G.

    Args:
        G: The network whose plotting layout shall be computed. ModularServer passes this argument
            automatically as abm.G, where abm is an instance of TaskModel.
        min_size: Minimum node size in plot.
        max_size: Maximum node size in plot.

    Returns:
        The layout as a dict with two entries:
        1. "nodes": A list whose entries correspond to the nodes in G and whose values are dicts
           specifying the plotting style of the respective nodes, i.e. "id", "size", and "color".
        2. "edges": Same as "nodes" but for the edges.
    """
    portrayal: Portrayal = {"nodes": [], "edges": []}

    # Node renderer
    agents = [ags[0] for _, ags in G.nodes(data="agent")]
    min_fitness = min(a.fitness for a in agents)
    max_fitness = max(a.fitness for a in agents)

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

    for edge_id, (source, target) in enumerate(G.edges):
        num_edges = int(adj_matrix[source, target])  # int() converts an np.int64 to Python int
        opacity = 0 if num_edges == 0 else rescale(num_edges, 0, max_adj, 0, 1)
        portrayal["edges"].append({
            "id": edge_id,
            "source": source,
            "target": target,
            "color": to_hex((0, 0, 0, opacity), keep_alpha=True),
            "width": 2,
        })

    return portrayal


class TaskModelViz(TaskModel):
    """Helper for ModularServer."""
    def __init__(self, *args, **kwargs):
        """Connect slider parameters to TaskModel.__init__ and collect data for PieChartModule."""

        # Adjust __init__ arguments
        kwargs["params"] = {
            "num_agents": kwargs.pop("num_agents"),
            "t_new": kwargs.pop("t_new"),
            "sigma": kwargs.pop("sigma"),
            "loc": kwargs.pop("loc"),
            "performance": kwargs.pop("performance"),
            "init_tasks": kwargs.pop("init_tasks"),
        }
        super().__init__(*args, **kwargs)

        # Collect additional variables
        self.initialize_data_collector(
            model_reporters={"Fraction_Failed": "fraction_failed_agents",
                             "Matrix_Entropy": "relative_entropy"},
            agent_reporters={"Task_Load": "task_load"}
        )

    @property
    def relative_entropy(self) -> float:
        """Normalize matrix entropy into [0, 1]."""
        # Maximum entropy: all edges equally likely except for selfloops, which have probability 0.
        num_free_entries = self.G.number_of_nodes() * (self.G.number_of_nodes() - 1)
        equal_probs = [1 / num_free_entries] * num_free_entries
        return self.matrix_entropy / entropy(equal_probs)


def server_factory(params: ModelParams, max_steps: int, seed: int,
                   description: Optional[str] = None) -> ModularServer:
    """Builds a ModularServer with the provided parameters as initial values of the sliders."""
    # Plot widgets
    network_plot = NetworkModule(network_portrayal, canvas_height=550, canvas_width=864)
    legend = """
        <div style="text-align: center;">
            Agents: <span style="padding:10px;"/>
            <span style="color:#CC0000;">&#x25CF;</span>
                active (color-intensity: task load, size: fitness)
            <span style="padding:10px;"/> &#x25CF; failed
        </div>
    """
    line_plot = ChartModule([{"Label": "Matrix_Entropy", "Color": "blue"},  # lock-in strength
                             {"Label": "Fraction_Failed", "Color": "black"}],
                            data_collector_name="datacollector")

    visualization_elements = [lambda _: legend, network_plot, line_plot]
    if description:
        visualization_elements.append(lambda _: description)

    # Input widgets: Sliders and NumberInputs
    model_params = {
        "seed": NumberInput("Random seed", value=seed, description="Random seed"),
        "num_agents": Slider("Number of agents", value=params["num_agents"],
                             min_value=0, max_value=100, description="Number of agents."),
        "t_new": Slider("Task arrival lag", value=params["t_new"], min_value=0, max_value=50,
                        description="Number of steps after which each agent receives one task."),
        "init_tasks": Slider("Initial task count", value=params["init_tasks"],
                             min_value=0, max_value=50,
                             description="Task count of each agent at step 0."),
        "performance": Slider("Performance", value=params["performance"], min_value=0, max_value=1,
                              step=0.01, description="Agents' performance."),
        "sigma": Slider("sigma", value=params["sigma"], min_value=0, max_value=50,
                        step=0.1, description="Standard deviation of the agents' fitness."),
        "loc": params["loc"],
        "max_steps": max_steps,
    }

    # Server
    return ModularServer(TaskModelViz, visualization_elements=visualization_elements,
                         name="Task Assignment Model", model_params=model_params, port=None)
