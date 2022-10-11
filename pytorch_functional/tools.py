#  Copyright (c) 2022 Szymon Mikler

from __future__ import annotations

from itertools import chain
from typing import TYPE_CHECKING, Any, Callable, Iterable, Tuple

if TYPE_CHECKING:
    from .functional_model import FunctionalModel
    from .symbolic import SymbolicTensor

from collections import defaultdict

import torch
from torch import nn


def get_parameter_count(model: nn.Module):
    """Get the number of parameters of a model."""
    cnt = 0
    for param in model.parameters():
        cnt += param.shape.numel()
    return cnt


def get_parameter_shapes(model: nn.Module):
    """Get the shapes of parameters of a model."""
    shapes = []
    for param in model.parameters():
        shapes.append(tuple(param.shape))
    return shapes


def model_similar(a: nn.Module, b: nn.Module):
    """Check whether two models have the same number of parameters and the same shapes of parameters."""
    if get_parameter_count(a) != get_parameter_count(b):
        return False

    if sorted(get_parameter_shapes(a)) != sorted(get_parameter_shapes(b)):
        return False
    return True


def hash_torch_tensor(tensor: torch.Tensor):
    """Interpret the tensor as a string and return its hash."""
    tensor_as_string = str(tensor.flatten().tolist())
    return hash(tensor_as_string)


def models_have_corresponding_parameters(a: nn.Module, b: nn.Module):
    """Check whether two models' parameters have identical hash values.

    Parameter order does not matter.
    So if two models have identical parameters but in different order, this will still return True.
    """
    hashes_a = [hash_torch_tensor(p) for p in a.parameters()]
    hashes_b = [hash_torch_tensor(p) for p in b.parameters()]
    return set(hashes_a) == set(hashes_b)


def default_node_text(plh: SymbolicTensor) -> str:
    return str(plh.shape)


def default_edge_text(layer: nn.Module | None) -> str:
    return str(layer)


def calc_sum_sq_distances(graph, positions_dict):
    sum_sq_distances = 0
    for node in positions_dict:
        related_nodes = list(graph.predecessors(node))

        for n2 in related_nodes:
            x1, y1 = positions_dict[node]
            x2, y2 = positions_dict[n2]
            sum_sq_distances += (x1 - x2) ** 2 + (y1 - y2) ** 2
    return sum_sq_distances


def fix_positions_in_multipartite_layout(graph, orig_positions_dict):
    import networkx as nx
    import numpy as np
    import scipy

    assert isinstance(graph, nx.DiGraph)

    orig_sum = calc_sum_sq_distances(graph, orig_positions_dict)

    positions = list(orig_positions_dict.values())
    selected_positions = {}

    layers = defaultdict(list)
    for pos in positions:
        layers[pos[1]].append(pos[0])

    layer_to_nodes = defaultdict(list)
    for node, pos in orig_positions_dict.items():
        layer_to_nodes[pos[1]].append(node)

    # Layers are just unique Y positions on the plot
    # Go through the highest layer to the lowest one
    for layer, nodes_in_layer in reversed(sorted(layer_to_nodes.items())):
        n = len(nodes_in_layer)
        distances = np.zeros(shape=(n, n))

        # Extract available X positions in the layer
        avail_xs = layers[layer]

        for node_idx, node in enumerate(nodes_in_layer):
            related_nodes = list(graph.predecessors(node))
            related_nodes += list(graph.successors(node))

            for x_idx, x in enumerate(avail_xs):
                sum_distances = 0
                for related_node in related_nodes:
                    if related_node in selected_positions:
                        r_x, r_y = selected_positions[related_node]
                        sum_distances += (x - r_x) ** 2 + (layer - r_y) ** 2
                    else:
                        # If node doesn't have a position yet
                        # we use the original position but we weight it down
                        r_x, r_y = orig_positions_dict[related_node]
                        sum_distances += ((x - r_x) ** 2 + (layer - r_y) ** 2) / 2
                distances[node_idx, x_idx] = sum_distances

        # Minimize the sum of squared distances
        # This might work poorly when there are interconnections in the layer
        rows, cols = scipy.optimize.linear_sum_assignment(distances)
        for row, col in zip(rows, cols):
            selected_positions[nodes_in_layer[row]] = (avail_xs[col], layer)

    new_sum = calc_sum_sq_distances(graph, selected_positions)

    if new_sum <= orig_sum:
        return selected_positions

    return orig_positions_dict


def draw_computation_graph(
    *,
    model: FunctionalModel | None = None,
    inputs: Iterable[SymbolicTensor] | SymbolicTensor | None = None,
    node_text_func: Callable[[SymbolicTensor], str] | None = None,
    edge_text_func: Callable[[nn.Module | None], str] | None = None,
    rotate: bool = True,
) -> None:
    """Plot graph of the computations, nodes being placeholder variables and nn.Modules being edges.

    This is not suitable for large graphs or large neural networks. This is a simple tool that
    was designed to demonstrate that Pytorch Functional creates sensible graphs that are nice
    to visualize.

    Parameters
    ----------
    model
        A FunctionalModel to be plotted. This or ``inputs`` must be provided.
    inputs
        Input in the graph of SymbolicTensor computations. This or ``model`` must be provided.
    node_text_func
        A function that returns text that will be written on Nodes.
    edge_text_func
        A function that returns text that will be written on Edges.
    rotate
        If True, text on edges will be rotated in the direction of the arrow.
    """
    try:
        import matplotlib.patches
        import matplotlib.pyplot as plt
        import networkx as nx
        import scipy
    except ImportError:
        print("To plot graphs, you need to install following packages:\nnetworkx\nmatplotlib\nscipy")
        return

    from .functional_model import FunctionalModel
    from .symbolic import SymbolicTensor

    if node_text_func is None:
        node_text_func = default_node_text

    if edge_text_func is None:
        edge_text_func = default_edge_text

    if model is not None:
        assert isinstance(model, FunctionalModel)
        inputs = model.inputs
    elif inputs is not None:
        if isinstance(inputs, SymbolicTensor):
            inputs = (inputs,)
        assert all(isinstance(x, SymbolicTensor) for x in inputs)
    else:
        raise KeyError("Provide either `model` or `inputs`!")

    graph = nx.DiGraph()
    assert isinstance(inputs, Iterable)
    to_visit = list(inputs)
    visited = set()

    node_labels = {}
    node_colors = {}
    edge_labels = {}

    INPUT_COLOR = (1.0, 0.3, 0.3)
    OUTPUT_COLOR = (0.3, 1.0, 0.3)
    OTHER_COLOR = (0.0, 1.0, 1.0)

    while to_visit:
        node = to_visit.pop()
        graph.add_node(id(node), node=node, depth=node.depth)

        if not node.parents:
            node_colors[id(node)] = INPUT_COLOR
        elif not node.children:
            node_colors[id(node)] = OUTPUT_COLOR
        else:
            node_colors[id(node)] = OTHER_COLOR

        node_labels[id(node)] = node_text_func(node)

        for parent in node.parents:
            graph.add_edge(id(parent), id(node), layer=node.layer)
            edge_labels[id(parent), id(node)] = edge_text_func(node.layer)

        for child in node.children:
            if id(child) not in visited:
                to_visit.append(child)
                visited.add(id(child))

    pos = nx.multipartite_layout(graph, subset_key="depth", align="horizontal", scale=-1)
    pos = fix_positions_in_multipartite_layout(graph, pos)

    nx.draw_networkx(
        graph,
        pos,
        with_labels=True,
        labels=node_labels,
        node_size=1000,
        arrowsize=20,
        node_shape="o",
        node_color=[node_colors[n] for n in graph.nodes],
        font_weight="bold",
        edge_color="grey",
    )

    nx.draw_networkx_edge_labels(
        graph,
        pos,
        edge_labels,
        label_pos=0.5,
        rotate=rotate,
        clip_on=False,
    )

    handles = [
        matplotlib.patches.Patch(color=INPUT_COLOR, label="Input node"),
        matplotlib.patches.Patch(color=OUTPUT_COLOR, label="Output node"),
        matplotlib.patches.Patch(color=OTHER_COLOR, label="Hidden node"),
    ]
    plt.legend(handles=handles)
