#  Copyright (c) 2022 Szymon Mikler


def get_parameter_count(model):
    cnt = 0
    for param in model.parameters():
        cnt += param.shape.numel()
    return cnt


def get_parameter_shapes(model):
    shapes = []
    for param in model.parameters():
        shapes.append(tuple(param.shape))
    return shapes


def model_similar(a, b):
    if get_parameter_count(a) != get_parameter_count(b):
        return False

    if sorted(get_parameter_shapes(a)) != sorted(get_parameter_shapes(b)):
        return False
    return True


def hash_tensor(tensor):
    return hash(" ".join([str(value) for value in tensor.flatten()]))


def model_hashes_identical(a, b):
    hashes_a = [hash_tensor(p) for p in a.parameters()]
    hashes_b = [hash_tensor(p) for p in b.parameters()]
    return set(hashes_a) == set(hashes_b)


def plot_computation_graph(
    *,
    model=None,
    inputs=None,
    node_label_func=None,
    edge_label_func=None,
    rotate=True,
):
    try:
        import matplotlib.patches
        import matplotlib.pyplot as plt
        import networkx as nx
    except ImportError:
        print("To plot graphs, you need to install following packages:\nnetworkx\nmatplotlib")
        return

    if node_label_func is None:

        def node_label_func(plh):
            return str(plh.shape)

    if edge_label_func is None:

        def edge_label_func(layer):
            return str(layer)

    from pytorch_functional.functional_model import FunctionalModel, Placeholder

    if model is not None:
        assert isinstance(model, FunctionalModel)
        inputs = model.inputs
    elif inputs is not None:
        if isinstance(inputs, Placeholder):
            inputs = (inputs,)
        assert all(isinstance(x, Placeholder) for x in inputs)
    else:
        raise KeyError("Provide either `model` or `inputs`!")

    graph = nx.DiGraph()
    visited = set()
    to_visit = list(inputs)

    node_labels = {}
    node_colors = {}
    edge_labels = {}

    INPUT_COLOR = (1, 0.3, 0.3)
    OUTPUT_COLOR = (0.3, 1, 0.3)
    OTHER_COLOR = (0, 1, 1)

    while to_visit:
        node = to_visit.pop()
        graph.add_node(id(node), node=node, depth=node.depth)

        if not node.parents:
            node_colors[id(node)] = INPUT_COLOR
        elif not node.children:
            node_colors[id(node)] = OUTPUT_COLOR
        else:
            node_colors[id(node)] = OTHER_COLOR

        node_labels[id(node)] = node_label_func(node)

        for parent in node.parents:
            graph.add_edge(id(parent), id(node), layer=node.layer)
            edge_labels[id(parent), id(node)] = edge_label_func(node.layer)

        for child in node.children:
            if id(child) not in visited:
                to_visit.append(child)
                visited.add(id(child))

    pos = nx.multipartite_layout(graph, subset_key="depth", align="horizontal", scale=-1)

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
    return graph
