# Copyright (c) 2022 Szymon Mikler

from __future__ import annotations

from typing import List, Tuple

from pytorch_symbolic.symbolic_tensor import SymbolicTensor


def generate_forward_basic(
    inputs: List[SymbolicTensor] | Tuple[SymbolicTensor, ...],
    outputs: List[SymbolicTensor] | Tuple[SymbolicTensor, ...],
    execution_order: List[SymbolicTensor] | Tuple[SymbolicTensor, ...],
):
    str_length = len(str(max(len(inputs), len(outputs), len(execution_order))))
    node_to_name = {}
    for idx, node in enumerate(inputs):
        node_to_name[node] = f"i{str(idx).zfill(str_length)}"
    for idx, node in enumerate(execution_order):
        node_to_name[node] = f"h{str(idx).zfill(str_length)}"
    for idx, node in enumerate(outputs):
        node_to_name[node] = f"o{str(idx).zfill(str_length)}"

    input_names = [node_to_name[node] for node in inputs]
    forward_definition = "def _generated_forward(self," + ",".join(input_names) + "):"
    code_lines = [forward_definition]

    TAB = " " * 4
    code_lines.append(TAB + "l = self._execution_order_layers")

    for exec_id, node in enumerate(execution_order):
        input_names = [node_to_name[node] for node in node.parents]
        output_name = node_to_name[node]
        code_line = TAB + output_name + f" = l[{exec_id}](" + ",".join(input_names) + ")"
        code_lines.append(code_line)

    return_line = TAB + "return " + ",".join(node_to_name[node] for node in outputs)
    code_lines.append(return_line)
    generated_forward = "\n".join(code_lines) + "\n"
    return generated_forward


def generate_forward_with_loops(
    inputs: List[SymbolicTensor] | Tuple[SymbolicTensor, ...],
    outputs: List[SymbolicTensor] | Tuple[SymbolicTensor, ...],
    execution_order: List[SymbolicTensor] | Tuple[SymbolicTensor, ...],
    min_loop_length: int | float = float("inf"),
):
    str_length = len(str(max(len(inputs), len(outputs), len(execution_order))))
    node_to_name = {}
    for idx, node in enumerate(inputs):
        node_to_name[node] = f"i{str(idx).zfill(str_length)}"
    for idx, node in enumerate(execution_order):
        node_to_name[node] = f"h{str(idx).zfill(str_length)}"
    for idx, node in enumerate(outputs):
        node_to_name[node] = f"o{str(idx).zfill(str_length)}"

    input_names = [node_to_name[node] for node in inputs]
    forward_definition = "def _generated_forward(self," + ",".join(input_names) + "):"
    code_lines = [forward_definition]

    TAB = " " * 4
    code_lines.append(TAB + "l = self._execution_order_layers")

    nodes_looped_over = set()
    all_nodes_used = {*inputs, *execution_order}
    # All parents must be in the graph. Otherwise, forward is impossible.
    parents = {node: node.parents for node in execution_order}
    # We only count children in the graph. Thus the intersection.
    children = {node: list(all_nodes_used.intersection(node.children)) for node in execution_order}

    for exec_id, node in enumerate(execution_order):
        if node in nodes_looped_over:
            continue

        input_names = [node_to_name[node] for node in node.parents]
        sequence = [node]
        last_node = node
        while (
            len(children[last_node]) == 1
            # this should never be false, but just in case we make sure the child is next in execution order
            and children[last_node][0] is execution_order[exec_id + len(sequence)]
            and len(parents[last_node]) == 1
            and len(parents[children[last_node][0]]) == 1
        ):
            last_node = children[last_node][0]
            sequence.append(last_node)

        if len(sequence) >= min_loop_length:
            output_name = node_to_name[sequence[-1]]
            code_lines.append(TAB + f"{output_name} = {input_names[0]}")
            code_lines.append(TAB + f"for layer in l[{exec_id}:{exec_id + len(sequence)}]:")
            code_lines.append(TAB + TAB + f"{output_name} = layer({output_name})")
            nodes_looped_over.update(sequence)
        else:
            output_name = node_to_name[node]
            code_line = TAB + output_name + f" = l[{exec_id}](" + ",".join(input_names) + ")"
            code_lines.append(code_line)

    code_lines.append(TAB + "return " + ",".join(node_to_name[node] for node in outputs))
    generated_forward = "\n".join(code_lines) + "\n"
    return generated_forward
