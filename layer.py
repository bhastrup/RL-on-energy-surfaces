import itertools
from typing import Tuple, List

import numpy as np
import torch
from torch import nn


def shifted_softplus(x):
    """
    Compute shifted soft-plus activation function.
    .. math::
       y = \ln\left(1 + e^{-x}\right) - \ln(2)

    Args:
        x (torch.Tensor): input tensor.

    Returns:
        torch.Tensor: shifted soft-plus of input.

    """
    return nn.functional.softplus(x) - np.log(2.0)


class ShiftedSoftplus(nn.Module):
    def forward(self, x):
        return shifted_softplus(x)


def unpad_and_cat(stacked_seq: torch.Tensor, seq_len: torch.Tensor):
    """
    Unpad and concatenate by removing batch dimension

    Args:
        stacked_seq: (batch_size, max_length, *) Tensor
        seq_len: (batch_size) Tensor with length of each sequence

    Returns:
        (prod(seq_len), *) Tensor

    """
    unstacked = stacked_seq.unbind(0)
    unpadded = [
        torch.narrow(t, 0, 0, l) for (t, l) in zip(unstacked, seq_len.unbind(0))
    ]
    return torch.cat(unpadded, dim=0)


def sum_splits(values: torch.Tensor, splits: torch.Tensor):
    """
    Sum across dimension 0 of the tensor `values` in chunks
    defined in `splits`

    Args:
        values: Tensor of shape (`prod(splits)`, *)
        splits: 1-dimensional tensor with size of each chunk

    Returns:
        Tensor of shape (`splits.shape[0]`, *)

    """
    # prepare an index vector for summation
    ind = torch.zeros(splits.sum(), dtype=splits.dtype, device=splits.device)
    ind[torch.cumsum(splits, dim=0)[:-1]] = 1
    ind = torch.cumsum(ind, dim=0)
    # prepare the output
    sum_y = torch.zeros(
        splits.shape + values.shape[1:], dtype=values.dtype, device=values.device
    )
    # do the actual summation
    sum_y.index_add_(0, ind, values)
    return sum_y


def gaussian_expansion(input_x: torch.Tensor, expand_params: List[Tuple]):
    """
    Expand each feature in a number of Gaussian basis function.
    Expand_params is a list of length input_x.shape[1]

    Args:
        input_x: (num_edges, num_features) tensor
        expand_params: list of None or (start, step, stop) tuples

    Returns:
        (num_edges, ``ceil((stop - start)/step)``) tensor

    """
    feat_list = torch.unbind(input_x, dim=1)
    expanded_list = []
    for step_tuple, feat in itertools.zip_longest(expand_params, feat_list):
        assert feat is not None, "Too many expansion parameters given"
        if step_tuple:
            start, step, stop = step_tuple
            feat_expanded = torch.unsqueeze(feat, dim=1)
            sigma = step
            basis_mu = torch.arange(
                start, stop, step, device=input_x.device, dtype=input_x.dtype
            )
            expanded_list.append(
                torch.exp(-((feat_expanded - basis_mu) ** 2) / (2.0 * sigma ** 2))
            )
        else:
            expanded_list.append(torch.unsqueeze(feat, 1))
    return torch.cat(expanded_list, dim=1)


class SchnetMessageFunction(nn.Module):
    """Message function"""

    def __init__(self, node_size, edge_size):
        """
        Args:
            node_size (int): Size of node state
            edge_size (int): Size of edge state
        """
        super().__init__()
        self.msg_function_edge = nn.Sequential(
            nn.Linear(edge_size, node_size),
            ShiftedSoftplus(),
            nn.Linear(node_size, node_size),
        )
        self.msg_function_node = nn.Sequential(
            nn.Linear(node_size, node_size),
            ShiftedSoftplus(),
            nn.Linear(node_size, node_size),
        )

    def forward(self, node_state, edge_state):
        """
        Args:
            node_state (tensor): State of each sender node (num_edges, node_size)
            edge_state (tensor): Edge states (num_edges, edge_size)

        Returns:
            (num_edges, node_size) tensor
        """
        gates = self.msg_function_edge(edge_state)
        nodes = self.msg_function_node(node_state)
        return nodes * gates


class Interaction(nn.Module):
    """Interaction network"""

    def __init__(self, node_size, edge_size):
        """
        Args:
            node_size (int): Size of node state
            edge_size (int): Size of edge state
        """
        super().__init__()

        self.message_function = SchnetMessageFunction(node_size, edge_size)

        self.state_transition_function = nn.Sequential(
            nn.Linear(node_size, node_size),
            ShiftedSoftplus(),
            nn.Linear(node_size, node_size),
        )

    def forward(self, node_state, edges, edge_state):
        """
        Args:
            node_state (tensor): Node states (num_nodes, node_size)
            edges (tensor): Directed edges with node indices (num_edges, 2)
            edge_state (tensor): Edge states (num_edges, edge_size)

        Returns:
            (num_nodes, node_size) tensor
        """
        # Compute all messages
        nodes = node_state[edges[:, 0]]  # Only include sender in messages
        messages = self.message_function(nodes, edge_state)

        # Sum messages
        message_sum = torch.zeros_like(node_state)
        message_sum.index_add_(0, edges[:, 1], messages)

        # State transition
        new_state = node_state + self.state_transition_function(message_sum)

        return new_state


class EdgeUpdate(nn.Module):
    """Edge update network"""

    def __init__(self, edge_size, node_size):
        """
        Args:
            edge_size (int): Size of edge state
            node_size (int): Size of node state
        """
        super().__init__()

        self.node_size = node_size
        self.edge_update_mlp = nn.Sequential(
            nn.Linear(2 * node_size + edge_size, 2 * edge_size),
            ShiftedSoftplus(),
            nn.Linear(2 * edge_size, edge_size),
        )

    def forward(self, edge_state, edges, node_state):
        """
        Args:
            edge_state (tensor): Edge states (num_edges, edge_size)
            edges (tensor): Directed edges with node indices (num_edges, 2)
            node_state (tensor): Node states (num_nodes, node_size)

        Returns:
            (num_nodes, node_size) tensor
        """
        combined = torch.cat(
            (node_state[edges].view(-1, 2 * self.node_size), edge_state), axis=1
        )
        return self.edge_update_mlp(combined)
