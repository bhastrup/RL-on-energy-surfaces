import math

import torch
from torch import nn

from models import layer

class SchnetModel(nn.Module):
    """SchNet model with optional edge updates."""

    def __init__(
        self,
        num_interactions,
        hidden_state_size,
        cutoff,
        update_edges=False,
        target_mean=[0.0],
        target_stddev=[1.0],
        normalize_atomwise=True,
        **kwargs,
    ):
        """
        Args:
            num_interactions (int): Number of interaction layers
            hidden_state_size (int): Size of hidden node states
            cutoff (float): Atomic interaction cutoff distance [Ãƒâ€¦]
            update_edges (bool): Enable edge updates
            target_mean ([float]): Target normalisation constant
            target_stddev ([float]): Target normalisation constant
            normalize_atomwise (bool): Use atomwise normalisation
        """
        super().__init__(**kwargs)
        self.num_interactions = num_interactions
        self.hidden_state_size = hidden_state_size
        self.cutoff = cutoff
        self.gaussian_expansion_step = 0.1

        num_embeddings = 119 # atomic numbers + 1
        edge_size = int(math.ceil(self.cutoff / self.gaussian_expansion_step))

        # Setup atom embeddings
        self.atom_embeddings = nn.Embedding(num_embeddings, hidden_state_size)

        # Setup interaction networks
        self.interactions = nn.ModuleList(
            [
                layer.Interaction(hidden_state_size, edge_size)
                for _ in range(num_interactions)
            ]
        )

        if update_edges:
            self.edge_updates = nn.ModuleList(
                [
                    layer.EdgeUpdate(edge_size, hidden_state_size)
                    for _ in range(num_interactions)
                ]
            )
        else:
            self.edge_updates = [lambda e_state, e, n: e_state] * num_interactions

        # Setup readout function - C/2
        self.readout_mlp_c_half = nn.Sequential(
            nn.Linear(hidden_state_size + 3, math.ceil(hidden_state_size/2)),
            layer.ShiftedSoftplus()
        )

        # Setup readout function
        self.readout_mlp = nn.Linear(math.ceil(hidden_state_size/2) + 2, 6)

        # Normalisation constants
        self.normalize_atomwise = torch.nn.Parameter(
            torch.tensor(normalize_atomwise), requires_grad=False
        )
        self.normalize_stddev = torch.nn.Parameter(
            torch.as_tensor(target_stddev), requires_grad=False
        )
        self.normalize_mean = torch.nn.Parameter(
            torch.as_tensor(target_mean), requires_grad=False
        )

    def forward(self, input_dict):
        """
        Args:
            input_dict (dict): Input dictionary of tensors with keys: nodes,
                               num_nodes, edges, edges_features, num_edges,
                               targets
        """



        # Unpad and concatenate edges and features into batch (0th) dimension
        edges_features = layer.unpad_and_cat(
            input_dict["edges_features"], input_dict["num_edges"]
        )
        edge_offset = torch.cumsum(
            torch.cat(
                (
                    torch.tensor([0], device=input_dict["num_nodes"].device),
                    input_dict["num_nodes"][:-1],
                )
            ),
            dim=0,
        )
        edge_offset = edge_offset[:, None, None]
        edges = input_dict["edges"] + edge_offset
        edges = layer.unpad_and_cat(edges, input_dict["num_edges"])

        # Unpad and concatenate all nodes into batch (0th) dimension
        nodes = layer.unpad_and_cat(input_dict["nodes"], input_dict["num_nodes"])
        nodes = self.atom_embeddings(nodes)

        # Expand edge features in Gaussian basis
        edge_state = layer.gaussian_expansion(
            edges_features, [(0.0, self.gaussian_expansion_step, self.cutoff)]
        )

        # Apply interaction layers
        for edge_layer, int_layer in zip(self.edge_updates, self.interactions):
            edge_state = edge_layer(edge_state, edges, nodes)
            nodes = int_layer(nodes, edges, edge_state)

        # For RL transition paths - build new output from edge_states or nodes
        # concat to edge_state


        # Find neighbor subset
        nodes_state_neighbor = nodes[input_dict["node_id_neighbors"]]

        # Concatenate internal coordinates to neighboring node states
        node_state_concat = torch.cat(
            (nodes_state_neighbor, input_dict["internal_coordinates_neighbors"]), axis=2
        )

        node_state_concat = node_state_concat.view(-1, self.hidden_state_size + 3).float()

	# node_id_neighbors.shape = 6, num_n
        # nodes_state_neighbor.shape = 6, 6, 64
	# internal_coordiates_neighbors.shape = 6, 6, 2
	# edges.shape = 2232, 2
        # nodes.shape = 150, 64
	# num_edges.shape = 6 (from perturbed)


        # Apply RL readout function
        #nodes = self.readout_mlp(node_state_concat)

        nodes_C_half = self.readout_mlp_c_half(node_state_concat)

	# Remove all invalid outputs correponding to padded edges/nodes
        nodes_C_half = layer.remove_pad_outputs(nodes_C_half, input_dict["num_neighbors"])

        # Obtain graph level output
        nodes_C_half_sum = layer.sum_splits(nodes_C_half, input_dict["num_neighbors"])

        #print("nodes_C_half_sum"); print(nodes_C_half_sum); print(nodes_C_half_sum.shape)
        #print(input_dict["B_dist"]); print(input_dict["B_dist"]); print(input_dict["B_dist"].shape)
        nodes_C_half_sum_cat = torch.cat(
            (nodes_C_half_sum, torch.unsqueeze(input_dict["A_dist"], 1).float(), torch.unsqueeze(input_dict["B_dist"], 1).float()), axis=1
        )
	
        graph_output = self.readout_mlp(nodes_C_half_sum_cat)

        ## Apply (de-)normalization
        #normalizer = (1.0 / self.normalize_stddev).unsqueeze(0)
        #graph_output = graph_output * normalizer
        #mean_shift = self.normalize_mean.unsqueeze(0)
        #if self.normalize_atomwise:
        #    mean_shift = mean_shift * input_dict["num_nodes"].unsqueeze(1)
        #graph_output = graph_output + mean_shift

        return graph_output
