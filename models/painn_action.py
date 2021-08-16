import math

import torch
from torch import nn

from models import layer_painn as layer


class PainnModel(nn.Module):
    """PainnModel with forces."""

    def __init__(
        self,
        num_interactions,
        hidden_state_size,
        cutoff,
        agent_num,
        action_space_size=6,
        target_mean=[0.0],
        target_stddev=[1.0],
        normalize_atomwise=True,
        direct_force_output=False,
        **kwargs,
    ):
        """
        Args:
            num_interactions (int): Number of interaction layers
            hidden_state_size (int): Size of hidden node states
            cutoff (float): Atomic interaction cutoff distance [Å]
            target_mean ([float]): Target normalisation constant
            target_stddev ([float]): Target normalisation constant
            normalize_atomwise (bool): Use atomwise normalisation
            direct_force_output (bool): Compute forces directly instead of using gradient
        """
        super().__init__(**kwargs)
        self.num_interactions = num_interactions
        self.hidden_state_size = hidden_state_size
        self.cutoff = cutoff
        self.distance_embedding_size = 20

        num_embeddings = 119  # atomic numbers + 1
        edge_size = self.distance_embedding_size

        # Setup atom embeddings
        self.atom_embeddings = nn.Embedding(num_embeddings, hidden_state_size)

        # Setup interaction networks
        self.interactions = nn.ModuleList(
            [
                layer.PaiNNInteraction(hidden_state_size, edge_size, self.cutoff)
                for _ in range(num_interactions)
            ]
        )
        self.scalar_vector_update = nn.ModuleList(
            [layer.PaiNNUpdate(hidden_state_size) for _ in range(num_interactions)]
        )

        # Setup readout function
        self.readout_mlp = nn.Sequential(
            nn.Linear(hidden_state_size, hidden_state_size),
            nn.SiLU(),
            nn.Linear(hidden_state_size, 1),
        )

        self.readout_surfrider = layer.SurfRiderPaiNNReadout(hidden_state_size, action_space_size=6)

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

        # Direct force output
        self.direct_force_output = direct_force_output
        if self.direct_force_output:
            self.force_readout_linear = nn.Linear(hidden_state_size, 1, bias=False)

    def forward(self, input_dict, compute_forces=True, compute_stress=True):
        """
        Args:
            input_dict (dict): Input dictionary of tensors with keys: nodes,
                               nodes_xyz, num_nodes, edges, edges_displacement, cell,
                               num_edges, targets
        Returns:
            result_dict (dict): Result dictionary with keys:
                                energy, forces, stress
                                Forces and stress are only included if requested (default).
        """
        if compute_forces and not self.direct_force_output:
            input_dict["nodes_xyz"].requires_grad_()
        if compute_stress:
            # Create displacement matrix of zeros and transform cell and atom positions
            displacement = torch.zeros_like(input_dict["cell"], requires_grad=True)
            input_dict["cell"] = input_dict["cell"] + torch.matmul(
                input_dict["cell"], displacement
            )
            input_dict["nodes_xyz"] = input_dict["nodes_xyz"] + torch.matmul(
                input_dict["nodes_xyz"], displacement
            )

        # Unpad and concatenate edges and features into batch (0th) dimension
        edges_displacement = layer.unpad_and_cat(
            input_dict["edges_displacement"], input_dict["num_edges"]
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
        #print("edge_offset_first")
        #print(edge_offset)
        # tensor([ 0, 28, 56, 84])

        edge_offset = edge_offset[:, None, None]
        
        #print("edge_offset_second")
        #print(edge_offset)
        #tensor([[[ 0]],
        #        [[28]],
        #        [[56]],
        #        [[84]]])

        #print("input dict edges")
        # print(input_dict["edges"]) 

        edges = input_dict["edges"] + edge_offset
        #print("edges first")
        #print(edges)
        edges = layer.unpad_and_cat(edges, input_dict["num_edges"])
        #print("edges second")
        #print(edges)

        # Unpad and concatenate all nodes into batch (0th) dimension
        nodes_xyz = layer.unpad_and_cat(
            input_dict["nodes_xyz"], input_dict["num_nodes"]
        )
        nodes_scalar = layer.unpad_and_cat(input_dict["nodes"], input_dict["num_nodes"])
        nodes_scalar = self.atom_embeddings(nodes_scalar)
        nodes_vector = torch.zeros(
            (nodes_scalar.shape[0], 3, self.hidden_state_size),
            dtype=nodes_scalar.dtype,
            device=nodes_scalar.device,
        )


        #positions = nodes_xyz
        #cells = input_dict["cell"]
        #splits = input_dict["num_edges"]


        # Normalize goal alignment vectors
        B_norm = input_dict["B"].pow(2).sum(dim=1).sqrt()
        # print("input_dict B")
        # print(input_dict["B"].shape)

        # print("B_norm")
        # print(B_norm.shape)
        # print(B_norm)

        input_dict["B"] = input_dict["B"] / B_norm.unsqueeze(dim=-1)
        n_norm = input_dict["n"].pow(2).sum(dim=1).sqrt()
        input_dict["n"] = input_dict["n"] / n_norm.unsqueeze(dim=-1)

        # Compute edge distances
        edges_distance, edges_diff = layer.calc_distance(
            nodes_xyz,
            input_dict["cell"],
            edges,
            edges_displacement,
            input_dict["num_edges"],
            input_dict["B"],
            input_dict["n"],
            return_diff=True,
        )

        # Rotate edges_diff vectors
        # edges_diff = layer.goal_align_edge_diffs(
        #     edges_diff,
        #     input_dict["B"],
        #     input_dict["n"]
        # )

        # Expand edge features in Gaussian basis
        edge_state = layer.sinc_expansion(
            edges_distance, [(self.distance_embedding_size, self.cutoff)]
        )

        # Apply interaction layers
        for int_layer, update_layer in zip(
            self.interactions, self.scalar_vector_update
        ):
            nodes_scalar, nodes_vector = int_layer(
                nodes_scalar,
                nodes_vector,
                edge_state,
                edges_diff,
                edges_distance,
                edges,
            )
            nodes_scalar, nodes_vector = update_layer(nodes_scalar, nodes_vector)




        # Find "cumsum vector" of agent atoms
        #print("input_dict agent_num")
        #print(input_dict["agent_num"].shape)
        #print(input_dict["agent_num"])

        agent_num = input_dict["agent_num"]

        #print("agent_num")
        #print(agent_num)

        agent_index_batch = agent_num + edge_offset.squeeze(-1).squeeze(-1)

        #print("agent_index_batch")
        #print(agent_index_batch.shape)
        #print(agent_index_batch)


        #print("edge_offset")
        #print(edge_offset.shape)
        #print(edge_offset)

        # agent_index_batch = agent_index_batch + .


        #print("nodes_scalar")
        #print(nodes_scalar.shape)
        # print(nodes_scalar)


        agent_nodes_scalar = nodes_scalar[agent_index_batch, :]


        #print("agent_nodes_scalar")
        #print(agent_nodes_scalar.shape)
        #print(agent_nodes_scalar)

        agent_nodes_vector = nodes_vector[agent_index_batch, :, :]


        # Pick out all "nodes_scalar" and "nodes_vector" for agent atoms (along batch dimension)

        # Apply readout function
        Q_value = self.readout_surfrider(agent_nodes_scalar, agent_nodes_vector)

        # print(Q_value.shape)

        return Q_value
