from typing import List
import sys
import warnings
import logging
import multiprocessing
import threading
import torch
import numpy as np
import scipy.spatial
import ase.db
import pandas as pd

try:
    import asap3
except ModuleNotFoundError:
    warnings.warn("Failed to import asap3 module for fast neighborlist")


class AseNeigborListWrapper:
    """
    Wrapper around ASE neighborlist to have the same interface as asap3 neighborlist

    """

    def __init__(self, cutoff, atoms):
        self.neighborlist = ase.neighborlist.NewPrimitiveNeighborList(
            cutoff, skin=0.0, self_interaction=False, bothways=True
        )
        self.neighborlist.build(
            atoms.get_pbc(), atoms.get_cell(), atoms.get_positions()
        )
        self.cutoff = cutoff
        self.atoms_positions = atoms.get_positions()
        self.atoms_cell = atoms.get_cell()

    def get_neighbors(self, i, cutoff):
        assert (
            cutoff == self.cutoff
        ), "Cutoff must be the same as used to initialise the neighborlist"

        indices, offsets = self.neighborlist.get_neighbors(i)

        rel_positions = (
            self.atoms_positions[indices]
            + offsets @ self.atoms_cell
            - self.atoms_positions[i][None]
        )

        dist2 = np.sum(np.square(rel_positions), axis=1)
        return indices, rel_positions, dist2


class TransformRowToGraph:
    def __init__(self, cutoff=5.0, targets="U0"):
        self.cutoff = cutoff

        if isinstance(targets, str):
            self.targets = [targets]
        else:
            self.targets = targets

    def __call__(self, row):
        atoms = row.toatoms()

        if np.any(atoms.get_pbc()):
            edges, edges_features = self.get_edges_neighborlist(atoms)
        else:
            edges, edges_features = self.get_edges_simple(atoms)

        # Extract targets if they exist
        targets = []
        for target in self.targets:
            if hasattr(row, target):
                t = getattr(row, target)
            elif hasattr(row, "data") and target in row.data:
                t = row.data[target]
            else:
                t = np.nan
            targets.append(t)
        targets = np.array(targets)

        default_type = torch.get_default_dtype()

        # pylint: disable=E1102
        graph_data = {
            "nodes": torch.tensor(atoms.get_atomic_numbers()),
            "num_nodes": torch.tensor(len(atoms.get_atomic_numbers())),
            "edges": torch.tensor(edges),
            "edges_features": torch.tensor(edges_features, dtype=default_type),
            "num_edges": torch.tensor(edges.shape[0]),
            "targets": torch.tensor(targets, dtype=default_type),
        }

        return graph_data
    
    
class TransformAtomsObjectToGraph:
    def __init__(self, cutoff=5.0):
        self.cutoff = cutoff

    def __call__(self, atoms, agent_num, A, B):

        if np.any(atoms.get_pbc()):
            edges, edges_features, rel_positions = self.get_edges_neighborlist(atoms)
        else:
            edges, edges_features = self.get_edges_simple(atoms)

        # targets = np.array(returns)

        default_type = torch.get_default_dtype()
        
        # Find internal coordinates
        edges = torch.LongTensor(edges)
        agent_num = torch.tensor(agent_num)
        # All positions are relative to receiving atom, same goes for A and B
        pos = torch.tensor(rel_positions)
        
        # If agent position coincides with initial agent position, A is zero,
        # meaning that azimut becomes undefined.
        if (np.linalg.norm(A) == 0) or (np.linalg.norm(np.cross(A, B)) == 0):
            A = A + 2*np.random.rand(3) - 1

        #np.isnan(A).any()

        A = torch.tensor(A)
        B = torch.tensor(B)
        
        # Let agent atom be origin of spherical internal coordinate system
        # (this is achived by using the relative positions above)
        
        # Find edges from neighbors to agent
        edges_neighbor_id = edges[: , 1] == agent_num
        edges_neighbor = edges[edges_neighbor_id]

        # Find index of neighboring node states
        node_id_neighbor = edges_neighbor[:, 0]

        # Reduce position tensor to consider only neighbor atoms
        pos = pos[edges_neighbor_id]
        
        # Find angles between position vectors and B
        alpha = torch.arccos(np.dot(pos, B)/(torch.linalg.norm(pos, axis=1)*torch.linalg.norm(B)))

        # Find unit vector normal to A and B as cross product
        AB_cross = torch.cross(A, B)
        n = AB_cross/torch.linalg.norm(AB_cross)

        # Find unit vector normal to B and n as cross product
        Bn_cross = torch.cross(B, n)
        Bn_cross = Bn_cross/torch.linalg.norm(Bn_cross)

        # Bn_cross and n spans the perpendicular subspace corresponding to alpha=pi/2, i.e. they form a basis for the subspace.
        # Project A onto the perpendicular subspace (https://www.cliffsnotes.com/study-guides/algebra/linear-algebra/real-euclidean-vector-spaces/projection-onto-a-subspace)
        A_perp = (np.dot(A, Bn_cross)/np.dot(Bn_cross, Bn_cross)) * Bn_cross + (np.dot(A, n)/np.dot(n, n)) * n

        # Likewise project positions onto the perpendicular subspace
        p_perp = (np.dot(pos, Bn_cross)/torch.dot(Bn_cross, Bn_cross)).unsqueeze(1) * Bn_cross + (np.dot(pos, n)/torch.dot(n, n)).unsqueeze(1) * n

        # The angle between these two projections are then caluculated as
        dihedral_abs = torch.arccos(np.dot(p_perp, A_perp)/(torch.linalg.norm(p_perp, axis=1)*torch.linalg.norm(A_perp)))
        kappa = torch.sign(torch.tensor(np.dot(pos, n)))
        dihedral = kappa * dihedral_abs

        # Calculate absolute distance between neighbors and agent
        r = torch.linalg.norm(pos, axis=1)

        # Collect coordinates by concatenation 
        internal_coordinates_neighbors = torch.cat(
            (torch.unsqueeze(alpha, dim=1), torch.unsqueeze(dihedral, dim=1), torch.unsqueeze(r, dim=1)), dim=1
        )

        # pylint: disable=E1102
        graph_data = {
            "nodes": torch.tensor(atoms.get_atomic_numbers()),
            "num_nodes": torch.tensor(len(atoms.get_atomic_numbers())),
            "edges": edges,
            "edges_features": torch.tensor(edges_features, dtype=default_type),
            "num_edges": torch.tensor(edges.shape[0]),
            "node_id_neighbors": node_id_neighbor,
            "internal_coordinates_neighbors": internal_coordinates_neighbors,
            "num_neighbors": torch.tensor(internal_coordinates_neighbors.shape[0])
            #"rel_positions": rel_positions
            #"targets": torch.tensor(targets, dtype=default_type),
            #"pos": torch.tensor(atoms.get_positions()),
            #"agent_num": torch.tensor(agent_num),
            #"A": ,
            #"B": torch.tensor(B)
        }
        
        return graph_data


    def get_edges_simple(self, atoms):
        # Compute distance matrix
        pos = atoms.get_positions()
        dist_mat = scipy.spatial.distance_matrix(pos, pos)

        # Build array with edges and edge features (distances)
        valid_indices_bool = dist_mat < self.cutoff
        np.fill_diagonal(valid_indices_bool, False)  # Remove self-loops
        edges = np.argwhere(valid_indices_bool)  # num_edges x 2
        edges_features = dist_mat[valid_indices_bool]  # num_edges
        edges_features = np.expand_dims(edges_features, 1)  # num_edges, 1

        return edges, edges_features

    def get_edges_neighborlist(self, atoms):
        edges = []
        edges_features = []
        rel_positions = []

        # Compute neighborlist
        if (
            np.any(atoms.get_cell().lengths() <= 0.0001)
            or (
                np.any(atoms.get_pbc())
                and np.any(atoms.get_cell().lengths() < self.cutoff)
            )
            or ("asap3" not in sys.modules)
        ):
            neighborlist = AseNeigborListWrapper(self.cutoff, atoms)
        else:
            neighborlist = asap3.FullNeighborList(self.cutoff, atoms)

        for i in range(len(atoms)):
            neigh_idx, rel_pos, neigh_dist2 = neighborlist.get_neighbors(i, self.cutoff)
            neigh_dist = np.sqrt(neigh_dist2)

            self_index = np.ones_like(neigh_idx) * i
            this_edges = np.stack((neigh_idx, self_index), axis=1)

            edges.append(this_edges)
            edges_features.append(neigh_dist)
            rel_positions.append(rel_pos)

        return np.concatenate(edges), np.expand_dims(np.concatenate(edges_features), 1), np.concatenate(rel_positions)


class AseDbData(torch.utils.data.Dataset):
    def __init__(self, asedb_path, transformer, **kwargs):
        super().__init__(**kwargs)

        self.asedb_path = asedb_path
        self.asedb_connection = ase.db.connect(asedb_path)
        self.transformer = transformer

    def __len__(self):
        return len(self.asedb_connection)

    def __getitem__(self, key):
        # Note that ASE databases are 1-indexed
        try:
            return self.transformer(self.asedb_connection[key + 1])
        except KeyError:
            raise IndexError("index out of range")


class QM9MetaGGAData(torch.utils.data.Dataset):
    """"""

    def __init__(self, qm9asedb_path, metaggaqm9csv_path, cutoff, **kwargs):
        super().__init__(**kwargs)

        self.asedb_connection = ase.db.connect(qm9asedb_path)
        self.metagga_df = pd.read_csv(metaggaqm9csv_path, index_col="index")
        self.metagga_df.drop(columns=["SOGGA", "SOGGA11"], inplace=True)
        self.transformer = TransformRowToGraph(cutoff=cutoff, targets=[])

    def __len__(self):
        return len(self.asedb_connection)

    def __getitem__(self, key):
        # Note that ASE databases are 1-indexed
        key = key + 1
        try:
            item = self.transformer(self.asedb_connection[key])
            targets = self.metagga_df.loc[key].values
            item["targets"] = torch.tensor(targets, dtype=torch.float32)
            return item
        except KeyError:
            raise IndexError("index out of range")


class BufferData(torch.utils.data.Dataset):
    """
    Wrapper for a dataset. Loads all data into memory.
    """

    def __init__(self, dataset, **kwargs):
        super().__init__(**kwargs)

        self.data_objects = [dataset[i] for i in range(len(dataset))]

    def __len__(self):
        return len(self.data_objects)

    def __getitem__(self, index):
        return self.data_objects[index]


def rotating_pool_worker(dataset, rng, queue):
    while True:
        for index in rng.permutation(len(dataset)):
            queue.put(dataset[index])


def transfer_thread(queue: multiprocessing.Queue, datalist: list):
    while True:
        for index in range(len(datalist)):
            datalist[index] = queue.get()


class RotatingPoolData(torch.utils.data.Dataset):
    """
    Wrapper for a dataset that continously loads data into a smaller pool.
    The data loading is performed in a separate process and is assumed to be IO bound.
    """

    def __init__(self, dataset, pool_size, **kwargs):
        super().__init__(**kwargs)
        self.pool_size = pool_size
        self.parent_data = dataset
        self.rng = np.random.default_rng()
        logging.debug("Filling rotating data pool of size %d" % pool_size)
        self.data_pool = [
            self.parent_data[i]
            for i in self.rng.integers(
                0, high=len(self.parent_data), size=self.pool_size, endpoint=False
            )
        ]
        self.loader_queue = multiprocessing.Queue(2)

        # Start loaders
        self.loader_process = multiprocessing.Process(
            target=rotating_pool_worker,
            args=(self.parent_data, self.rng, self.loader_queue),
        )
        self.transfer_thread = threading.Thread(
            target=transfer_thread, args=(self.loader_queue, self.data_pool)
        )
        self.loader_process.start()
        self.transfer_thread.start()

    def __len__(self):
        return self.pool_size

    def __getitem__(self, index):
        return self.data_pool[index]


def pad_and_stack(tensors: List[torch.Tensor]):
    """ Pad list of tensors if tensors are arrays and stack if they are scalars """
    if tensors[0].shape:
        return torch.nn.utils.rnn.pad_sequence(
            tensors, batch_first=True, padding_value=0
        )
    return torch.stack(tensors)


def collate_atomsdata(graphs: List[dict], pin_memory=True):
    # Convert from "list of dicts" to "dict of lists"
    dict_of_lists = {k: [dic[k] for dic in graphs] for k in graphs[0]}
    # Convert each list of tensors to single tensor with pad and stack
    if pin_memory:
        pin = lambda x: x.pin_memory()
    else:
        pin = lambda x: x

    collated = {k: pin(pad_and_stack(dict_of_lists[k])) for k in dict_of_lists}
    return collated

