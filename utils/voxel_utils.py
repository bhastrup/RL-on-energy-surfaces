import time
import numpy as np
import torch

# if gpu is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_3d_grid(unit_cell_len, n_grid_points):
    """
        Generates a 3d grid. Crappy implementation but is only called once.
    """
    # Create array of voxel center coordinates, grid_3d.
    x_grid = np.linspace(0, unit_cell_len, n_grid_points + 1)  # grid points
    d = x_grid[:-1]+(x_grid[1]-x_grid[0])/2    # grid centers
    grid_3d = np.zeros((n_grid_points**3, 3))  # voxel center coordinates

    counter = 0
    for i in range(n_grid_points):
        for j in range(n_grid_points):
            for k in range(n_grid_points):
                grid_3d[counter, :] = np.array([d[i], d[j], d[k]])
                counter += 1

    return grid_3d



from typing import List
import sys
import numpy as np
import scipy.spatial

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


class TransformAtomsObjectToVoxelBox:
    def __init__(self, cutoff=5.0, agent_atom=0, n_grid=10, sigma=0.3, device=torch.device("cuda")):

        self.cutoff = cutoff
        self.agent_num = agent_atom
        self.n_grid = n_grid
        self.box_length = cutoff * 2
        self.device = device
        self.sigma = sigma

        # Create array of voxel center coordinates, grid_3d.
        x_grid = np.linspace(0, self.box_length, n_grid + 1)  # grid points
        d = x_grid[:-1]+(x_grid[1]-x_grid[0])/2    # grid centers
        self.grid_3d = np.zeros((n_grid**3, 3))  # voxel center coordinates

        counter = 0
        for i in range(n_grid):
            for j in range(n_grid):
                for k in range(n_grid):
                    self.grid_3d[counter, :] = np.array([d[i], d[j], d[k]])
                    counter += 1

    def __call__(self, atoms, agent_num, A, B):

        if np.any(atoms.get_pbc()):
            rel_positions = self.get_edges_neighborlist(atoms, agent_index=self.agent_num)
        else:
            edges, edges_features = self.get_edges_simple(atoms)

        default_type = torch.get_default_dtype()

        # Let agent atom be origin of spherical internal coordinate system
        # (this is achived by using the relative positions above)

        # All positions are relative to agent atom, same goes for A and B
        pos = torch.tensor(rel_positions)
        A = torch.tensor(A)
        B = torch.tensor(B)

        # Find unit vector normal to A and B as cross product
        AB_cross = torch.cross(A, B)
        n = AB_cross/torch.linalg.norm(AB_cross)

        # Find unit vector normal to B and n as cross product
        Bn_cross = torch.cross(B, n)
        Bn_cross = Bn_cross/torch.linalg.norm(Bn_cross)

        # (https://www.cliffsnotes.com/study-guides/algebra/linear-algebra/real-euclidean-vector-spaces/projection-onto-a-subspace)
 
        # Notation: here vs. in the thesis:   B=B, A=n, n=left, Bn_cross=up
        # coordinate system: B, n, Bn_cross aka.  B, left, up

        x_coef = (np.dot(pos, B)/torch.dot(B, B)).unsqueeze(1)
        y_coef = (np.dot(pos, n)/torch.dot(n, n)).unsqueeze(1)
        z_coef = (np.dot(pos, Bn_cross)/torch.dot(Bn_cross, Bn_cross)).unsqueeze(1)

        pos_align = x_coef * torch.tensor([1, 0, 0]) + y_coef * torch.tensor([0, 1, 0]) + z_coef * torch.tensor([0, 0, 1])

        ##################################################################
        ###### Create voxel density of surrounding atoms (channel1) ######
        ##################################################################


        # Recenter the postions such that the agent is placed in the cube center
        pos_conv = pos_align.numpy() + np.repeat(self.cutoff, 3)

        # Number of nearby atoms
        n_atoms_local = pos_conv.shape[0]

        # start_time = time.time()
        channel1 = np.zeros(self.n_grid ** 3)
        for i in range(n_atoms_local):
            channel1 += np.exp(-(np.linalg.norm(self.grid_3d - pos_conv[i,:],
                            axis=1)**2) / (2*self.sigma**2))
        # print("Channel1: --- %s seconds ---" % (time.time() - start_time))
        channel1 = np.reshape(channel1, (self.n_grid, self.n_grid, self.n_grid))

        voxel = torch.from_numpy(channel1).float()
        
        return voxel.unsqueeze(0).to(device)


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

    def get_edges_neighborlist(self, atoms, agent_index):

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


        neigh_idx, rel_pos, neigh_dist2 = neighborlist.get_neighbors(agent_index, self.cutoff)
        rel_positions.append(rel_pos)

        return np.concatenate(rel_positions)



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


def get_voxel_repr(pos_align, agent_atom, grid_3d, n_grid, box_length, radius=3,
                   sigma=0.3, device=torch.device("cpu")):
    """
        Calculates the total voxel density from an array of Cartesian coordinates
    """

    # All atom positions
    pos = atom_object.get_positions()
    
    # Position of agent atom
    pos_agent = pos[agent_atom]

    # Unit cell
    cell = atom_object.get_cell()
    
    # Create enhanced cell using unit cell periodicity
    #   (speed up: maybe do if statemant to only enhance in
    #   some directions depending on positions of agent atom)
    # start_time = time.time()
    pos_list = pos.tolist()
    pos_e = pos_list.copy()
    pos_e += (pos_list+cell[0]).tolist()
    pos_e += (pos_list-cell[0]).tolist()
    pos_e += (pos_list+cell[1]).tolist()
    pos_e += (pos_list-cell[1]).tolist()
    pos_e += (pos_list+cell[0]+cell[1]).tolist()
    pos_e += (pos_list+cell[0]-cell[1]).tolist()
    pos_e += (pos_list-cell[0]+cell[1]).tolist()
    pos_e += (pos_list-cell[0]-cell[1]).tolist()
    pos_e = np.array(pos_e)
    # print("ENHANCEMENT: --- %s seconds ---" % (time.time() - start_time))
    
    # assert radius > np.linalg.norm(env.predict_goal_location()-pos_agent)
    # Remove agent atom from channel 1
    pos_e = np.delete(pos_e, agent_atom, axis=0)

    ##################################################################
    ###### Create voxel density of surrounding atoms (channel1) ######
    ##################################################################
    
    # Only keep atoms less than "radius" Ãƒâ€¦ from the agent atom
    pos_local = pos_e[np.linalg.norm(pos_e-pos_agent, axis=1) < radius]

    # Recenter the postions such that the agent is placed in the cube center
    pos_conv = pos_local-pos_agent + np.repeat(box_length*0.5, 3)
    
    # Number of nearby atoms
    n_atoms_local = pos_conv.shape[0]

    # start_time = time.time()
    channel1 = np.zeros(n_grid ** 3)
    for i in range(n_atoms_local):
        channel1 += np.exp(-(np.linalg.norm(grid_3d - pos_conv[i,:],
                                    axis=1)**2) / (2*sigma**2))
    # print("Channel1: --- %s seconds ---" % (time.time() - start_time))
    channel1 = np.reshape(channel1, (n_grid, n_grid, n_grid))
    
    ##################################################################
    ###### Create voxel density of agent atom (channel2) #############
    ##################################################################
    # start_time = time.time()

    #pos_agent_conv = pos_agent - pos_agent + np.repeat(box_length*0.5, 3)
    # so silly, how should we actually represent the current agent state

    # (redundant if we always place it in the center anyway)
    #channel2 = np.exp(-(np.linalg.norm(grid_3d - pos_agent_conv,
    #                                axis=1)**2) / (2*sigma**2))

    #channel2 = np.reshape(channel2, (n_grid, n_grid, n_grid))
    
    ##################################################################
    ###### Create voxel density of goal location (channel3) ##########
    ##################################################################

    # Recenter the postions
    #pos_conv_goal = pos_goal-pos_agent + np.repeat(box_length*0.5, 3)

    #channel3 = np.exp(-(np.linalg.norm(grid_3d - pos_conv_goal,
    #                                axis=1)**2) / (2*sigma**2))

    #channel3 = np.reshape(channel3, (n_grid, n_grid, n_grid))
    # print("TWO MORE CHANNELS: --- %s seconds ---" % (time.time() - start_time))

    ##################################################################
    ###################### Stacking ##################################
    ##################################################################
    
    #voxel = np.stack((channel1, channel2, channel3))
    voxel = torch.from_numpy(channel1).double()
    
    return voxel.unsqueeze(0).to(device)
