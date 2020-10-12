
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


def get_voxel_repr(atom_object, agent_atom, pos_goal, grid_3d, n_grid, radius=3,
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
    
    # Remove agent atom from channel 1
    pos_e = np.delete(pos_e, agent_atom, axis=0)
    
    # assert radius > np.linalg.norm(env.predict_goal_location()-pos_agent)
    
    # Only keep atoms less than "radius" Ãƒâ€¦ from the agent atom
    pos_local = pos_e[np.linalg.norm(pos_e-pos_agent, axis=1) < radius]

    # Recenter the postions such that the agent is placed in the cube center
    pos_conv = pos_local-pos_agent + np.repeat(np.sqrt(radius), 3)
    
    # Recenter the postions such that the agent is placed in the cube center
    pos_conv_goal = pos_goal-pos_agent + np.repeat(np.sqrt(radius), 3)
    
    # Number of nearby atoms
    n_atoms_local = pos_conv.shape[0]
    
    # Create voxel density of surrounding atoms (channel1)
    channel1 = np.zeros(n_grid ** 3)
    for i in range(n_atoms_local):
        channel1 += np.exp(-(np.linalg.norm(grid_3d - pos[i,:],
                                    axis=1)**2) / (2*sigma**2))

    channel1 = np.reshape(channel1, (n_grid, n_grid, n_grid))
    
    # Create voxel density of agent atom (channel2)
    # (redundant if we always place it in the center anyway)
    channel2 = np.exp(-(np.linalg.norm(grid_3d - pos_agent,
                                    axis=1)**2) / (2*sigma**2))

    channel2 = np.reshape(channel2, (n_grid, n_grid, n_grid))
    
    # Create voxel density of goal location (channel3)
    channel3 = np.exp(-(np.linalg.norm(grid_3d - pos_goal,
                                    axis=1)**2) / (2*sigma**2))

    channel3 = np.reshape(channel3, (n_grid, n_grid, n_grid))
    
    voxel = np.stack((channel1, channel2, channel3))

    voxel = torch.from_numpy(voxel).double()
    
    return voxel.unsqueeze(0).to(device)