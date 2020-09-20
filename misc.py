
import numpy as np

def create_action_space(step_size: float=0.1):
    """
        Creates flattened array of action displacement vectors in 3d
        Should probably be a list instead
    """
    action_space = np.zeros((3,3,3), object)
    for i in range(0, 3):
        for j in range(0, 3):
            for k in range(0, 3):
                action_space[i,j,k] = np.array([i-1,j-1,k-1])

    action_space = action_space.flatten()
    if max(action_space[13]) == 0:
        action_space = np.delete(action_space, 13)

    for i in range(len(action_space)):
        action_space[i] = np.divide(action_space[i], np.linalg.norm(action_space[i])) * step_size

    # np.stack(action_space)
    return action_space


def drift_projection(action_space: np.ndarray, agent_to_goal: np.ndarray):
    """
        Calculates the dot product (i.e. projection) of each action
        vector and the vector pointing towards the agent's goal position.
    """
    proj_list = [np.dot(action_space[i], agent_to_goal) for i in range(action_space.size)]

    return np.array(proj_list)