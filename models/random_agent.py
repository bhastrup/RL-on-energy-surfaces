
import numpy as np
from sklearn.utils.extmath import softmax

class RandomAgent():
    def __init__(self, action_space: np.ndarray, k: float=100.0, sigma: float=3.0):
        """
            The random agent takes a random first step and is then biased
            to continue in this direction.

            args:
                k: float - specifies width of softmax distribution
                           large k means the probability accumulates
                           on one or few actions
                sigma: float - specifies the strength of the bias away
                               from A relative to the bias towards B

        """

        self.action_space = action_space
        self.n_actions = len(self.action_space)
        self.k = k
        self.sigma = sigma

    
    def select_action(self, agent_to_start: np.ndarray, agent_to_goal: np.ndarray,
                      t: int, t_max: int):
        
        # TODO: add uniform distribution contribution to softmax

        start_proj = self._action_projection(agent_to_start)
        goal_proj = self._action_projection(agent_to_goal)

        lambda_sm = (t_max-t)/t_max # lambda softmax

        if t>0:
            p_action = softmax([(
                (-lambda_sm)*start_proj/np.linalg.norm(agent_to_start) * self.sigma
                + (1-lambda_sm)*goal_proj) * self.k])[0]
        else:
            p_action = None

        action = np.random.choice(
            self.n_actions,
            size=1,
            p=p_action
        )
        return action

    def _action_projection(self, agent_to_goal: np.ndarray):
        """
            Calculates the dot product (i.e. projection) of each action
            vector and the vector pointing towards the agent's goal position.
        """
        proj_list = [np.dot(self.action_space[i], agent_to_goal) for i in range(self.n_actions)]

        return np.array(proj_list)