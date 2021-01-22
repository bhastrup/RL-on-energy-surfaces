
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
            
            NOTE:
            Trajectory-Based Off-Policy Deep Reinforcement Learning - https://arxiv.org/pdf/1905.05710.pdf
            "Instead of independent noise, temporally-correlated noise (Osband et al., 2016), or exploration 
            directly in parameter space can lead to a larger variety of behaviours (Plappert et al., 2017)."
        """

        self.action_space = action_space
        self.n_actions = len(self.action_space)

        self.k = k
        self.k_mu = k
        self.k_lamba = 0.01
        self.k_delta = 0.2
        self.k_p = 0.5
        self.k_ub = k * 3
        self.k_lb = k / 3

        self.s = sigma
        self.s_mu = sigma
        self.s_lamba = 0.0025
        self.s_delta = 0.2
        self.s_p = 0.5
        self.s_ub = sigma * 3
        self.s_lb = sigma / 3


    def select_action(self, agent_to_start: np.ndarray, agent_to_goal: np.ndarray,
                      t: int, t_max: int):
        
        # TODO: add uniform distribution contribution to softmax?
        
        # Evolve k and sigma according to random walk
        self._evolve_params()

        # Project actions vectors onto start and goal distance vectors
        start_proj = self._action_projection(agent_to_start)
        goal_proj = self._action_projection(agent_to_goal)

        # How far (temporally) have we progressed in the episode (between 0 and 1)
        lambda_sm = (t_max-t)/t_max

        start_dist = np.linalg.norm(agent_to_start)
        if t>0 and start_dist>0.:
            p_action = softmax([(
                (-lambda_sm)*start_proj/start_dist * self.s
                + (1-lambda_sm)*goal_proj) * self.k])[0]
        else:
            p_action = None
        
        # Draw action
        action = np.random.choice(
            self.n_actions,
            size=1,
            p=p_action
        )
        
        if p_action is not None:
            action_prob = p_action[action]
        else:
            action_prob = 1./self.n_actions

        return action.item(), action_prob


    def _action_projection(self, agent_to_goal: np.ndarray) -> np.ndarray:
        """
            Calculates the dot product (i.e. projection) of each action
            vector and the vector pointing towards the agent's goal position.
        """
        proj_list = [np.dot(self.action_space[i], agent_to_goal) for i in range(self.n_actions)]

        return np.array(proj_list)


    def _evolve_params(self) -> None:
        """
            Propagates k and sigma according to a stochastic differential equation
        """

        # Update k
        self.k = self._sde_walk(
            x_old=self.k,
            mu=self.k_mu,
            lamba=self.k_lamba,
            delta=self.k_delta,
            p=self.k_p,
            upper_bound=self.k_ub,
            lower_bound=self.k_lb,
            dw=np.random.randn(1)
        )

        # Update sigma
        self.s = self._sde_walk(
            x_old=self.s,
            mu=self.s_mu,
            lamba=self.s_lamba,
            delta=self.s_delta,
            p=self.s_p,
            upper_bound=self.s_ub,
            lower_bound=self.s_lb,
            dw=np.random.randn(1)
        )

        return None


    def _sde_walk(
        self,
        x_old: np.ndarray,
        mu: float,
        lamba: float,
        delta: float,
        p: float,
        upper_bound: float,
        lower_bound: float,
        dw: np.ndarray
    ) -> np.ndarray:
        """
        Function for propagating time varying parameters according to a stochastic difference equation.
        For stationary properties, see https://benjaminmoll.com/wp-content/uploads/2019/07/Lecture4_2149.pdf
        :param x_old: current x value
        :param mu: asymptotic mean
        :param lamba: decay/growth rate
        :param delta: size of noise
        :param p: x exponent
        :param upper_bound: reflection boundary
        :param lower_bound: reflection boundary
        :param dw: wiener process increment
        :return updated_random_walk: the random walk
        """

        # Obtain dimensions of x
        # dim_x = len(x_old)

        # Compute drift term
        drift_term = - lamba * (x_old - mu)

        # Compute diffusion term
        diffusion_term = delta * (x_old ** p) * dw

        # Update the random walk
        x_new = x_old + drift_term + diffusion_term

        # Reflect output in lower bound
        lb_diff = x_new - lower_bound
        x_new[lb_diff < 0] = x_new[lb_diff < 0] - 2 * lb_diff[lb_diff < 0]

        ub_diff = x_new - upper_bound
        x_new[ub_diff > 0] = x_new[ub_diff > 0] - 2 * ub_diff[ub_diff > 0]

        return x_new