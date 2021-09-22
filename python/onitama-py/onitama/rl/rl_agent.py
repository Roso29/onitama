from onitama.rl import DQNMaskedCNNPolicy, get_mask, SimpleAgent, actionToMove
from onitama.rl.env import _get_obs, get_reward
from stable_baselines import DQN, PPO2

import numpy as np


class RLAgent:
    """
    Wraps policy to work with backend API calls
    """
    def __init__(self, seed, model_path, algorithm="PPO", isPlayer1=False):
        """
        Assumes player 2 as this is normal
        """
        self.isPlayer1 = isPlayer1

        if algorithm == "PPO":
            self.policy = PPO2.load(model_path)
        else:
            self.policy = DQN.load(model_path)
        np.random.seed(seed)

    def get_action(self, state):
        obs = np.concatenate([_get_obs(state, self.isPlayer1), get_mask(state, self.isPlayer1)], -1)
        ac, _ = self.policy.predict([obs])
        ac = np.squeeze(ac)
        # action is index into 5 x 5 x 50
        mask_shape = (5, 5, 50)
        ac = np.unravel_index(ac, mask_shape)
        move = actionToMove(ac, state, self.isPlayer1, mask_shape)
        return move
