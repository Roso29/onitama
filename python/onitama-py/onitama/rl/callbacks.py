import tensorflow as tf
from stable_baselines.common.callbacks import BaseCallback

class EvalCB:
    def __init__(self, logdir=None, isPlayer1=True):
        self.logdir = logdir
        self.writer = None
        if self.logdir:
            self.writer = tf.summary.FileWriter(self.logdir)

        self.isPlayer1 = isPlayer1
        self.num_timesteps = 0
        self.reset()

    def reset(self):
        self.n_wins = 0
        self.n_eps = 0
        self.eps_rew = 0

    def callback(self, locals, globals):
        self.num_timesteps += 1
        info = locals["_info"] if not type(locals["_info"]) == list else locals["_info"][0]
        self.eps_rew += locals["reward"]
        if "is_success" in info:
            self.n_wins += info["is_success"]
        if locals["done"]:
            self.n_eps += 1
            if self.writer:
                summary = tf.Summary()
                summary.value.add(tag="eval episode returns", simple_value=self.eps_rew)
                self.writer.add_summary(summary, self.num_timesteps)
                self.writer.flush()
            self.eps_rew = 0

    def print(self):
        print("Won {} / {}".format(self.n_wins, self.n_eps))
        winRate = self.n_wins/self.n_eps
        if self.writer:
            summary = tf.Summary()
            summary.value.add(tag="eval win rate", simple_value=self.n_wins / self.n_eps)
            self.writer.add_summary(summary, self.num_timesteps)
            self.writer.flush()
        self.reset()
        return winRate
