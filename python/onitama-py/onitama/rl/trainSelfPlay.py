from onitama.rl import DQNMaskedCNNPolicy, ACMaskedCNNPolicy, SimpleAgent, RandomAgent
from onitama.rl.eval import EvalCB
from onitama.rl.train import setup_monitor
from stable_baselines import DQN
from stable_baselines import PPO2
from stable_baselines.common.callbacks import CheckpointCallback, EvalCallback, CallbackList
import argparse
import onitama
import gym


def getPolicy(isDQN, seed):
    env = gym.make("OnitamaSelfPlay-v0", seed=seed, verbose=False)

    if isDQN:
        basedir = "./logs/dqn-self-tb/"
        env, logdir = setup_monitor(basedir, env)
        policy = DQN(DQNMaskedCNNPolicy,
                     env,
                     seed=seed,
                     prioritized_replay=True,
                     verbose=1,
                     tensorboard_log=logdir
                     )
    else:
        basedir = "./logs/ppo-self-tb/"
        env, logdir = setup_monitor(basedir, env)
        policy = PPO2(ACMaskedCNNPolicy,
                      env,
                      seed=seed,
                      verbose=1,
                      tensorboard_log=logdir
                      )

    env.setSelfPlayModel(policy)
    return policy, logdir


def train_rl(isDQN, seed, isRandom):
    policy, logdir = getPolicy(isDQN, seed)

    agent_type = RandomAgent if isRandom else SimpleAgent
    eval_env = gym.make("Onitama-v0", seed=seed, agent_type=agent_type, verbose=False)

    checkpoint_callback = CheckpointCallback(save_freq=5e3, save_path=logdir,
                                             name_prefix='rl_model', verbose=2)
    eval_policy_cb = EvalCB(logdir)
    eval_callback = EvalCallback(eval_env, best_model_save_path=logdir,
                                 log_path=logdir, eval_freq=1e3, n_eval_episodes=20,
                                 deterministic=True, render=False,
                                 evaluate_policy_callback=eval_policy_cb)
    callback = CallbackList([checkpoint_callback, eval_callback])
    policy.learn(int(1e6), callback=callback, log_interval=100 if isDQN else 10)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', default=12314, type=int)
    parser.add_argument('--DQN', action="store_true", help="Use DQN")
    parser.add_argument('--random', action="store_true", help="Use random agent")
    args = parser.parse_args()

    train_rl(args.DQN, args.seed, args.random)
