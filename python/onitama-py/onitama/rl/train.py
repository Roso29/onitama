import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from datetime import datetime
from onitama.rl import DQNMaskedCNNPolicy, ACMaskedCNNPolicy, SimpleAgent, RandomAgent
from onitama.rl.callbacks import EvalCB
from stable_baselines import DQN, PPO2
from stable_baselines.common.evaluation import evaluate_policy
from stable_baselines.common.callbacks import CheckpointCallback, EvalCallback, CallbackList
from stable_baselines.bench.monitor import Monitor
import numpy as np
import argparse
import onitama
import gym


def train_rl(seed, eval_freq, total_timesteps, isDQN, isRandom, decrease_threshold, threshold_decrease_factor, win_rate_threshold, reward_dict, initial_threshold=0.0):
    agent_type = RandomAgent if isRandom else SimpleAgent
    env = gym.make("Onitama-v0", seed=seed, agent_type=agent_type, verbose=False, reward_dict=reward_dict)
    eval_env = gym.make("Onitama-v0", seed=seed, agent_type=agent_type, verbose=False)


    #Only decrease threshold if playing SimpleAgent
    assert not isRandom or not decrease_threshold

    if initial_threshold > 0:
        env.game.agent.threshold = initial_threshold
        eval_env.game.agent.threshold = initial_threshold

    if not isRandom and decrease_threshold:
        env.game.agent.threshold = 1
        eval_env.game.agent.threshold=1
    if isDQN:
        basedir = "./logs/dqn-tb/"
        env, logdir = setup_monitor(basedir, env)
        policy = DQN(DQNMaskedCNNPolicy,
                     env,
                     seed=seed,
                     prioritized_replay=True,
                     verbose=1,
                     tensorboard_log=logdir
                     )

    else:
        basedir = "./logs/ppo-tb/"
        env, logdir = setup_monitor(basedir, env)
        policy = PPO2(ACMaskedCNNPolicy,
                      env,
                      seed=seed,
                      verbose=1,
                      tensorboard_log=logdir
                      )

    checkpoint_callback = CheckpointCallback(save_freq=5e3, save_path=logdir,
                                             name_prefix='rl_model', verbose=2)
    eval_policy_cb = EvalCB(logdir)
    eval_callback = EvalCallback(eval_env, best_model_save_path=logdir,
                                 log_path=logdir, eval_freq=eval_freq, n_eval_episodes=20,
                                 deterministic=True, render=False,
                                 evaluate_policy_callback=eval_policy_cb, env=env,
                                 decrease_threshold=decrease_threshold,
                                 threshold_decrease_factor=threshold_decrease_factor,
                                 win_rate_threshold=win_rate_threshold
                                 )
    callback = CallbackList([checkpoint_callback, eval_callback])
    policy.learn(int(total_timesteps), callback=callback, log_interval=100 if isDQN else 10)



def setup_monitor(basedir, env):
    logdir = basedir + datetime.now().strftime("%Y_%m_%d-%I_%M_%S_%p")
    if not os.path.exists(basedir): os.mkdir(basedir)
    if not os.path.exists(logdir): os.mkdir(logdir)
    env = Monitor(env, logdir + "/logs")
    return env, logdir


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', default=12314, type=int)
    parser.add_argument('--eval_freq', default=2500, type=int)
    parser.add_argument('--total_timesteps', default=1e6, type=float)
    parser.add_argument('--DQN', default=False, action="store_true", help="Use DQN")
    parser.add_argument('--random', default=False, action="store_true", help="Use random agent")
    parser.add_argument('--decrease_threshold', default=False, action="store_true", help="Decrease number of Simple Agent random moves as RL agent improves")
    parser.add_argument('--threshold_decrease_factor', default=1, type=float, help="How much to decrease proportion of random moves made by simple agent by. -(n * 0.1)")
    parser.add_argument('--win_rate_threshold', default=0.8, type=float, help="Proportion of wins by RL agent before decreasing Simple Agent Stochasicity")
    parser.add_argument('--reward_dict', default=0, type=int, help="Which reward dict index to use.")
    parser.add_argument('--initial_threshold', default=0.0, type=float, help="Set fixed random/simple blend")

    args = parser.parse_args()

    train_rl(args.seed, args.eval_freq, args.total_timesteps, args.DQN, args.random, args.decrease_threshold, args.threshold_decrease_factor, args.win_rate_threshold, args.reward_dict, args.initial_threshold)
