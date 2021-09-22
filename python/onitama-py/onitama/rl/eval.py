from onitama.rl import DQNMaskedCNNPolicy, SimpleAgent, RandomAgent
from onitama.rl.callbacks import EvalCB
from stable_baselines import DQN, PPO2
from stable_baselines.common.evaluation import evaluate_policy
import numpy as np
import argparse
import onitama
import gym


def evaluate_rl(policy, env, n_eps=100):
    eval_cb = EvalCB()
    episode_rewards, episode_lengths = evaluate_policy(policy, env,
                                                       callback=eval_cb.callback,
                                                       return_episode_rewards=True,
                                                       n_eval_episodes=n_eps)
    print("Mean reward: {}".format(np.mean(episode_rewards)))
    print("Std reward: {}".format(np.std(episode_rewards)))
    print("Min reward: {}".format(np.min(episode_rewards)))
    print("Max reward: {}".format(np.max(episode_rewards)))
    print("Mean episode length: {}".format(np.mean(episode_lengths)))
    print("Std episode length: {}".format(np.std(episode_lengths)))
    print("Min episode length: {}".format(np.min(episode_lengths)))
    print("Max episode length: {}".format(np.max(episode_lengths)))
    eval_cb.print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('model_path', type=str)
    parser.add_argument('--seed', default=12314, type=int)
    parser.add_argument('--DQN', action="store_true", help="Use DQN")
    parser.add_argument('--simple_threshold', default=0.0, type=float, help="Threshold for simple agent")
    parser.add_argument('--random', action="store_true", help="Use random agent")
    args = parser.parse_args()

    agent_type = RandomAgent if args.random else SimpleAgent
    env = gym.make("Onitama-v0", seed=args.seed, agent_type=agent_type, verbose=False)
    if not args.random:
        env.game.agent.threshold = args.simple_threshold
    if args.DQN:
        policy = DQN.load(args.model_path)
    else:
        policy = PPO2.load(args.model_path)

    evaluate_rl(policy, env)
