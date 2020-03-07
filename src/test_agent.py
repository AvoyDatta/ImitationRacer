# Third-party packages and modules:
from collections import deque
from datetime import datetime
import numpy as np
import gym, os, json
# My packages and modules:
from agent import Agent
import utils
import argparse
import time 

ckpt_dir = '../ckpts/'
results_dir = '../results/'
ep_len = 60 #length of episdoe in seconds


def run_episode(env, agent, rendering=True, max_timesteps=1000):
    # Reset reward accumulator
    episode_reward = 0
    # Inform environment and agent that a new episode is about to begin:
    env_state = env.reset()
    agent.begin_new_episode(state0=env_state)


    start_time = time.time()
    current_time = 0
    for _ in range(max_timesteps):
        # Request action from agent:
        agent_action = agent.get_action(env_state)
        # Given this action, get the next environment state and reward:
        env_state, r, done, info = env.step(agent_action)   
        # Render the state screen:
        if rendering:
            env.render()
        # Accumulate reward
        episode_reward += r
        # Check if environment signaled the end of the episode:
        current_time = time.time()
        delta_time = current_time - start_time
        if done or delta_time >= ep_len: break

    return episode_reward

def save_performance_results(episode_rewards, directory):
    results = dict()
    results["episode_rewards"] = episode_rewards
    results["mean"] = np.array(episode_rewards).mean()
    results["std"] = np.array(episode_rewards).std()
    
    time_stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    fname = f"{directory}results_bc_agent-{time_stamp}.json"
    fname = os.path.join(directory, "results_bc_agent-{}.json".format(time_stamp))
    with open(fname, "w") as fh:
        json.dump(results, fh)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--user", type=str, default="user", help="Insert name of user generating data.")
    parser.add_argument("--eps", type=int, default=15, help="Insert number of episodes to run.")

    args = parser.parse_args()

    user_path = os.path.join(ckpt_dir, args.user)
    saved_path = os.path.join(user_path, utils.get_model_path(user_path, metric='best'))

    print("Loading model from {}".format(saved_path))
    results_path = os.path.join(results_dir, args.user)
    if not os.path.exists(results_path):
        os.mkdir(results_path)

    # Number of episodes to test:
    n_test_episodes = args.eps #15

    # Initialize environment and agent:
    env = gym.make('CarRacing-v0').unwrapped
    agent = Agent.from_file(saved_path)

    # Episodes loop:
    episode_rewards = []
    for i in range(n_test_episodes):
        episode_reward = run_episode(env, agent, rendering=True)
        episode_rewards.append(episode_reward)
        print(f'Episode {i+1} reward:{episode_reward:.2f}')
    env.close()


    # save reward statistics in a .json file
    save_performance_results(episode_rewards, results_path)
    print("Saving results to {}".format(results_path))
    print('... finished')
