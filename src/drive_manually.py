## Author: Gui Miotto

from __future__ import print_function

import argparse
from pyglet.window import key
import gym
import numpy as np
import pickle
import os
from datetime import datetime
import gzip
import json
import time
from tqdm import tqdm
import copy

'''
def key_press(k, mod):
    global restart
    #if k == 0xff0d: restart = True
    if k == key.ESCAPE: restart = True
    if k == key.LEFT:  a[0] = -1.0
    if k == key.RIGHT: a[0] = +1.0
    if k == key.UP:    a[1] = +1.0
    if k == key.DOWN:  a[2] = +0.4  # stronger brakes
'''

ep_len = 60 #length of episdoe in seconds

def key_press(k, mod):
    global restart
    #if k == 0xff0d: restart = True
    if k == key.ESCAPE: restart = True
    if k == key.UP:    
        a[3] = +1.0
        if a[0] == 0.0:
            a[1] = +1.0
    if k == key.LEFT:  
        a[0] = -1.0
        a[1] =  0.0  # Cut gas while turning
    if k == key.RIGHT: 
        a[0] = +1.0
        a[1] =  0.0  # Cut gas while turning
    if k == key.DOWN:  
        a[2] = +0.4  # stronger brakes

def key_release(k, mod):
    if k == key.LEFT and a[0] == -1.0: 
        a[0] = 0.0
        if a[3] == 1.0:
            a[1] = 1.0
    if k == key.RIGHT and a[0] == +1.0: 
        a[0] = 0.0
        if a[3] == 1.0:
            a[1] = 1.0
    if k == key.UP:    
        a[1] = 0.0
        a[3] = 0.0
    if k == key.DOWN:  
        a[2] = 0.0


def store_data(data, datasets_dir="../data", timestamp = None):
    # save data
    if not os.path.exists(datasets_dir):
        os.mkdir(datasets_dir)
    if timestamp is None:
        save_name = 'data.pkl.gzip'
    else:
        save_name = 'data_{}.pkl.gzip'.format(timestamp)
    data_file = os.path.join(datasets_dir, save_name)
    f = gzip.open(data_file,'wb')
    pickle.dump(data, f)


def save_output(episode_rewards, output_dir="../data/"):
    # save output
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

     # save statistics in a dictionary and write them into a .json file
    output = dict()
    output["number_episodes"] = len(episode_rewards)
    output["episode_rewards"] = episode_rewards

    output["mean_all_episodes"] = np.array(episode_rewards).mean()
    output["std_all_episodes"] = np.array(episode_rewards).std()
 
    fname = os.path.join(output_dir, "results_manually-%s.json" % datetime.now().strftime("%Y%m%d-%H%M%S"))
    fh = open(fname, "w")
    json.dump(output, fh)
    print('... finished')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--collect_data", action="store_true", default=True, help="Collect the data in a pickle file.")
    parser.add_argument("--user", type=str, default="user", help="Insert name of user generating data.")

    args = parser.parse_args()

    # create directory if not present
    if not os.path.isdir('../data'):
        os.mkdir('../data')
    if not os.path.exists('../data/output'):
        os.mkdir('../data/output')

    data_dir = os.path.join("../data", args.user)
    output_dir = os.path.join("../data", "output", args.user)

    good_samples = {
        "state": [],
        "next_state": [],
        "reward": [],
        "action": [],
        "terminal" : [],
    }
    episode_samples = copy.deepcopy(good_samples)

    env = gym.make('CarRacing-v0').unwrapped
    env.reset()
    env.viewer.window.on_key_press = key_press
    env.viewer.window.on_key_release = key_release

    a = np.zeros(4, dtype=np.float32)
    
    episode_rewards = []
    good_steps = episode_steps = 0
    # Episode loop
    start_timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    while True:
        episode_samples["state"] = []
        episode_samples["action"] = []
        episode_samples["next_state"] = []
        episode_samples["reward"] = []
        episode_samples["terminal"] = []
        episode_reward = 0

        start_time = time.time()
        current_time = 0

        state = env.reset()
        restart = False
        episode_steps = good_steps

        # State loop

        with tqdm(total = ep_len) as pbar:
            while True:
                next_state, r, done, info = env.step(a[:3])
                episode_reward += r

                episode_samples["state"].append(state)            # state has shape (96, 96, 3)
                episode_samples["action"].append(np.array(a[:3]))     # action has shape (1, 3)
                episode_samples["next_state"].append(next_state)
                episode_samples["reward"].append(r)
                episode_samples["terminal"].append(done)
                
                state = next_state
                episode_steps += 1

                if episode_steps % 1000 == 0 or done:
                    print("\nstep {}".format(episode_steps))

                env.render()
                # break
                pbar.update(time.time() - current_time)

                current_time = time.time()
                delta_time = current_time - start_time

                # print('Time delta is: ', delta_time)
                if done or restart or delta_time >= ep_len:
                    print(delta_time)
                    break

                # if done or restart:
                #     break
            
            if not restart:
                good_steps = episode_steps

                episode_rewards.append(episode_reward)
                
                good_samples["state"].append(episode_samples["state"])
                good_samples["action"].append(episode_samples["action"])
                good_samples["next_state"].append(episode_samples["next_state"])
                good_samples["reward"].append(episode_samples["reward"])
                good_samples["terminal"].append(episode_samples["terminal"])

                print('... saving data')
                store_data(good_samples, data_dir, start_timestamp)
                save_output(episode_rewards, output_dir)

    env.close()

    

   
