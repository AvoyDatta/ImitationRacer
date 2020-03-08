import numpy as np
import multiprocessing as mp
from skimage.color import rgb2gray
from skimage import filters
import os
import json

import argparse
import numpy as np
import pickle
import gzip
import glob
import copy
import os
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
from datetime import datetime
import pdb

# Tells how long should the history be.
# Altering this variable has effects on ALL modules
history_length = 10


# Number of first states of each episode that shall be ignored
# from the expert dataset:
dead_start = 50
# Set of allowed actions:
actions = np.array([
    [ 0.0, 0.0, 0.0],  # STRAIGHT
    [ 0.0, 1.0, 0.0],  # ACCELERATE
    [ 1.0, 0.0, 0.0],  # RIGHT
    [ 1.0, 0.0, 0.4],  # RIGHT_BRAKE
    [ 0.0, 0.0, 0.4],  # BRAKE
    [-1.0, 0.0, 0.4],  # LEFT_BRAKE
    [-1.0, 0.0, 0.0],  # LEFT
], dtype=np.float32)


ranges = [0, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
n_actions = len(actions)



config = {
    'lstm_inp_dim': 64,
    'history_length':history_length,
    'lstm_hidden': 64,
    'num_classes': n_actions,
    'class_balancing': False,
    'sample_interval': 2,
    'random_seed': 10
}


def read_json(json_path):
    """
    Reads json, outputs dict 
    """
    with open(json_path, 'r') as fp:
        results = json.load(fp)
        print(results.keys())
    return results

def comparison_histogram(save_path, expert_json, agent_json):
    """
    Plot histogram of score distribution for a user 
    """
    agent = read_json(agent_json)
    expert = read_json(expert_json)
    # agent_rewards = np.expand_dims(np.array(agent["episode_rewards"]), 1)
    # expert_rewards = np.expand_dims(np.array(expert["episode_rewards"]), 1)
    agent_rewards = np.array(agent["episode_rewards"])
    expert_rewards = np.array(expert["episode_rewards"])
    # data = np.concatenate((expert_rewards, agent_rewards), 1)
    means = (expert['mean_all_episodes'], agent['mean'])
    stdevs = (expert['std_all_episodes'], agent['std'])

    # print(type(data[0,0]))
    # dist = pd.DataFrame(data, columns=['expert', 'agent'])

    # fig, ax = plt.subplots()
    # dist.plot.kde(ax=ax, legend=False, title='Histogram: Expert vs. Agent')
    plt.hist(expert_rewards, color='salmon', bins=ranges, label='expert', alpha=.75)
    plt.hist(agent_rewards, color='royalblue', bins=ranges, label='agent', alpha=.75)
    plt.legend()
    plt.ylabel("Episodes"); plt.xlabel('Episode Rewards'); plt.title("Distribution of rewards for {} expert and {} agent demonstrations".format(len(expert_rewards), len(agent_rewards)))
    # dist.plot.hist(ax=ax)
    # ax.set_ylabel('Probability')
    # ax.grid(axis='y')

    plt.savefig(save_path)
    # ax.set_facecolor('#d8dcd6')

def read_one_gzip(filename):
    '''

    :param data_path: path to the gzip file
    :return: state, next_state, reward, action, terminal
    '''
    if not os.path.exists(filename):
        raise Exception("File {0} does not exist".format(filename))

    file_handler = gzip.open(filename, 'rb')
    data = pickle.load(file_handler)
    state = data['state']
    next_state = data['next_state']
    reward = data['reward']
    action = data['action']
    terminal = data['terminal']

    return state, next_state, reward, action, terminal

def get_model_path(save_dir, metric='best'):
    if not os.path.exists(save_dir):
        raise Exception("The ckpt dir doesn't exist!")
    if metric == 'best':
        path = sorted(list([entry.name for entry in os.scandir(save_dir)]))[::-1][0]
    elif metric == 'recent':
        path = sorted(list([entry.name.split('_')[1] for entry in os.scandir(save_dir)]))[::-1][0]
    return path 


def read_all_gzip(user_dir):
    '''

    :param user_dir: path to user directory
    :return: 5 lists for state, next_state, reward, action, terminal with all of users data
    '''
    all_files = sorted(glob.glob(user_dir + '/*.gzip'))

    # empty lists for all
    all_states = [] #list of sessions
    all_next_states = [] 
    all_rewards = []
    all_actions = []
    all_terminals = []

    for file in all_files:
        state, next_state, reward, action, terminal = read_one_gzip(file)
        #state, next_state ... list of episodes (all stored as lists)

        state = np.concatenate([np.array(ep) for ep in state], 0)
        next_state = np.concatenate([np.array(ep) for ep in next_state], 0)
        action = np.concatenate([np.array(ep) for ep in action], 0)
        reward = np.concatenate([np.array(ep) for ep in reward], 0)
        terminal = np.concatenate([np.array(ep) for ep in terminal], 0)

        all_states.append(state)
        all_next_states.append(next_state)
        all_rewards.append(reward)
        all_actions.append(action)
        all_terminals.append(terminal)

    return all_states, all_next_states, all_rewards, all_actions, all_terminals




def action_arr2id(arr):
    """ Converts action from the array format to an id (ranging from 0 to n_actions) """
    ids = []
    for a in arr:
        id = np.where(np.all(actions==a, axis=1))
        ids.append(id[0][0])
    return np.array(ids)

def action_id2arr(ids):
    """ Converts action from id to array format (as understood by the environment) """
    return actions[ids]

def one_hot(labels):
    """ One hot encodes a set of actions """
    one_hot_labels = np.zeros(labels.shape + (n_actions,))
    for c in range(n_actions):
        one_hot_labels[labels == c, c] = 1.0
    return one_hot_labels

def unhot(one_hot_labels):
    """ One hot DEcodes a set of actions """
    return np.argmax(one_hot_labels, axis=1)

def transl_action_env2agent(acts):
    """ Translate actions from environment's format to agent's format """
    act_ids = action_arr2id(acts)
    return one_hot(act_ids)

def transl_action_agent2env(acts):
    """ Translate actions from agent's format to environment's format """
    act_arr = action_id2arr(acts)
    return act_arr[0]

def check_invalid_actions(y):
    """ Check if there is any forbidden actions in the expert database """
    inval_actions = [
        [ 0.0, 1.0, 0.4],  # ACCEL_BRAKE
        [ 1.0, 1.0, 0.4],  # RIGHT_ACCEL_BRAKE
        [-1.0, 1.0, 0.4],  # LEFT_ACCEL_BRAKE
        [ 1.0, 1.0, 0.0],  # RIGHT_ACCEL
        [-1.0, 1.0, 0.0],  # LEFT_ACCEL
    ]
    ia_count = 0
    for ia in inval_actions:
        ia_count += np.sum(np.all(y == ia, axis=1))
    if ia_count > 0:
        raise Exception('Invalid actions. Do something developer!')

def reduce_accelerate(X, y, drop_prob):
    """ Balance samples. Gets hide of a share of the most common action (accelerate) """
    # Enconding of the action accelerate
    acceler = np.zeros(7)
    acceler[1] = 1.
    # Find out what samples are labeled as accelerate
    is_accel = np.all(y==acceler, axis=1)
    # Get the index of all other samples (not accelerate)
    other_actions_index = np.where(np.logical_not(is_accel))
    # Randomly pick drop some accelerate samples. Probabiliy of dropping is given by drop_prob
    drop_mask = np.random.rand(len(is_accel)) > drop_prob
    accel_keep = drop_mask * is_accel
    # Get the index of accelerate samples that were kept
    accel_keep_index = np.where(accel_keep)
    # Put all actions that we want to keep together
    final_keep = np.squeeze(np.hstack((other_actions_index, accel_keep_index)))
    final_keep = np.sort(final_keep)
    X_bal, y_bal = X[final_keep], y[final_keep]

    return X_bal, y_bal


def action_weights(labels):
    _, counts = np.unique(labels, axis=0, return_indices=True, return_counts=True)
    weights = 1. / counts
    weights /= np.linalg.norm(weights)
    return weights

def balance_min_actions(X, y):
    actions, counts = np.unique(y, axis=0, return_counts=True)
    min_count = counts.min() # get min count
    # pick these many instances of all actions
    final_indices = np.empty(());
    for action in actions:
        mask = np.all(y==action, axis=1)
        idx = np.where(mask==True)[0][:min_count]
        final_indices = np.hstack((final_indices, idx))

    # get rid of the first element of the array
    final_indices = final_indices[1:].astype(int) # discard first
    X_bal, y_bal = X[final_indices], y[final_indices]

    return X_bal, y_bal    

def preprocess_state(states):
    """ Preprocess the images (states) of the expert dataset before feeding them to agent """
    states_pp = np.copy(states)
    
    # Paint black over the sum of rewards
    states_pp[:, 85:, :15] = [0.0, 0.0, 0.0]

    # Replace the colors defined bellow
    def replace_color(old_color, new_color):
        mask = np.all(states_pp == old_color, axis=3)
        states_pp[mask] = new_color

    # Black bar
    replace_color([000., 000., 000.], [120.0, 120.0, 120.0])

    # Road
    #new_road_color = [255.0, 255.0, 255.0]
    new_road_color = [102.0, 102.0, 102.0]
    replace_color([102., 102., 102.], new_road_color)
    replace_color([105., 105., 105.], new_road_color)
    replace_color([107., 107., 107.], new_road_color)
    # Curbs
    replace_color([255., 000., 000.], new_road_color)
    replace_color([255., 255., 255.], new_road_color)
    # Grass
    #new_grass_color = [0.0, 0.0, 0.0]
    new_grass_color = [102., 229., 102.]
    replace_color([102., 229., 102.], new_grass_color)
    replace_color([102., 204., 102.], new_grass_color)

    # Float RGB represenattion
    states_pp /= 255.

    # Converting to gray scale
    states_pp = rgb2gray(states_pp)

    return states_pp

def stack_history(X, y, N, shuffle=False, si=1):
    """ Stack states from the expert database into volumes of depth=history_length """
    x_stack = [X[i - N : i] for i in range(N, len(X)+1)]
    # x_stack = [X[::-1][len(X)-i:len(X)-i+si*N:si][::-1] for i in range(2*N, len(X)+1)]

    x_stack = np.moveaxis(x_stack, 1, -1)
    y_stack = y[N-1:]


    # Unused
    # if shuffle:
    #     order = np.arange(len(x_stack))
    #     np.random.shuffle(order)
    #     x_stack = x_stack[order]
    #     y_stack = y_stack[order]
    return x_stack, y_stack


def vstack(arr):
    """ 
    Expert database is divided by episodes.
    This function stack all those episodes together but discarding the 
    first dead_start samples of every episode.
    """
    stack = np.array(arr[0][dead_start:], dtype=np.float32)
    for i in range(1, len(arr)):
        stack = np.vstack((stack, arr[i][dead_start:]))
    return stack

def curr_time():
    return datetime.now().strftime("%Y%m%d-%H%M%S")

if __name__ == "__main__":
    # print(read_json("../results/avoy/results_bc_agent-20200225-135131.json"))

    comparison_histogram('histo.png', "../results_json/results_manually-20200223-175929.json", "../results_json/oneresults_bc_agent-20200225-134707.json")


