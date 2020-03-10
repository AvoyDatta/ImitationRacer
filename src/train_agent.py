# Third-party packages and modules:
import pickle, os, gzip, json
import numpy as np
import matplotlib.pyplot as plt
# My packages and modules:
from agent import Agent
import utils
from utils import config
import pdb
import argparse

data_dir = '../data/'
ckpt_dir = '../ckpts/'
log_dir = '../logs'
save_every = 1000

# np.random.seed(config['random_seed'])



def read_data(data_path, use_last = False):
    # TODO: Fix the file thing
    print("Reading data...")
    all_states, _, _, all_actions, _ = utils.read_all_gzip(data_path)

    if use_last:
        all_states = all_states[-1:]
        all_actions = all_actions[-1:]

    X = utils.vstack(all_states)
    y = utils.vstack(all_actions)

    return X, y

# def read_data():
#     """Reads the states and actions recorded by drive_manually.py"""
#     print("Reading data")
#     with gzip.open('./data_from_expert/data_02.pkl.gzip','rb') as f:
#         data = pickle.load(f)
#     X = utils.vstack(data["state"])
#     y = utils.vstack(data["action"])
#     return X, y

def preprocess_data(X, y, hist_len, shuffle, sample_interval=1):
    """ Preprocess states and actions from expert dataset before feeding them to the agent """
    print('Preprocessing states. Shape:', X.shape)
    utils.check_invalid_actions(y)
    y_pp = utils.transl_action_env2agent(y)
    X_pp = utils.preprocess_state(X)
    X_pp, y_pp = utils.stack_history(X_pp, y_pp, hist_len, shuffle=shuffle, si=sample_interval)
    return X_pp, y_pp

def split_data(X, y, frac = 0.1, shuffle = True, seed=10):
    """ Splits data into training and validation set """
    split = int((1-frac) * len(y))

    if shuffle:
        idxs = np.arange(len(X))
        np.random.seed(seed)
        np.random.shuffle(idxs)
        X = X[idxs]
        y = y[idxs]

    X_train, y_train = X[:split], y[:split]
    X_valid, y_valid = X[split:], y[split:]
    return X_train, y_train, X_valid, y_valid

def plot_states(x_pp, X_tr=None, n=3):
    """ Plot some random states before and after preprocessing """
    pick = np.random.randint(0, len(x_pp), n)
    fig, axes = plt.subplots(n, 2, sharex=True, sharey=True, figsize=(20,20))
    for i, p in enumerate(pick):
        if X_tr is not None:
            axes[i,0].imshow(X_tr[p]/255)
        axes[i,1].imshow(np.squeeze(x_pp[p]), cmap='gray')
    fig.tight_layout()
    plt.show()

def plot_action_histogram(actions, title):
    """ Plot the histogram of actions from the expert dataset """
    acts_id = utils.unhot(actions)
    fig, ax = plt.subplots()
    bins = np.arange(-.5, utils.n_actions + .5)
    ax.hist(acts_id, range=(0,6), bins=bins, rwidth=.9)
    ax.set(title=title, xlim=(-.5, utils.n_actions -.5))
    plt.show()

if __name__ == "__main__":
    # Read data:
    # Preprocess it:
    parser = argparse.ArgumentParser()
    parser.add_argument("--user", type=str, default="user", help="Insert name of user generating data.")
    parser.add_argument("--model", type=str, default="lstm", help="Insert name of model.")
    parser.add_argument("--seed", type=int, default=10, help="Insert random seed.")
    parser.add_argument("--lr", type=float, default=5e-4, help="Insert learning rate.")
    parser.add_argument("--si", type=int, default=1, help="Insert sample_interval.")


    args = parser.parse_args()

    config['sample_interval'] = args.si
    lr = args.lr

    c_time = utils.curr_time()
    ckpt_path = os.path.join(os.getcwd(), ckpt_dir, args.user, args.model, c_time)
    if not os.path.exists(ckpt_path):
        os.makedirs(ckpt_path)
        print("Created ckpt dir: ", ckpt_path)
    ckpt_path = os.path.join(os.getcwd(), ckpt_dir, args.user, args.model)


    # pdb.set_trace()
    data_path = os.path.join(data_dir, args.user)
    X, y = read_data(data_path)
    X_pp, y_pp = preprocess_data(X, y, hist_len=utils.history_length, shuffle=False, sample_interval=config['sample_interval'])

    # return
    # Plot action histogram. JUST FOR DEBUGGING.
    plot_action_histogram(y_pp, 'Action distribution BEFORE balancing')   

    # Balance as per min action  
    # X_pp, y_pp = utils.balance_min_actions(X_pp, y_pp)
    drop_prob = 0.5
    X_pp, y_pp = utils.reduce_accelerate(X_pp, y_pp, drop_prob, seed=args.seed)

    # Plot action histogram. JUST FOR DEBUGGING.
    plot_action_histogram(y_pp, 'Action distribution AFTER balancing')   

    # Plot some random states before and after preprocessing. JUST FOR DEBUGGING. 
    # Requires to run the above fucntion with hist_len=1, shuffle=False.
    # plot_states(X_pp, X)
    # Split data into training and validation:
    X_train, y_train, X_valid, y_valid = split_data(X_pp, y_pp, frac=.1, seed=args.seed)
    # Create a new agent from scratch:
    # agent = Agent.from_scratch(n_channels=utils.history_length)

    agent = Agent.from_scratch(args.model, utils.config, n_channels=config['history_length'])
    # lr = 5e-3
    log_path = os.path.join(os.getcwd(), log_dir, args.user, args.model)
    if not os.path.exists(log_path):
    	os.makedirs(log_path)
    	print("Creating log dir: ", log_path)
    model_config = dict()
    model_config['config'] = config
    model_config['user'] = args.user
    model_config['acc_drop_prob'] = drop_prob
    model_config['learning_rate'] = lr
    fname = os.path.join(log_path, c_time + ".json")
    with open(fname, 'w') as fh:
    	json.dump(model_config, fh)

    # Train it:
    agent.train(X_train, y_train, X_valid, y_valid, n_batches=200000, batch_size=128, lr=lr, display_step=100,
                ckpt_step=save_every,
                ckpt_path = ckpt_path,
                seed=args.seed 
                ) # added more arguments

 
