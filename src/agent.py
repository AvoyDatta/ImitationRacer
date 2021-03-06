# Third-party packages and modules:
from collections import deque
import numpy as np
import abc
# My modules and packages:
import utils
from utils import config

import my_neural_network as mnn
import tensorflow as tf


class Agent:
    # Constructor is "overloaded" by the functions bellow.
    def __init__(self, model):
        # The neural network:
        self.model = model
        # Just a constant:
        self.accelerate = np.array([0.0, 1.0, 0.0], dtype=np.float32)


    @classmethod  # Constructor for a brand new model
    def from_scratch(cls, model_type, config, n_channels=1):
        # model=None
        print("Creating {} model".format(model_type))
        if model_type == 'baseline':
            layers = [
                mnn.layers.Input(input_shape=[96, 96, n_channels]), 
                mnn.layers.Conv2d(filters=16, kernel_size=5, stride=4), 
                mnn.layers.ReLU(), 
                mnn.layers.Dropout(drop_probability=0.5),
                mnn.layers.Conv2d(filters=32, kernel_size=3, stride=2), 
                mnn.layers.ReLU(), 
                mnn.layers.Dropout(drop_probability=0.5),
                mnn.layers.Flatten(), 
                mnn.layers.Linear(n_units=128), 
                mnn.layers.Linear(n_units=utils.n_actions) 
            ]
        else:
            layers = [             
                mnn.layers.Input(input_shape=[96, 96, n_channels]),

                mnn.layers.Reshape([96, 96, 1]), 
                mnn.layers.Conv2d(filters=16, kernel_size=5, stride=4), 
                mnn.layers.ReLU(), 
                # tf.layers.BatchNormalization(),
                mnn.layers.Dropout(drop_probability=0.5),
                mnn.layers.Conv2d(filters=32, kernel_size=3, stride=2), 
                mnn.layers.ReLU(), 
                mnn.layers.Dropout(drop_probability=0.5),
                # tf.layers.BatchNormalization(),
                mnn.layers.Flatten(), 
                mnn.layers.Linear(n_units=config['lstm_inp_dim']),

                mnn.layers.Reshape([config["history_length"], config['lstm_inp_dim']]), 

                ## RNN Layer
                mnn.layers.LSTMCell(config['lstm_hidden'], config['num_classes'])
                #(N, n_actions)

            ]        
        model = mnn.models.Classifier_From_Layers(layers, class_balancing=config['class_balancing'])
        return Agent(model)
    
    @classmethod  # Constructor to load a model from a file
    def from_file(cls, file_name):
        model = mnn.models.Classifier_From_File(file_name)#('saved_models/')
        return Agent(model)

    def train(self, X_train, y_train, X_valid, y_valid, n_batches, batch_size, lr, display_step, ckpt_step, ckpt_path, seed, train_start):
        print("Training model")
        self.model.train(X_train, y_train, X_valid, y_valid, n_batches, batch_size, lr, train_start, display_step, ckpt_step, ckpt_path, seed)

    def begin_new_episode(self, state0, hist_len=3, si=1):
        # A history of the last n agent's actions
        self.action_history = deque(maxlen=100)
        # Buffer for actions that may eventually overwrite the model
        self.overwrite_actions = []
        # Keep track of how many state transitions were made
        self.action_counter = 0
        # This data structure (kind of a deque) will always store the
        # last 'history_lenght' states and will be fed to the model:
        # self.state_hist = np.empty((1, state0.shape[0], state0.shape[1], utils.history_length))
        self.si = si
        self.history_length = hist_len
        self.state_hist = np.empty((1, state0.shape[0], state0.shape[1], self.si*self.history_length))

        for _ in range(self.history_length):
            self.__push_state(state0)

    def __push_state(self, state):
        # Push the current state to the history. 
        # Oldest state in history is discarded.
        sg = state.astype(np.float32)
        sg = np.expand_dims(sg, 0)
        sg = utils.preprocess_state(sg)
        # self.state_hist[0,:,:,1:] = self.state_hist[0,:,:,:-1]
        self.state_hist[0,:,:,:-1] = self.state_hist[0,:,:,1:]
        # self.state_hist[0,:,:,0] = sg[0]
        self.state_hist[0,:,:,-1] = sg[0]

    def get_action(self, env_state):

        ## Assume utils.dead_start >> utils.history_len * config['sample_interval']

        # Add the current state to the state history:
        self.__push_state(env_state)

        # First actions will always be to accelerate:
        if self.action_counter < utils.dead_start:
            self.action_history.append(self.accelerate)
            self.action_counter += 1
            return self.accelerate

        # If the car is stuck for too long, the neural network is overwritten:
        if len(self.overwrite_actions) > 0:
            print('Neural network overwritten')
            action = self.overwrite_actions.pop()
            self.action_history.append(action)
            return action

        # Check if the car is frozen:
        if self.check_freeze():
            print('Freeze detected. Overwritting neural network from next state onwards')

        # Uses the NN to choose the next action:
        downsampled_hist = utils.downsample_along_last(self.state_hist, factor=self.si)
        agent_action = self.model.predict(downsampled_hist)

        agent_action = utils.transl_action_agent2env(agent_action)
        self.action_history.append(agent_action)
        return agent_action
    
    def check_freeze(self):
        # If all the last actions are all the same and they 
        # are not accelerate, then the car is stuck somewhere.
        fa = self.action_history[0]
        for a in self.action_history:
            if not np.all(a==fa):
                return False
            if np.all(a == self.accelerate):
                return False
        
        # If the code reaches this point, the car is stuck
        fa[2] = 0.0  # release break
        overwrite_cycles = 2
        one_cicle = 10 * [fa] + 10 * [self.accelerate]
        self.overwrite_actions = overwrite_cycles * one_cicle
        return True

    def save(self, file_name):
        # Save model to a file
        self.model.save(file_name, close_session=True)

