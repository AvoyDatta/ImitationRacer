# Some helper functions to deal

import argparse
import numpy as np
import pickle
import gzip
import copy

def read(filename):
    '''

    :param data_path: path to the gzip file
    :return: state, next_state, reward, action, terminal
    '''
    if not exists(filename):
        raise Exception("File {0} does not exist".format(filename))

    file_handler = gzip.open(filename, 'rb')
    data = pickle.load(file_handler)
    state = data['state']
    next_state = data['next_state']
    reward = data['reward']
    action = data['action']
    terminal = data['terminal']

    return state, next_state, reward, action, terminal


