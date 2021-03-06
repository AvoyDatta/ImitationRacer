from typing import List
import tensorflow as tf
import numpy as np
from tensorflow.python.saved_model import tag_constants
import shutil, os, time
from datetime import datetime
import pdb

import os
tensorboard_path = "../tensorboard_data/"

if __name__ == "__main__":
    from layers import Layer
else:
    from my_neural_network.layers import Layer    

import utils
import sklearn
from sklearn.utils.class_weight import compute_class_weight
from utils import config
# tf.set_random_seed(config['random_seed'])
# np.random.seed(config['random_seed'])



class Classifier_From_Layers:
    """ 
    A neuro-network for classification created from scratch capable of
    training and predicting. 'Save to file' function is also available.
    """

    def __init__(self, layers: List[Layer], class_balancing=False):

        self.class_balancing = class_balancing

        # Placeholder for putting the network in training mode:
        self.train_mode = tf.placeholder(tf.bool, name='Train_mode')
        # Input layer is initialized with x=None:
        layers[0].build(None, self.train_mode)
        # For all the others:
        for i, layer in enumerate(layers[1:]):
            print(i, str(layer))
            layer.build(layers[i](), self.train_mode)
        # Some useful aliases:
        self.input = layers[ 0]()  # Input placeholder
        self.logit = layers[-1]()  # Output logits
        # Placeholder for the labels:
        self.labels = tf.placeholder("float", self.logit.shape, name='Labels')
        # Loss function:
        if self.class_balancing:
            self.class_weights = tf.placeholder_with_default(np.ones((self.logit.shape[-1],1), dtype=np.float32), [self.logit.shape[-1],1], name='class_weights')
            self.sample_wts = tf.matmul(self.labels, self.class_weights)  #(N, )

        with tf.name_scope('Training_loss'):
            self.cross_entrop = tf.nn.softmax_cross_entropy_with_logits(
                logits=self.logit, 
                labels=self.labels, 
                name='Cross_entropy')

            if self.class_balancing: 
                print("Balancing output classes...")
                self.cross_entrop = self.sample_wts * self.cross_entrop

            self.loss_fn = tf.reduce_mean(self.cross_entrop, name='Loss_function')
        # Evaluation graph:
        with tf.name_scope('Evaluation'):
            self.prediction = tf.argmax(self.logit, 1, name='Prediction')
            correct_pred = tf.equal(self.prediction, tf.argmax(self.labels, 1), name='Score')
            self.accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32), name='Accuracy')
            self.mean_preds = tf.reduce_mean(self.prediction, name='Prediction_mean')

        # Model's session:
        self.sess = tf.Session()
        # Scalar summaries for Tensorboard
        summ_loss = tf.summary.scalar('Loss', self.loss_fn)
        summ_train_acc = tf.summary.scalar('Training accuracy', self.accuracy)
        summ_preds = tf.summary.scalar('Means of predictions', self.mean_preds)

        self.summ_train = tf.summary.merge([summ_loss, summ_train_acc, summ_preds])
        self.summ_valid = tf.summary.scalar('Validation accuracy', self.accuracy)


    def train(self, X_train, y_train, X_valid, y_valid, n_batches, batch_size, lr, train_start=None, display_step=100,
              ckpt_step=1e4, ckpt_path=None, seed=10):

        tf.set_random_seed(seed)
        np.random.seed(seed)

        optimizer = tf.train.AdamOptimizer(learning_rate=lr)
        #optimizer = tf.train.MomentumOptimizer(lr, 0.9, use_nesterov=True)
        train_op = optimizer.minimize(self.loss_fn)
        init = tf.global_variables_initializer()
        self.sess.run(init)
        # Setup a writter for tensorboard summaries
        if train_start is None:
            train_start = utils.curr_time()
        ckpt_path = os.path.join(ckpt_path, train_start)

        if not os.path.exists(ckpt_path):
            os.makedirs(ckpt_path)
            print("Created ckpt dir: ", ckpt_path)


        writer = tf.summary.FileWriter(os.path.join(tensorboard_path, train_start), self.sess.graph)
        # Training loop:

        if self.class_balancing: 
            wts = np.expand_dims(compute_class_weight('balanced', np.arange(y_train.shape[-1]), np.argmax(y_train, axis=1)), 1)

        for step in range(n_batches):
            # Sample training data
            pick = np.random.randint(0, len(y_train), batch_size)
            batch_x = X_train[pick]
            batch_y = y_train[pick]
            # batch_wts = wts[pick]
            # Foward and backward pass
            # print(wts.shape, wts)
            if self.class_balancing:
                _, loss, train_acc, summ = self.sess.run([train_op, self.loss_fn, self.accuracy, self.summ_train], 
                    feed_dict={self.input: batch_x, self.labels: batch_y, self.class_weights:wts, self.train_mode: True})
            else:
                # _, lossself.sess.run(train_op, feed_dict={self.input: batch_x, self.labels: batch_y, self.train_mode: True})
                _, loss, train_acc, summ = self.sess.run([train_op, self.loss_fn, self.accuracy, self.summ_train], 
                    feed_dict={self.input: batch_x, self.labels: batch_y, self.train_mode: True})
            # Display and store loss and accuracies every display_step
            ##

            if step % display_step == 0:
                ## DEBUG
                debug_argmax = self.sess.run(self.prediction, 
                        feed_dict={self.input: batch_x, self.train_mode: False})
                unique, counts = np.unique(debug_argmax, return_counts=True)

                debug_dict = dict(zip(unique, counts))
                print("Preds: ", debug_dict)

                unique_labels, counts_labels = np.unique(np.argmax(batch_y, 1), return_counts=True)

                print("Labels: ", dict(zip(unique_labels, counts_labels)))

                # Training statistics
                # loss, train_acc, summ = self.sess.run([self.loss_fn, self.accuracy, self.summ_train], 
                #     feed_dict={self.input: batch_x, self.labels: batch_y, self.train_mode: False})
                writer.add_summary(summ, step)
                # Validation statistics
                val_acc, summ = self.sess.run([self.accuracy, self.summ_valid], 
                    feed_dict={self.input: X_valid, self.labels: y_valid, self.train_mode: False})
                writer.add_summary(summ, step)
                print(f'Step: {step}, Loss: {loss:.5f}, Training accuracy: {train_acc:.4f}, Validation accuracy: {val_acc:.4f}')
                print(pick)

            if step % ckpt_step ==0:

                # Validation statistics
                val_acc, summ = self.sess.run([self.accuracy, self.summ_valid],
                                              feed_dict={self.input: X_valid, self.labels: y_valid,
                                                         self.train_mode: False})
                #writer.add_summary(summ, step)

                ## save model as ValAcc_timeStamp
                # create string_name
                current_time = utils.curr_time()
                val_string = str(round(val_acc, 3))

                new_path = val_string + "_" + current_time
                save_path = os.path.join(ckpt_path, new_path)

                if not os.path.exists(save_path):
                    os.makedirs(save_path)               
                #pdb.set_trace()
                print('Saving Model\n')
                self.save(save_path)
        
        writer.close()
        print("Training finished.")

        # #pdb.set_trace()
        # print('Saving Model\n')
        # save_path = os.path.join(ckpt_path, new_path)

        # if not os.path.exists(save_path):
        #     os.mkdir(save_path)    
        # self.save(save_path)

    def eval_test_accuracy(self, dataset):
        test_acc = self.sess.run(self.accuracy, 
            feed_dict={self.input: dataset.test.images, self.labels: dataset.test.labels, self.train_mode: False})
        print(f"Testing Accuracy: {test_acc:.4f}")
    
    def predict(self, x):
        pred = self.sess.run(self.prediction, feed_dict={self.input: x, self.train_mode: False})
        return pred
    
    def save(self, file_path: str, close_session: bool=False):
        if os.path.exists(file_path):
            shutil.rmtree(file_path)
        tf.saved_model.simple_save(self.sess, file_path, {'x':self.input, 'Train_mode':self.train_mode}, {'pred':self.prediction})
        if close_session:
            self.sess.close()


class Classifier_From_File:
    """ 
    A neuro-network for classification restored from a file.
    It can only make predictions.
    """
    def __init__(self, file_path: str):
        self.sess = tf.Session()
        # Loads model into default graph:
        tf.saved_model.loader.load(
            self.sess, 
            [tag_constants.SERVING],
            file_path,
            )

        defgraph = tf.get_default_graph()
        all_nodes = [n.name for n in defgraph.as_graph_def().node]
        self.train_mode = defgraph.get_tensor_by_name('Train_mode:0')
        self.input = defgraph.get_tensor_by_name('Network_Input:0')
        self.prediction = defgraph.get_tensor_by_name('Evaluation/Prediction:0')

    def predict(self, x):
        pred = self.sess.run(self.prediction, feed_dict={self.input: x, self.train_mode: False})
        return pred 

