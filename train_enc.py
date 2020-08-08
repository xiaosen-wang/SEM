#! /usr/bin/env python
# coding: utf-8

"""
    Train models using the encoded texts.
"""
import numpy as np
import pandas as pd
import os
import pickle
import time
import math
import yaml
import datetime
import tensorflow as tf
from tensorflow.contrib import learn
from sklearn.model_selection import KFold

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

import data_helpers
import synonyms_encode
from text_cnn import TextCNN
from text_rnn import TextRNN
from text_birnn import TextBiRNN

import encode_utils


# Parameters
# ==================================================

# Data loading params
tf.flags.DEFINE_string("model_type", "clf", "The type of model, classification or regression (default: clf)")  # clf/reg
tf.flags.DEFINE_string("nn_type", "textrnn", "The type of neural network type (default: textcnn)")  # fasttext/textdnn/textcnn/textrnn/textbirnn/textrcnn/texthan
tf.flags.DEFINE_string("data", "aclImdb", "The type of data (aclImdb, yahoo_answers, ag_news)")
tf.flags.DEFINE_float("sn", 10, "The number of the synonyms that use the same code, default to 5")
tf.flags.DEFINE_string("gpu", "0", "gpu to use") 
tf.flags.DEFINE_string("sigma", "1.0", "sigma to use") 
tf.flags.DEFINE_string("language_type", "en", "Text language type (default: en)")  # en/zh
tf.flags.DEFINE_float("dev_sample_percentage", .1, "Percentage of the training data to use for validation")
tf.flags.DEFINE_float("cross_val_folds", 10, "Split the training data to validation with k folds")

# Model Hyperparameters
tf.flags.DEFINE_boolean("enable_word_embeddings", True, "Enable/disable the word embedding (default: True)")
tf.flags.DEFINE_integer("embedding_dim", 128, "Dimensionality of character embedding (default: 128)")
tf.flags.DEFINE_integer("hidden_size", 128, "Number of hidden layer units (default: 128)")
tf.flags.DEFINE_integer("hidden_layers", 2, "Number of hidden layers (default: 2)")
tf.flags.DEFINE_string("filter_sizes", "3,4,5", "Comma-separated filter sizes (default: '3,4,5')")
tf.flags.DEFINE_integer("num_filters", 128, "Number of filters per filter size (default: 128)")
tf.flags.DEFINE_integer("rnn_size", 128, "Number of units rnn_size (default: 128)")
tf.flags.DEFINE_integer("num_rnn_layers", 3, "Number of rnn layers (default: 3)")
tf.flags.DEFINE_float("dropout_keep_prob", 0.5, "Dropout keep probability (default: 0.5)")
tf.flags.DEFINE_float("l2_reg_lambda", 0.0, "L2 regularization lambda (default: 0.0)")

# Training parameters
tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
tf.flags.DEFINE_integer("num_epochs", 5, "Number of training epochs (default: 15)")
tf.flags.DEFINE_integer("evaluate_every", 200, "Evaluate model on dev set after this many steps (default: 100)")
tf.flags.DEFINE_integer("checkpoint_every", 200, "Save model after this many steps (default: 100)")
tf.flags.DEFINE_integer("num_checkpoints", 2, "Number of checkpoints to store (default: 5)")
tf.flags.DEFINE_float("grad_clip", 5, "grad clip to prevent gradient explode")
tf.flags.DEFINE_float("decay_coefficient", 2.5, "Decay coefficient (default: 2.5)")

# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

FLAGS = tf.flags.FLAGS

IMDB_PATH = 'aclImdb'
MAX_VOCAB_SIZE = 50000
GLOVE_PATH = 'glove.840B.300d.txt'

def preprocess(path):
    """
    Get the data for train and test.
    Args:
        path: the path for the dictionary, tokenizer and embedding matrix.
    Returns:
        pass
    """
    """Load the dictionary and the tokenizer."""
    with open(('aux_files/enc_dic_%s_%d_%d_%s.pkl' % (FLAGS.data, MAX_VOCAB_SIZE, FLAGS.sn, FLAGS.sigma)), 'rb') as f:
        enc_dic = pickle.load(f)
    with open(('aux_files/tokenizer_%s_%d.pkl' % (FLAGS.data, MAX_VOCAB_SIZE)), 'rb') as f:
        tokenizer = pickle.load(f)

    """We only use the original sequence `train_seq` and `test_seq`"""
    train_seq, train_seq_o, train_labels = encode_utils.text_encode(tokenizer, enc_dic, FLAGS.data+'/train', MAX_VOCAB_SIZE)
    test_seq, test_seq_o, test_labels = encode_utils.text_encode(tokenizer, enc_dic, FLAGS.data+'/test', MAX_VOCAB_SIZE)

    """Load the embedding matrix, and pad sequence to the same length"""
    embedding_matrix = np.load(('aux_files/embeddings_glove_%s_%d.npy' %(FLAGS.data, MAX_VOCAB_SIZE)))
    max_len = 250
    x_train = pad_sequences(train_seq, maxlen=max_len, padding='post')
    y_train = np.array(train_labels)
    x_test = pad_sequences(test_seq, maxlen=max_len, padding='post')
    y_test = np.array(test_labels)

    # Get the totoal number of words encoded.
    encode_length = 1
    for key in enc_dic:
        encode_length = max(encode_length, enc_dic[key])
    encode_length += 1

    return x_train, y_train, x_test, y_test, embedding_matrix, encode_length

# Training
# ==================================================
os.environ["CUDA_VISIBLE_DEVICES"]=FLAGS.gpu

def train(x_train, y_train, x_dev, y_dev, embedding_matrix, vocab_encoded_length, num_classes=2):

    batch_size = 64
    lstm_size = 128
    num_epochs = 20
    max_len = 250
    with tf.Graph().as_default():
        session_conf = tf.GPUOptions(allow_growth=True)
        # session_conf = tf.ConfigProto(
        #   allow_soft_placement=FLAGS.allow_soft_placement,
        #   log_device_placement=FLAGS.log_device_placement)
        # session_conf.gpu_options.per_process_gpu_memory_fraction = 0.4
        sess = tf.Session(config=tf.ConfigProto(gpu_options=session_conf))

        with sess.as_default():
            # Get different model according to different value of `nn_type`.
            if FLAGS.nn_type == 'textcnn':
                nn = TextCNN(
                    sequence_length=max_len,
                    num_classes=num_classes,
                    vocab_size=vocab_encoded_length,
                    filter_sizes=list(map(int, FLAGS.filter_sizes.split(","))),
                    num_filters=FLAGS.num_filters,
                    l2_reg_lambda=FLAGS.l2_reg_lambda)
            elif FLAGS.nn_type == 'textrnn':
                nn = TextRNN(
                    sequence_length=max_len,
                    num_classes=num_classes,
                    vocab_size=vocab_encoded_length,
                    rnn_size=FLAGS.rnn_size,
                    num_layers=FLAGS.num_rnn_layers,
                    # batch_size=FLAGS.batch_size,
                    l2_reg_lambda=FLAGS.l2_reg_lambda)
            elif FLAGS.nn_type == 'textbirnn':
                nn = TextBiRNN(
                    sequence_length=max_len,
                    num_classes=num_classes,
                    vocab_size=vocab_encoded_length,
                    rnn_size=FLAGS.rnn_size,
                    num_layers=FLAGS.num_rnn_layers,
                    # batch_size=FLAGS.batch_size,
                    l2_reg_lambda=FLAGS.l2_reg_lambda)
            elif FLAGS.nn_type == 'textrcnn':
                nn = TextRCNN(
                    sequence_length=max_len,
                    num_classes=num_classes,
                    vocab_size=vocab_encoded_length,
                    batch_size=FLAGS.batch_size,
                    l2_reg_lambda=FLAGS.l2_reg_lambda)

            # Define Training procedure
            global_step = tf.Variable(0, name="global_step", trainable=False)
            optimizer = tf.train.AdamOptimizer(nn.learning_rate)
            # Clip the gradient to avoid larger ones
            tvars = tf.trainable_variables()
            grads, _ = tf.clip_by_global_norm(tf.gradients(nn.loss, tvars), FLAGS.grad_clip)
            # grads_and_vars = optimizer.compute_gradients(nn.loss)
            grads_and_vars = tuple(zip(grads, tvars))
            train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

            # Output directory for models and summaries
            timestamp = str(int(time.time()))
            out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs_enc_%s" % FLAGS.nn_type, timestamp))
            print("Writing to {}\n".format(out_dir))

            # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
            checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
            checkpoint_prefix = os.path.join(checkpoint_dir, "model")
            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)
            saver = tf.train.Saver(tf.global_variables(), max_to_keep=FLAGS.num_checkpoints)

            # Initialize all variables
            sess.run(tf.global_variables_initializer())
            # sess.run(tf.assign(nn.W, embedding_matrix.T))
            print('Training..')
            def train_step(x_batch, y_batch, learning_rate):
                """
                A single training step
                """
                feed_dict = {
                    nn.input_x: x_batch,
                    nn.input_y: y_batch,
                    nn.dropout_keep_prob: 0.8,
                    nn.learning_rate: learning_rate
                }
                _, step, loss, accuracy = sess.run(
                    [train_op, global_step, nn.loss, nn.accuracy], feed_dict)
                time_str = datetime.datetime.now().isoformat()
                if step % 100 == 0:
                    print("{}: step {}, lr {:g}, loss {:g}, acc {:g}".format(time_str, step, learning_rate, loss, accuracy))

            def dev_step(x_batch, y_batch):
                """
                Evaluates model on a dev set
                """
                if FLAGS.nn_type in ['textcnn','textrnn', 'textbirnn']:
                    feed_dict = {
                        nn.input_x: x_batch,
                        nn.input_y: y_batch,
                        nn.dropout_keep_prob: 1.0
                    }
                    step, loss, accuracy = sess.run(
                        [global_step, nn.loss, nn.accuracy], feed_dict)
                elif FLAGS.nn_type in ['textrcnn']:
                    loss_sum = 0
                    accuracy_sum = 0
                    step = None
                    batches_in_dev = len(y_batch) // FLAGS.batch_size
                    for batch in range(batches_in_dev):
                        start_index = batch * FLAGS.batch_size
                        end_index = (batch + 1) * FLAGS.batch_size
                        feed_dict = {
                            nn.input_x: x_batch[start_index:end_index],
                            nn.input_y: y_batch[start_index:end_index],
                            nn.dropout_keep_prob: 1.0
                        }
                        step, loss, accuracy = sess.run(
                            [global_step, nn.loss, nn.accuracy],feed_dict)
                        loss_sum += loss
                        accuracy_sum += accuracy
                    loss = loss_sum / batches_in_dev
                    accuracy = accuracy_sum / batches_in_dev
                time_str = datetime.datetime.now().isoformat()
                return step, loss, accuracy

            # Generate batches
            batches = data_helpers.batch_iter(
                list(zip(x_train, y_train)), FLAGS.batch_size, FLAGS.num_epochs)
            # It uses dynamic learning rate with a high value at the beginning to speed up the training
            max_learning_rate = 0.005
            min_learning_rate = 0.0001
            decay_speed = FLAGS.decay_coefficient*len(y_train)/FLAGS.batch_size
            # Training loop. For each batch...
            counter = 0
            best_eval_accuracy = 0
            last_val, curr_val = 0, 0
            acc_all = []
            for batch in batches:
                learning_rate = min_learning_rate + (max_learning_rate - min_learning_rate) * math.exp(-counter/decay_speed)
                counter += 1
                x_batch, y_batch = zip(*batch)
                train_step(x_batch, y_batch, learning_rate)

                batches_dev = data_helpers.batch_iter(list(zip(x_dev, y_dev)), FLAGS.batch_size*3, 1)
                time_str = datetime.datetime.now().isoformat()

                current_step = tf.train.global_step(sess, global_step)
                if current_step % FLAGS.evaluate_every == 0:
                    print("Evaluation:")
                    loss_all, accuracy_all = [], []
                    for batch_dev in batches_dev:
                        x_batch_dev, y_batch_dev = zip(*batch_dev)
                        step, loss, accuracy = dev_step(x_batch_dev, y_batch_dev)
                        loss_all.append(loss)
                        accuracy_all.append(accuracy)
                    accuracy_mean = np.mean(accuracy_all)
                    acc_all.append(accuracy_mean)
                    print("Evaluation: {}: step {}, loss {:g}, best acc {:g}, acc {:g}".format(time_str, step, np.mean(loss_all), best_eval_accuracy, accuracy_mean))

                    if accuracy_mean > best_eval_accuracy and step > 300:
                        last_val = best_eval_accuracy
                        best_eval_accuracy = accuracy_mean
                        if(best_eval_accuracy == last_val):
                            best_remain += 1
                            if best_remain == 10:
                                print(best_eval_accuracy)
                                break
                        path = saver.save(sess, checkpoint_prefix, global_step=current_step)
            print(np.mean(acc_all[-4:]))


def main(argv=None):
    path = FLAGS.data
    train_x, train_y, test_x, test_y, embedding_matrix, vocab_encoded_length = preprocess(path)

    if path == 'aclImdb':
        print('\nData: aclImdb!!!\n')
        num_classes = 2
    elif path == 'yahoo_answers':
        print('\nData: yahoo_answers!!!\n')
        num_classes = 10
    elif path == 'yelp':
        print('\nData: yelp!!!\n')
        num_classes = 2
    elif path == 'yelp_full':
        print('\nData: yelp multi class!!!\n')
        num_classes = 5
    else:
        print('\nData: ag_news!!!\n')
        num_classes = 4

    train(train_x, train_y, test_x, test_y, embedding_matrix, vocab_encoded_length, num_classes)

if __name__ == '__main__':
    tf.app.run()
