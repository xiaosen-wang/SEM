# -*- coding: UTF-8 -*-

import tensorflow as tf
import numpy as np
import os
import time
import datetime
import data_helpers
from text_cnn import TextCNN
from tensorflow.contrib import learn
import csv
import pickle
import glove_utils
import random

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences


# Eval Parameters
tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
tf.flags.DEFINE_integer("pop_len", 60, "Batch Size (default: 200)")
tf.flags.DEFINE_string("nn_type", "textrnn", "The type of neural network type (default: textrnn)") 
tf.flags.DEFINE_string("pre", "enc", "org or enc to attack (default: org (org or enc))") 
tf.flags.DEFINE_string("gpu", "0", "gpu to use")
tf.flags.DEFINE_string("sigma", "1.0", "sigma to use") 
tf.flags.DEFINE_integer("sn", 10, "synonyms number (default: 10)")
tf.flags.DEFINE_integer("done_num", 0, "The attacked number")
tf.flags.DEFINE_boolean("gen_adv", False, "Generate adversarial samples or not.")
tf.flags.DEFINE_string("data", "aclImdb", "The type of data (aclImdb, yahoo_answers, ag_news)")
tf.flags.DEFINE_string("time", "1560756982", "The time generated.") 
tf.flags.DEFINE_string("tr_type", 'clean', "clean, def, adv")
tf.flags.DEFINE_integer("nth_split", -1, "Used when generating adversarial samples, the nth-split to run.")
tf.flags.DEFINE_string("log_name", "log", "the name of the log file.") 

# tf.flags.DEFINE_string("data", "aclImdb", "The type of data (aclImdb, yahoo_answers, ag_news)")
tf.flags.DEFINE_string("f_attack_type", "ours", "Transferability: The type of attacking: ours, ucla, pwws, greedy, used for transferability") 
tf.flags.DEFINE_string("f_nn_type", "textrnn", "Transferability: The type of network: textrnn, textcnn, textbirnn, used for transferability") 

# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")
FLAGS = tf.flags.FLAGS
FLAGS.flag_values_dict()

os.environ["CUDA_VISIBLE_DEVICES"]=FLAGS.gpu

if eval(FLAGS.sigma) == 0:
    FLAGS.pre = 'org'
    FLAGS.sigma = '0.5'


MAX_VOCAB_SIZE = 50000
# Load the dictionary after encoding.
with open(('aux_files/enc_dic_%s_%d_%d_%s.pkl' % (FLAGS.data, MAX_VOCAB_SIZE, FLAGS.sn, FLAGS.sigma)), 'rb') as f:
    enc_dic = pickle.load(f)

# Load the original dictionary
with open(('aux_files/org_dic_%s_%d.pkl' % (FLAGS.data, MAX_VOCAB_SIZE)), 'rb') as f:
    org_dic = pickle.load(f)
with open(('aux_files/org_inv_dic_%s_%d.pkl' % (FLAGS.data, MAX_VOCAB_SIZE)), 'rb') as f:
    org_inv_dic = pickle.load(f)

CHECKPOINT_DIR = './runs_%s_%s/%s/checkpoints/' % (FLAGS.pre, FLAGS.nn_type, FLAGS.time)

# reconstruct session
# get the output of the model
# ==================================================
checkpoint_file = tf.train.latest_checkpoint(CHECKPOINT_DIR)
graph = tf.Graph()
with graph.as_default():
    session_conf = tf.GPUOptions(allow_growth=True)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=session_conf))
    with sess.as_default():
        # Load the saved meta graph and restore variables
        saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
        saver.restore(sess, checkpoint_file)

        # Get the placeholders from the graph by name
        input_x = graph.get_operation_by_name("input_x").outputs[0]
        # is_train = graph.get_operation_by_name("is_train").outputs[0]
        # input_y = graph.get_operation_by_name("input_y").outputs[0]
        dropout_keep_prob = graph.get_operation_by_name("dropout_keep_prob").outputs[0]

        # text_vector = graph.get_operation_by_name("dropout/text_vector").outputs[0]
        scores=graph.get_operation_by_name("output/scores").outputs[0]
        predictions = graph.get_operation_by_name("output/predictions").outputs[0]


def calculate_clf_score(x):
    """
    The only interface of the model exposed to the attacker
    """
    if type(x) == str:
        x = [x]
    clf_list, score_list = calculate_clf_score_batch_multiclass(x)
    return clf_list, score_list


def get_show_diff(org, adv):
    """
    Get the difference of two sentences: `org` vs `adv`
    """
    org, adv = org.split(' '), adv.split(' ')
    a = b = 0
    print('The length of two words: %d vs %d.' % (len(org), len(adv)))
    short=(len(org) if(len(org)<=len(adv)) else len(adv))
    for i in range(short):
        if org[i] != adv[i]:
            if FLAGS.pre == 'enc':
                # If the codes of two words are the same, then change the word back,
                # It means that we can get the same effect if we don't change this word for the texts encoded are the same
                if enc_dic[org[i]] == enc_dic[adv[i]]:
                    adv[i] = org[i]
                else:
                    print('%d-th word: %s(%d) vs %s(%d).' % (i, org[i], enc_dic[org[i]], adv[i], enc_dic[adv[i]]))
                    a += 1
            else:
                print('%d-th word: %s(%d) vs %s(%d).' % (i, org[i], enc_dic[org[i]], adv[i], enc_dic[adv[i]]))
                a += 1
        b+=1
    ratio=a/b
    new_adv_sentence = ' '.join(adv)
    return a, ratio, new_adv_sentence


def calculate_clf_score_batch_multiclass(x, max_len=250):
    """
    Calculate the classification and the score for the input texts.
    """
    assert type(x) == list
    seqs = []
    for txt in x:
        words = txt.split(' ')
        for i in range(len(words)):
            if FLAGS.pre == 'enc':
                words[i] = enc_dic[words[i]]
            else:
                words[i] = org_dic[words[i]] if words[i] in org_dic else MAX_VOCAB_SIZE
        seqs.append(words)

    seqs = list(pad_sequences(seqs, maxlen=max_len, padding='post'))
    x_test = np.array(seqs)

    score, classification = sess.run([scores, predictions],{input_x:x_test,dropout_keep_prob:1.0})

    # [[-0.46071488  0.69902337]]
    score = list(score)

    score = list(softmax(np.array(score)))
    # print(x, 'score calculated!'), [3], [[0.1,0.3,0.5]]
    return classification, score



def softmax(x):
    """
    Arguments:
    x -- A D dimensional vector or N x D dimensional numpy matrix.
    Return:
    x -- You are allowed to modify x in-place
    """
    orig_shape = x.shape

    if len(x.shape) > 1:
        # Matrix
        _c_matrix = np.max(x, axis=1)
        _c_matrix = np.reshape(_c_matrix, [_c_matrix.shape[0], 1])
        _diff = np.exp(x - _c_matrix)
        x = _diff / np.reshape(np.sum(_diff, axis=1), [_c_matrix.shape[0], 1])
    else:
        # Vector
        _c = np.max(x)
        _diff = np.exp(x - _c)
        x = _diff / np.sum(_diff)

    assert x.shape == orig_shape
    return x

def save_adv_samples(sentence, new_adv_sentence, fname):
    with open(fname, 'wt') as fo:
        fo.write('orig: ' + sentence + '\n')
        fo.write('adv: ' + new_adv_sentence)


def save_adv_samples_1(sentence, fname):
    with open(fname, 'wt') as fo:
        fo.write(sentence)
