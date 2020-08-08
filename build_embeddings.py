# coding: utf-8

import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import pickle

import os
import re
from collections import Counter
import tensorflow as tf


import glove_utils
# import synonyms_encode
import encode_utils


tf.flags.DEFINE_string("nn_type", "textrnn", "The type of neural network type (default: textcnn)")
tf.flags.DEFINE_string("data", "aclImdb", "The type of data (aclImdb, yahoo_answers, ag_news)")
tf.flags.DEFINE_integer("sn", 10, "The number of the synonyms that use the same code, default to 5")
tf.flags.DEFINE_string("gpu", "0", "gpu to use")
tf.flags.DEFINE_string("sigma", "1.0", "sigma to use") 

FLAGS = tf.flags.FLAGS


DATA_TYPE = FLAGS.data
MAX_VOCAB_SIZE = 50000
GLOVE_PATH = 'glove.840B.300d.txt'
SN = FLAGS.sn
SIGMA = eval(FLAGS.sigma)

if not os.path.exists('aux_files'):
	os.mkdir('aux_files')


# Obtain the dictionary before encoding.
org_dic, org_inv_dic, tokenizer = None, None, None
if not os.path.isfile('aux_files/org_dic_%s_%d.pkl' %(DATA_TYPE, MAX_VOCAB_SIZE)):
    print('org_dic and org_inv_dic already exists, build and save the dict...')
    org_dic, org_inv_dic, tokenizer = encode_utils.build_dict(DATA_TYPE, MAX_VOCAB_SIZE)
    with open(('aux_files/org_dic_%s_%d.pkl' % (DATA_TYPE, MAX_VOCAB_SIZE)), 'wb') as f:
        pickle.dump(org_dic, f, protocol=4)
    with open(('aux_files/org_inv_dic_%s_%d.pkl' % (DATA_TYPE, MAX_VOCAB_SIZE)), 'wb') as f:
        pickle.dump(org_inv_dic, f, protocol=4)
    with open(('aux_files/tokenizer_%s_%d.pkl' % (DATA_TYPE, MAX_VOCAB_SIZE)), 'wb') as f:
        pickle.dump(tokenizer, f, protocol=4)
else:
    print('org_dic and org_inv_dic already exists, load the dict and tokenizer file...')
    with open(('aux_files/org_dic_%s_%d.pkl' % (DATA_TYPE, MAX_VOCAB_SIZE)), 'rb') as f:
        org_dic = pickle.load(f)
    with open(('aux_files/org_inv_dic_%s_%d.pkl' % (DATA_TYPE, MAX_VOCAB_SIZE)), 'rb') as f:
        org_inv_dic = pickle.load(f)
    with open(('aux_files/tokenizer_%s_%d.pkl' % (DATA_TYPE, MAX_VOCAB_SIZE)), 'rb') as f:
        tokenizer = pickle.load(f)

# Calculate the distance matrix
if not os.path.isfile('aux_files/dist_counter_%s_%d.npy' %(DATA_TYPE, MAX_VOCAB_SIZE)):
    print('dist_counter not exists, create and save...')
    dist_mat = encode_utils.compute_dist_matrix(org_dic, DATA_TYPE, MAX_VOCAB_SIZE)
    np.save(('aux_files/dist_counter_%s_%d.npy' %(DATA_TYPE, MAX_VOCAB_SIZE)), dist_mat)
else:
    print('dist_counter exists, loading...')
    dist_mat = np.load(('aux_files/dist_counter_%s_%d.npy' %(DATA_TYPE, MAX_VOCAB_SIZE)))


if not os.path.isfile('aux_files/embeddings_glove_%s_%d.npy' % (DATA_TYPE, MAX_VOCAB_SIZE)):
    print('embeddings_glove not exists, creating')
    glove_model = glove_utils.loadGloveModel('glove.840B.300d.txt')
    glove_embeddings, _ = glove_utils.create_embeddings_matrix(glove_model, org_dic)
    np.save('aux_files/embeddings_glove_%s_%d.npy' % (DATA_TYPE, MAX_VOCAB_SIZE), glove_embeddings)


# Optimize the memory usage by creating a smaller distance matrix using the embedding amtrix created above.
if not os.path.isfile('aux_files/small_dist_counter_%s_%d.npy' %(DATA_TYPE, MAX_VOCAB_SIZE)):
    print('small dist matrix not exists, create and save...')
    small_dist_mat = glove_utils.create_small_embedding_matrix(dist_mat, MAX_VOCAB_SIZE, threshold=1.5, retain_num=50)
    np.save(('aux_files/small_dist_counter_%s_%d.npy' %(DATA_TYPE, MAX_VOCAB_SIZE)), small_dist_mat)
else:
    print('small dist matrix exists, loading...')
    small_dist_mat = np.load(('aux_files/small_dist_counter_%s_%d.npy' %(DATA_TYPE, MAX_VOCAB_SIZE)))


# Test the synonyms
print('Test synonyms:')
# encode_utils.synonyms_test(org_dic, org_inv_dic, small_dist_mat, 'good', sigma=0.5)
# encode_utils.synonyms_test(org_dic, org_inv_dic, small_dist_mat, 'friday', sigma=0.5)

# Encode the original dictionary.
enc_dic = None
if not os.path.isfile(('aux_files/enc_dic_%s_%d_%d_%s.pkl' % (DATA_TYPE, MAX_VOCAB_SIZE, SN, str(SIGMA)))):
    print('enc_dic not exists, creating and saving...')
    enc_dic, enc_length = encode_utils.synonyms_encode_v1(org_dic, org_inv_dic, dist_mat, SN, SIGMA)
    print('The lenth of encoded dict: %d ' % len(enc_dic))
    print('The lenth of encoded dict: %d' % enc_length)
    with open(('aux_files/enc_dic_%s_%d_%d_%s.pkl' % (DATA_TYPE, MAX_VOCAB_SIZE, SN, str(SIGMA))), 'wb') as f:
        pickle.dump(enc_dic, f, protocol=4)
else:
    pass

