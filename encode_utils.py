# coding: utf-8

import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import pickle

import os
import re
from collections import Counter

from collections import defaultdict

import csv

import glove_utils

#################################
def compute_dist_matrix(dic, data_type='aclImdb', vocab_size=50000):
    """
    Caculate the distance of the words
    """
    embedding_matrix, missed = None, None
    if not os.path.isfile(('aux_files/embeddings_counter_%s_%d.npy' %(data_type, vocab_size))):
        # Load the glove model
        glove_tmp = glove_utils.loadGloveModel('counter-fitted-vectors.txt')
        # Caculate the embedding matrix using the dictionary
        embedding_matrix, missed = glove_utils.create_embeddings_matrix(glove_tmp, dic)
        # Save file to disk
        np.save(('aux_files/embeddings_counter_%s_%d.npy' %(data_type, vocab_size)), embedding_matrix)
        np.save(('aux_files/missed_embeddings_counter_%s_%d.npy' %(data_type,vocab_size)), missed)
    if embedding_matrix is None:
        embedding_matrix = np.load(('aux_files/embeddings_counter_%s_%d.npy' %(data_type,vocab_size)))
        missed = np.load(('aux_files/missed_embeddings_counter_%s_%d.npy' %(data_type, vocab_size)))
    # Calculate the distance of the words according to the embedding matrix of them
    c_ = -2*np.dot(embedding_matrix.T , embedding_matrix)
    a = np.sum(np.square(embedding_matrix), axis=0).reshape((1,-1))
    b = a.T
    dist = a+b+c_

    dist[0,:] = 100000
    dist[:,0] = 100000

    # Save file to disk
    np.save(('aux_files/dist_counter_%s_%d.npy' %(data_type, vocab_size)), dist)

    return dist


def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()


def helper_name(x):
    name = x.split('/')[-1]
    return int(name.split('_')[0])


def read_text(path):
    print('reading path: %s' % path)
    """ 
    Returns a list of text texts and a list of their labels
    """
    if path.startswith('aclImdb'):
        # Read dataset of aclImdb
        pos_list = []
        neg_list = []
        
        pos_path = path + '/pos'
        neg_path = path + '/neg'
        pos_files = [pos_path + '/' + x for x in os.listdir(pos_path) if x.endswith('.txt')]
        neg_files = [neg_path + '/' + x for x in os.listdir(neg_path) if x.endswith('.txt')]

        pos_files = sorted(pos_files, key=lambda x : helper_name(x))
        neg_files = sorted(neg_files, key=lambda x : helper_name(x))

        pos_list = [open(x, 'r', encoding='utf-8').read().lower().strip() for x in pos_files]
        neg_list = [open(x, 'r', encoding='utf-8').read().lower().strip() for x in neg_files]
        text_list = pos_list + neg_list
        # clean the texts
        text_list = [clean_str(s) for s in text_list]
        labels_list = [1]*len(pos_list) + [0]*len(neg_list)
        return text_list, labels_list
    else:
        # Read dataset of other datasets
        labels_list = []
        text_list = []
        with open('%s.csv' % path, 'r', encoding='utf-8') as csvfile:
            csv_reader = csv.reader(csvfile, delimiter=',')
            count = 0
            for row in csv_reader:
                count += 1
                if count % 1000 == 0:
                    print('%d records loaded.\r' % count, end='')
                labels_list.append(int(row[0]) - 1)
                text = ' // '.join(row[1:])
                text_list.append(clean_str(text).lower().strip())

        return text_list, labels_list


def build_dict(data_type='aclImdb', vocab_size=50000):
    tokenizer = Tokenizer()
    train_text, train_labels = read_text(data_type+'/train')

    tokenizer.fit_on_texts(train_text)

    dic = dict()
    dic['UNK'] = vocab_size
    inv_dict = dict()
    inv_dict[vocab_size] = 'UNK'

    for word, idx in tokenizer.word_index.items():
        if idx < vocab_size:
            inv_dict[idx] = word
            dic[word] = idx

    if 0 not in inv_dict:
        inv_dict[0] = ''
    return dic, inv_dict, tokenizer


def synonyms_test(org_dic, org_inv_dic, dist_mat, src_word, sigma=1.0):
    """
    Test the synonyms for a word
    """
    # src_word = self.dict['good']
    neighbours = find_synonyms(org_dic, org_inv_dic, dist_mat, src_word, sigma=sigma)
    print('Closest words to `%s` are :' % src_word)
    print(neighbours)


# Choose `M` synonyms
def find_synonyms(org_dic, org_inv_dic, dist_mat, search_word, M=10, sigma=0.5):
    # search_word: 'good'
    search_word = org_dic[search_word]  # Get the id of the word.
    nearest_ids,nearest_dist = glove_utils.pick_most_similar_words_old(search_word,dist_mat,M,sigma)
    nearest_words=[]
    # print(nearest_dist)
    for word in nearest_ids:
        nearest_words.append(org_inv_dic[word])
    if(len(nearest_words) >= M):
        nearest_words = nearest_words[:M]
    return nearest_words


def get_dist(dist_mat, org_dic, word1, word2):
    id1, id2 = org_dic[word1], org_dic[word2]
    return dist_mat[id1, id2]


def synonyms_encode_v1(org_dic, org_inv_dic, dist_mat, sn=10, sigma=1.0):
    """
    Encode the synonyms
    Args:
        sn: a word and its `sn` nearest neighbors share the same code.
    """
    encode_count, encode_dict = 1, defaultdict(int)

    # Encode the words using the training set.
    for word in org_dic:
        if word in encode_dict: # If the word has been encoded.
            continue

        # For common words like "the", "a", assign to a new code.
        if len(word) == 1 or org_dic[word] < 16:
            encode_dict[word] = encode_count
            encode_count += 1
            continue

        # Find the first `sn` synonyms within the distance of 0.5
        synonyms_words = find_synonyms(org_dic, org_inv_dic, dist_mat, word, M=sn, sigma=sigma)
        has_neighbor_in_dict = False
        dist_to_neighbor = float('inf')
        # Assign the code to the word as well as it's neighbors that havn't been encoded 
        # if one of its neighbors is encoded.
        # If there exists more than one neighbors that are encoded, then use the code of its nearest neighbor.
        for neighbor in synonyms_words:
            if neighbor in encode_dict:
                dist = get_dist(dist_mat, org_dic, word, neighbor)
                if dist < dist_to_neighbor:
                    dist_to_neighbor = dist
                    encode_dict[word] = encode_dict[neighbor]
                has_neighbor_in_dict = True    

        if has_neighbor_in_dict:
             for neighbor in synonyms_words:
                if neighbor not in encode_dict:
                    encode_dict[neighbor] = encode_dict[word]
        else:
            # Use a new code if the neighbors of the word and the word itself are not in the dictionary.
            encode_dict[word] = encode_count
            for neighbor in synonyms_words:
                encode_dict[neighbor] = encode_count
            encode_count += 1

    return encode_dict, encode_count


def text_encode(tokenizer, enc_dic, path, vocab_size=50000):
    """
    Encode the data using the encoded dictionary.
    """
    text, labels = read_text(path)

    seqs_o = tokenizer.texts_to_sequences(text)
    seqs_o = [[w if w < vocab_size else vocab_size for w in doc] for doc in seqs_o]

    seqs = []
    for txt in text:
        words = txt.split(' ')
        for i in range(len(words)):
            words[i] = enc_dic[words[i]]
        seqs.append(words)

    return seqs, seqs_o, labels


# Get the score from the name of the file name.
helper = lambda x : int(x.split('_')[1][:-4])

def read_adv_text(path, n_samples):
    """
    Read adversarial samples
    """
    files = [[x, helper(x)] for x in os.listdir(path) if x.endswith('.txt')][:n_samples]
    file_names, labels = zip(*files)    # tuple returned.
    text_list = [open(os.path.join(path, x), 'r', encoding='utf-8').read().lower().strip() for x in file_names]
    return text_list, list(labels)


def adv_text_encode(tokenizer, enc_dic, data_type, nn_type, vocab_size):
    """
    Process the adversarial samples generated by adversarial training
    """
    n_samples = 0
    if data_type == 'aclImdb':
        n_samples = 2500
    elif data_type == 'ag_news':
        n_samples = 12000
    elif data_type == 'yahoo_answers':
        n_samples = 140000
    else:
        print('n_samples is set to zero!!! assume the dataset is wrong!')

    text, labels = read_adv_text(os.path.join(data_type, 'adv_' + nn_type), n_samples)

    seqs_o = tokenizer.texts_to_sequences(text)
    seqs_o = [[w if w < vocab_size else vocab_size for w in doc] for doc in seqs_o]

    return seqs_o, labels
