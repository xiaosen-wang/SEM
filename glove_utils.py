# coding: utf-8

import numpy as np
import pickle

def loadGloveModel(gloveFile):
    """
    Load the glove model.
    """
    print ("Loading Glove Model")
    f = open(gloveFile,'r', encoding='utf-8')
    model = {}
    for line in f:
        row = line.strip().split(' ')
        word = row[0]
        #print(word)
        embedding = np.array([float(val) for val in row[1:]])
        model[word] = embedding
    print ("Done.",len(model)," words loaded!")
    return model

def save_glove_to_pickle(glove_model, file_name):
    with open(file_name, 'wb', encoding='utf-8') as f:
        pickle.dump(glove_model, f)
        
def load_glove_from_pickle(file_name):
    with open(file_name, 'rb') as f:
        return pickle.load(f)

def create_embeddings_matrix(glove_model, dictionary, full_dictionary=None, d=300):
    MAX_VOCAB_SIZE = len(dictionary)
    # Matrix size is 300
    embedding_matrix = np.zeros(shape=((d, MAX_VOCAB_SIZE+1)))
    cnt  = 0
    unfound = []
    
    tmp = 0
    for w, i in dictionary.items():
        if not w in glove_model:
            cnt += 1
            #if cnt < 10:
            # embedding_matrix[:,i] = glove_model['UNK']
            unfound.append(i)
        else:
            embedding_matrix[:, i] = glove_model[w]
    print('Number of not found words = ', cnt)
    return embedding_matrix, unfound


def pick_most_similar_words_old(src_word, dist_mat, ret_count=10, threshold=None):
    """
    embeddings is a matrix with (d, vocab_size)
    """
    dist_order = np.argsort(dist_mat[src_word,:])[1:1+ret_count]
    dist_list = dist_mat[src_word][dist_order]
    if dist_list[-1] == 0:
        return [], []
    mask = np.ones_like(dist_list)
    if threshold is not None:
        mask = np.where(dist_list < threshold)
        return dist_order[mask], dist_list[mask]
    else:
        return dist_order, dist_list


def pick_most_similar_words(src_word, small_dist_mat, ret_count=10, threshold=None):
    """
    embeddings is a matrix with (d, vocab_size)
    """
    # dist_order = np.argsort(small_dist_mat[src_word,:])[1:1+ret_count]

    dist_order = small_dist_mat[src_word, :, 0]
    dist_list = small_dist_mat[src_word, :, 1]
    n_return = np.sum(dist_order > 0)
    dist_order, dist_list = dist_order[:n_return], dist_list[:n_return]

    dist_order, dist_list = dist_order[:ret_count], dist_list[:ret_count]
    
    mask = np.ones_like(dist_list)
    if threshold is not None:
        mask = np.where(dist_list < threshold)
        dist_order, dist_list = dist_order[mask], dist_list[mask]
    
    mask1 = np.where(dist_order <= 50000)
    dist_order, dist_list = dist_order[mask1], dist_list[mask1]
    return dist_order, dist_list


def create_small_embedding_matrix(dist_mat, MAX_VOCAB_SIZE, threshold=1.5, retain_num=50):
    """
    the memory optimized method of method `create_embeddings_matrix()`
    """
    # Matrix size is 300
    small_embedding_matrix = np.zeros(shape=((MAX_VOCAB_SIZE+1, retain_num, 2)))
    
    for i in range(MAX_VOCAB_SIZE+1):
        if i % 1000 == 0:
            print('%d/%d processed.' % (i, MAX_VOCAB_SIZE))
        dist_order = np.argsort(dist_mat[i,:])[1:1+retain_num]
        dist_list = dist_mat[i][dist_order]

        mask = np.ones_like(dist_list)
        if threshold is not None:
            mask = np.where(dist_list < threshold)
            dist_order, dist_list = dist_order[mask], dist_list[mask]
        n_return = len(dist_order)
        dist_order_arr = np.pad(dist_order, (0, retain_num-n_return), 'constant', constant_values=(-1, -1))
        dist_list_arr = np.pad(dist_list, (0, retain_num-n_return), 'constant', constant_values=(-1, -1))

        small_embedding_matrix[i, :, 0] = dist_order_arr
        small_embedding_matrix[i, :, 1] = dist_list_arr

    return small_embedding_matrix




