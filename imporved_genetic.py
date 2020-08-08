# coding: utf-8
#! /usr/bin/env python

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

from scipy.sparse import dok_matrix
from collections import defaultdict

# import mxnet as mx

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

import re

import attack_utils
import encode_utils


FLAGS = attack_utils.FLAGS

################################################################
# Config the logger
import logging

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s: - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S')

# Output the log to the file
# fh = logging.FileHandler('log/our_%s_%s_%d_%s.log' % (FLAGS.pre, FLAGS.data, FLAGS.sn, FLAGS.sigma))
fh = logging.FileHandler('log/ours_%s.log' % FLAGS.log_name)
fh.setLevel(logging.DEBUG)
fh.setFormatter(formatter)

# Output the log to the screen using StreamHandler
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
ch.setFormatter(formatter)

# Add two handler
logger.addHandler(ch)
logger.addHandler(fh)
# logger.info('this is info message')

logger.info('******************************\n\n\n******************************')
################################################################


VOCAB_SIZE = attack_utils.MAX_VOCAB_SIZE
MAX_ITER_NUM = 20   # The maximum number of iteration.
TOP_N = 4   # The number of the synonyms
PATH = FLAGS.data

nth_split = FLAGS.nth_split

# To be convenient, we use the objects in the `attack_utils`
attack_dict = attack_utils.org_dic
attack_inv_dict = attack_utils.org_inv_dic
attack_encode_dict = attack_utils.enc_dic
# attack_embed_mat = None
attack_dist_mat = dist_mat = np.load(('aux_files/small_dist_counter_%s_%d.npy' %(FLAGS.data, VOCAB_SIZE)))


FITNESS_W = 0.5 # default to 0.5
CROSSOVER_COEFF = 0.5   # default to 0.5
VARIATION_COEFF = 0.01   # default to 0.01
POP_MAXLEN = 60 # set the pop size to 60, which is the same to the paper



def clean_str(string):
    """
    Reuse the function in the `se_model`
    """
    return encode_utils.clean_str(string)


def read_text(path):
    """
    Reuse the function in the `encode_utils`
    """
    return encode_utils.read_text(path)


def pick_most_similar_words(src_word, dist_mat, ret_count=10, threshold=None):
    """
    Reuse the function in the `glove_utils`
    """
    return glove_utils.pick_most_similar_words(src_word, dist_mat, ret_count, threshold)


def CalculateTheDifferenceRatio(x1,x2):
    """
    Calculate the difference of two sentences
    """
    x1=x1.split(" ")
    x2=x2.split(" ")
    a=0
    b=0
    short=(len(x1) if(len(x1)<=len(x2)) else len(x2))
    for i in range(short):
        if(x1[i]!=x2[i]):
            a+=1
        b+=1
    ratio=a/b
    return a, ratio


def FindTheDifferenceWord(x1,x2):
    """
    If two words in the same position are different, print them!
    """
    x1=x1.split(" ")
    x2=x2.split(" ")
    short=(len(x1) if(len(x1)<=len(x2)) else len(x2))
    a = 0
    logger.info('The length of two words: %d vs %d.' % (len(x1), len(x2)))
    for i in range(short):
        if(x1[i]!=x2[i]):
            logger.info('%d-th word: %s(%d) vs %s(%d).' % (i, x1[i], attack_encode_dict[x1[i]], x2[i], attack_encode_dict[x2[i]]))


def replaceword(x,position,word):
    """Replace the word in `position`"""
    x=x.split(' ')
    x_new=x
    x_new[position]=word
    x_new=' '.join(x_new)
    return x_new


def FindSynonyms(search_word, M=4):
    """
    Select `M` words of the `search_word`.
    """
    search_word=attack_dict[search_word]
    nearest, nearest_dist=pick_most_similar_words(search_word,attack_dist_mat,8,0.5)
    near=[]
    for word in nearest:
        near.append(attack_inv_dict[word])
    if len(near) >= M:
        near = near[:M]
    return near


def FindBestReplace(sentence,position,near,original_label,N=2):
    result_sentences=[]
    score_list=[]
    sentence_list=[]
    for word in near:
        new_sentence=replaceword(sentence,position,word)
        new_classification,new_score=attack_utils.calculate_clf_score(new_sentence)
        score_list.append(1-new_score[0][original_label])
        # if target==1:
        #     score_list.append(new_score[0])
        # else:
        #     score_list.append(1-new_score[0])
        sentence_list.append(new_sentence)
    if(len(near)<2):
        result_sentences=sentence_list
    else:
        best_score_list=[]
        for i in range(N):
            best_score_list.append(score_list.index(max(score_list)))
            score_list[score_list.index(max(score_list))] = float('-inf')
        for j in best_score_list:
            result_sentences.append(sentence_list[j])
    return result_sentences


def Generate_seed_population(x, M=2):
    """
    Generate the seed population
    Args:
        M: the restrictive conditions for the words to be replaced
    """
    seed_population=[]
    classification,score=attack_utils.calculate_clf_score(x)
    original_label = classification[0]
    x_list=x.split(' ')
    for i in range(len(x_list)):
        sear_word=x_list[i]
        if len(sear_word)>=M and sear_word in attack_dict and attack_dict[sear_word] > 27:
            near=FindSynonyms(sear_word)
            result_sentences=FindBestReplace(x,i,near,original_label)
            seed_population.extend(result_sentences)
    return seed_population


def JudgeAdv(pop, original_label,thresold=0.6):
    """
    Judge if there exists some adversarial samples in the population
    """
    for sentence in pop:
        classification,score=attack_utils.calculate_clf_score(sentence)

        classification, score = classification[0], score[0][classification[0]]

        if classification!=original_label:
            print('There is adversarial samples, be other = %.3f, score = %.3f' % (classification, score))
            return sentence
    return None


def FitnessFunction(new_sentence,old_sentence,original_label,a=0.5):
    """
    The fitness function for the population
    """
    classification,score=attack_utils.calculate_clf_score(new_sentence)
    score1=0

    # The smaller of the original score of the output, the higher of the score we get.
    score1 = 1 - score[0][original_label]
    _, score2=CalculateTheDifferenceRatio(new_sentence,old_sentence)
    all_score=a*score1+(1-a)*score2
    return all_score


def select_high_fitness(old_sentence,pop,original_label,pop_max_size=60):
    all_score_list=[]

    if len(pop)<=pop_max_size:
        return pop
    for new_sentence in pop:
        all_score=FitnessFunction(new_sentence,old_sentence,original_label,a=FITNESS_W)
        all_score_list.append(all_score)
    best_allscore_list=[]
    for i in range(pop_max_size):
        best_allscore_list.append(all_score_list.index(max(all_score_list)))
        all_score_list[all_score_list.index(max(all_score_list))]= float('inf')
    new_pop=[]
    for score_index in best_allscore_list:
        new_pop.append(pop[score_index])
    return new_pop


def Crossover(pop,Cross_coefficient=0.5):
    """Cross Over"""
    for i in range(len(pop)):
        temp=pop[i]
        pop[i]=temp.split(' ')
    if len(pop) <= 2:
        return pop
    new_pop=pop.copy()
    for i in range(len(pop)):
        if np.random.randn()<Cross_coefficient:
            j=random.randint(1,len(pop)-1)
            k=random.randint(0,len(pop[i])-1)
            new_pop[i]=pop[i][0:k]+pop[j][k:len(pop[j])]
    for i in range(len(new_pop)):
        new_pop[i]=' '.join(new_pop[i])
    return new_pop


def  Variation(pop,original_label,Variation_coefficient=0.01,M=2):
    """Variation of the population"""
    for i in range(len(pop)):
        temp=pop[i]
        pop[i]=temp.split(' ')
    new_pop=[]
    for sentence in pop:
        if np.random.randn()<Variation_coefficient:
            j=random.randint(0,len(sentence)-1)
            if len(sentence[j])>M and sentence[j] in attack_dict and attack_dict[sentence[j]] > 27:
                near=FindSynonyms(sentence[j])
                sentence_temp=' '.join(sentence)
                result_sentences=FindBestReplace(sentence_temp,j,near,original_label)
                new_pop.extend(result_sentences)
            else:
                sentence=' '.join(sentence)
                new_pop.append(sentence)
        else:
            sentence=' '.join(sentence)
            new_pop.append(sentence)
    return new_pop


def attacksingle(sentence,iterations_num=MAX_ITER_NUM,pop_max_size=60, seq=0):
    """attack single words"""
    classification, score = attack_utils.calculate_clf_score(sentence)
    original_label = classification[0]
    logger.info('Attacked samples, classification = %.3f, score = %.3f' % (original_label, score[0][original_label]))
    # target=1
    # if classification==[1]:
    #     target=0
    seed_population=Generate_seed_population(sentence)
    pop=seed_population
    find_it = False
    for i in range(iterations_num):
        logger.info("%d-th iteration"%i)
        start = time.clock()
        adv_sentence=JudgeAdv(pop,original_label)
        if adv_sentence:
            find_it = True
            sentence = adv_sentence
            break
        pop=select_high_fitness(sentence,pop,original_label,pop_max_size=POP_MAXLEN)
        pop=Crossover(pop, Cross_coefficient=CROSSOVER_COEFF)
        pop=Variation(pop,original_label,Variation_coefficient=VARIATION_COEFF)
        end = time.clock()
        logger.info('%d-th iteration costs time %.2fs' % (i, end-start))

    # if FLAGS.gen_adv:
    #     print('>>>>>> trans_yahoo/%s/%s/ours/%d_%d.txt saved.' % (FLAGS.tr_type, FLAGS.nn_type, seq+1, original_label))
    #     with open('trans_yahoo/%s/%s/ours/%d_%d.txt' % (FLAGS.tr_type, FLAGS.nn_type, seq+1, original_label), 'wt') as fo:
    #         fo.write(sentence)

    if find_it:
        return sentence

    return None


def sample(sentences, labels):
    sentences = np.array(sentences)
    labels = np.array(labels)

    np.random.seed(0)
    shuffled_idx = np.array([i for i in range(len(sentences))])
    np.random.shuffle(shuffled_idx)

    return list(sentences[shuffled_idx]), list(labels[shuffled_idx]), shuffled_idx


TEST_SIZE = 1000
def attack_main():
    x_sentences, y_label = read_text('%s/test' % FLAGS.data)
    x_sentences, y_label, sampled_idx = sample(x_sentences, y_label)
    print('sampled indexes(top 30): ')
    print(sampled_idx[:30])
    print('-------------------------')

    all_sentences_nums=0
    successful_attack_nums=0
    change_ratio=0

    for i, (idx, sentence) in enumerate(zip(sampled_idx, x_sentences)):
        sentence = str(sentence)
        x_len = len(sentence.split())
        # if idx not in common_set:
        #     continue
        if x_len >= 100 or x_len <= 10:
            continue
        classification, score = attack_utils.calculate_clf_score(sentence)
        if y_label[i] != classification[0]:
            print('Error.................. for %d' % idx)
            continue
        logger.info("%d/%d attacking..." % (all_sentences_nums, i+1))
        all_sentences_nums+=1
        adv_sentence=attacksingle(sentence, iterations_num=20, pop_max_size=POP_MAXLEN, seq=i+1)
        if adv_sentence:
            curr_change_num, curr_change_ratio, new_adv_sentence = attack_utils.get_show_diff(sentence, adv_sentence)
            if curr_change_ratio < 0.25:
                successful_attack_nums+=1
                logger.info("original sentence: %s " % sentence)
                logger.info("adversarial sentence: %s" % new_adv_sentence)
                if FLAGS.pre=='org' and FLAGS.gen_adv:
                    attack_utils.save_adv_samples(sentence, new_adv_sentence, 'adv_samples/%s/ours/%s/%d_%d_%d.txt' % (FLAGS.data, FLAGS.nn_type, idx, y_label[i], classification[0]))
                change_ratio += curr_change_ratio
                logger.info("Current change number: %d, change ratio: %.3f" % (curr_change_num, curr_change_ratio))
                logger.info("Current mean change ratio: %.3f" % (change_ratio/successful_attack_nums))
        logger.info("Current attack success rate: %.3f" % (successful_attack_nums/all_sentences_nums))
        if successful_attack_nums >= TEST_SIZE:
            break
    successful_attack_ratio=successful_attack_nums/all_sentences_nums
    logger.info("Total attack success rate: %.3f" % successful_attack_ratio)
    logger.info("Total mean change ratio: %.3f " % (change_ratio/successful_attack_nums))


attack_main()

