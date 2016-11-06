# -*- coding: utf-8 -*-
"""
Created on Fri Jul 08 09:37:08 2016

@author: L
"""
import cPickle
import numpy as np
from collections import defaultdict, OrderedDict
import theano
import theano.tensor as T
import re
import warnings
import sys
reload(sys)    
sys.setdefaultencoding('utf-8')
import time
import math

Words, word_idx_map = cPickle.load(open("E:\doc\data\CNN_sentence-master\CNN_result_0","rb"))
limit = 10
count = 0
wd = Words[1:100]

for count in range(limit):
    print str(count) + "=" + str(wd[count])
'''
for k,v in word_idx_map.items():
    #print k.decode('utf-8')
    print k.decode('utf-8') + "==" + str(v)
    count = count + 1
    if count>limit:
        break
'''
idx_words_map = {}
for k,v in word_idx_map.items():
    idx_words_map[v] = k

def find_min(wordMap):
    minv = 1000
    for key,va in wordMap.items():
        #print key
        if minv>key:
            minv = key
    return minv
    
def getNearestWord(org_index,words_number):
    wordMap = {}
    wordMap = dict.fromkeys((v - words_number for v in range(int(words_number))), "")
    org_vector = Words[org_index]
    length = len(Words)-1
    min_value = -1
    for index in range(length):
        index = index + 1
        vector = Words[index]
        value = np.dot(org_vector,vector)/(math.sqrt(np.dot(org_vector,org_vector))*math.sqrt(np.dot(vector,vector)))
        if value > min_value and index!=org_index:
            del wordMap[min_value]
            wordMap[value] = idx_words_map[index]
            min_value = find_min(wordMap)
    
    return wordMap
    

words_number = 5.0
word='海军'.encode('utf-8')
print word.decode('utf-8') + "===="
if word in word_idx_map:
    index = word_idx_map[word]
    wordMap = getNearestWord(index,words_number)
    for k,v in wordMap.items():
        print k,v.decode('utf-8')
word='导弹'.encode('utf-8')
print word.decode('utf-8') + "===="
if word in word_idx_map:
    index = word_idx_map[word]
    wordMap = getNearestWord(index,words_number)
    for k,v in wordMap.items():
        print k,v.decode('utf-8')
word='药品'.encode('utf-8')
print word.decode('utf-8') + "===="
if word in word_idx_map:
    index = word_idx_map[word]
    wordMap = getNearestWord(index,words_number)
    for k,v in wordMap.items():
        print k,v.decode('utf-8')

'''
while(1):
    word = raw_input("Please input a key: ")#.encode(sys.getdefaultencoding())
    #word = raw_input("Enter your input: ")
    #word = word.encode('utf-8')
    print "start find:" + word
    if word in word_idx_map:
        index = word_idx_map[word]
        wordMap = getNearestWord(index,words_number)
        for k,v in wordMap:
            print k,v.decode('utf-8')
    else:
        print "not in word_idx_map"
'''
