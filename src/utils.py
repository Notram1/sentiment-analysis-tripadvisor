import numpy as np
import matplotlib.pyplot as plt
import bcolz
import pickle

from wordcloud import WordCloud
import torch

def wordcloud_generator(data):
    wordcloud = WordCloud(width = 800, height = 800,
                          background_color ='black',
                          min_font_size = 10,
                          colormap='Paired'
                         ).generate(" ".join(data.values))
    # plot the WordCloud image                        
    plt.figure(figsize = (10, 10), facecolor = None) 
    plt.imshow(wordcloud, interpolation='bilinear') 
    plt.axis("off") 
    plt.tight_layout(pad = 0) 
    # plt.title("Most common words in Reviews",fontsize=30)
    # plt.savefig('tripadvisor_wordcloud.png') 
    plt.show() 

def calc_distr(combined_data):
    count={'0':0,'1':0,'2':0}
    for raw_text, tokens, indices, label in combined_data:
        if label==0:
            count['0']+=1
        elif label==1:
            count['1']+=1
        elif label==2:
            count['2']+=1
    sum = count['0']+count['1']+count['2']
    count['0'] = count['0']/sum
    count['1'] = count['1']/sum
    count['2'] = count['2']/sum
    return count

def parse_glove(glove_path: str, embedding_dim: int):
    words = []
    idx = 0
    word2idx = {}
    vectors = bcolz.carray(np.zeros(1), rootdir=f'{glove_path}/6B.{embedding_dim}.dat', mode='w')

    with open(f'{glove_path}/glove.6B.{embedding_dim}d.txt', 'rb') as f:
        for l in f:
            line = l.decode().split()
            word = line[0]
            words.append(word)
            word2idx[word] = idx
            idx += 1
            vect = np.array(line[1:]).astype(float)
            vectors.append(vect)
        
    vectors = bcolz.carray(vectors[1:].reshape((400000, embedding_dim)), rootdir=f'{glove_path}/6B.{embedding_dim}.dat', mode='w')
    vectors.flush()
    pickle.dump(words, open(f'{glove_path}/6B.{embedding_dim}_words.pkl', 'wb'))
    pickle.dump(word2idx, open(f'{glove_path}/6B.{embedding_dim}_idx.pkl', 'wb'))