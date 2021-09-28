import matplotlib.pyplot as plt
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
