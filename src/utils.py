import matplotlib.pyplot as plt
from wordcloud import WordCloud

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