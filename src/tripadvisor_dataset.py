
import re
import nltk

import torch
from torch.utils.data.dataset import Dataset
from torch.nn.utils.rnn import pad_sequence

def stopwords_removal(stopwords, sentence):
    return [word for word in nltk.word_tokenize(sentence) if word not in stopwords]

def clean_data(data, name):
    # Convert words to lowercase
    data[name]=data[name].str.lower()
    # Remove the Hashtags from the text
    data[name]=data[name].apply(lambda x:re.sub(r'\B#\S+', '',x))
    # Remove the links from the text
    data[name]=data[name].apply(lambda x:re.sub(r"http\S+", "", x))
    # Remove the twitter handlers
    data[name]=data[name].apply(lambda x:re.sub('@[^\s]+','',x))
    # Remove the Special characters from the text 
    data[name]=data[name].apply(lambda x:' '.join(re.findall(r'\w+', x)))
    # Remove all the single characters in the text
    data[name]=data[name].apply(lambda x:re.sub(r'\s+[a-zA-Z]\s+', ' ', x))
    # Substitute the multiple spaces with single spaces
    data[name]=data[name].apply(lambda x:re.sub(r'\s+', ' ', x, flags=re.I))

def stem_data(data, name, stemmer=None):
    if stemmer is None:
        return None   
    var=[]
    for review in data[name].values:
        output = [stemmer.stem(word) for word in review]

        # Remove characters which have length less than 2  
        without_single_chr = [word for word in output if len(word) >= 2]
        # Remove numbers
        cleaned_data_title = [word for word in without_single_chr if not word.isnumeric()]
        var.append(cleaned_data_title)
    data[name] = var

def lem_data(data, name, lemmatizer=None):
    if lemmatizer is None:
        return None
    var=[]
    for review in data[name].values:
        output = [lemmatizer.lemmatize(word) for word in review]

        # Remove characters which have length less than 2  
        without_single_chr = [word for word in output if len(word) >= 2]
        # Remove numbers
        cleaned_data_title = [word for word in without_single_chr if not word.isnumeric()]
        var.append(cleaned_data_title)
    data[name] = var
    

class SentimentDataset(Dataset):
    def __init__(self, data):
        """
        Inputs:
            data: list of tuples (raw_text, tokens, token_indices, label)
        """
        self.data = data
        self.data.sort(key=lambda x: len(x[1]), reverse=True)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        """
        Inputs:
            i: an integer value to index data
        Outputs:
            data: A dictionary of {data, label}
        """
        _, _, indices, label = self.data[i]
        return {
            'data': torch.tensor(indices).long(),
            'label': torch.tensor(label).float()
        }


def collate(batch):
    """
        To be passed to DataLoader as the `collate_fn` argument
    """
    assert isinstance(batch, list)
    data = pad_sequence([b['data'] for b in batch])
    lengths = torch.tensor([len(b['data']) for b in batch])
    label = torch.stack([b['label'] for b in batch])
    return {
        'data': data,
        'label': label,
        'lengths': lengths
    }
