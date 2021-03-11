import re

import torch
from torch.utils.data.dataset import Dataset
from torch.nn.utils.rnn import pad_sequence

def preprocess_data():
    


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
