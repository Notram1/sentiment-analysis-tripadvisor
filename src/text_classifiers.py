import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence


class RNNClassifier(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, hidden_size, rnn_type, **additional_kwargs):
        """
        Inputs:
            num_embeddings: size of the vocabulary
            embedding_dim: size of an embedding vector
            hidden_size: hidden_size of the rnn layer
            rnn_type: use vanilla RNN, LSTM or GRU
        """
        super().__init__()

        # Change this if you edit arguments
        self.hparams = {
            'num_embeddings': num_embeddings,
            'embedding_dim': embedding_dim,
            'hidden_size': hidden_size,
            'rnn_type': rnn_type,
            **additional_kwargs
        }
        
        self.embedding = nn.Embedding(self.hparams['num_embeddings'], self.hparams['embedding_dim'], padding_idx=0)
        
        if rnn_type == 'lstm':
            self.rnn = nn.LSTM(input_size=self.hparams['embedding_dim'], hidden_size=hidden_size, num_layers=additional_kwargs['n_layers'], 
                            dropout=additional_kwargs['dropout'], bidirectional=additional_kwargs['bidirectional'])
        elif rnn_type == 'rnn':
            self.rnn = nn.RNN(input_size=self.hparams['embedding_dim'], hidden_size=hidden_size, num_layers=additional_kwargs['n_layers'], 
                            dropout=additional_kwargs['dropout'], bidirectional=additional_kwargs['bidirectional'])
        elif rnn_type == 'gru':
            self.rnn = nn.GRU(input_size=self.hparams['embedding_dim'], hidden_size=hidden_size, num_layers=additional_kwargs['n_layers'], 
                            dropout=additional_kwargs['dropout'], bidirectional=additional_kwargs['bidirectional'])
        else:
            raise ValueError('Unknown recurrent layer type given!')    

        self.dropout = nn.Dropout(additional_kwargs['dropout'])                
        self.fc = nn.Linear(in_features=hidden_size*2 if additional_kwargs['bidirectional'] else hidden_size, out_features=additional_kwargs['output_size'])
            

    def forward(self, sequence, lengths=None):
        """
        Inputs
            sequence: A long tensor of size (seq_len, batch_size)
            lengths: A long tensor of size batch_size, represents the actual
                sequence length of each element in the batch. If None, sequence
                lengths are identical.
        Outputs:
            output: A tensor of size (batch_size, nb_classes) represents the probabilities of being
                positive, i.e. in range (0, 1)
        """
        embeds = self.embedding(sequence)
        embeds = self.dropout(embeds)
        if lengths is not None:
            embeds = pack_padded_sequence(embeds, lengths, enforce_sorted=True)
        
        if self.hparams['rnn_type'] == 'lstm':
            _, (h, _) = self.rnn(embeds)
        else:
            _, h  = self.rnn(embeds)

        if self.rnn.bidirectional:
            h = torch.cat([h[-1,:,:], h[-2,:,:]], dim=1)           
        else:
            h = h[-1]

        h = self.dropout(h)
        return self.fc(h) 

def initialize_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        nn.init.zeros_(m.bias)
    elif isinstance(m, nn.RNN) or isinstance(m, nn.LSTM) or isinstance(m, nn.GRU):
        for name, param in m.named_parameters():
            if 'bias' in name:
                nn.init.zeros_(param)
            elif 'weight' in name:
                nn.init.orthogonal_(param)
