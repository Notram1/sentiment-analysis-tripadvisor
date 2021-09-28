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

        # self.dropout = nn.Dropout(0.3)                
        self.fc = nn.Linear(in_features=hidden_size, out_features=additional_kwargs['output_size'])
            

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
        if lengths is not None:
            embeds = pack_padded_sequence(embeds, lengths)
        
        if self.hparams['rnn_type'] == 'lstm':
            _, (h, _) = self.rnn(embeds)
        else:
            _, h  = self.rnn(embeds)
        
        # h = self.dropout(h)
        h = h.mean(0) # calculate mean across layers
        output = self.fc(h)

        return output
