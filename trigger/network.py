# network.py

import torch
import torch.nn as nn

class RNN(nn.Module):
    def __init__(self, vocab_size, 
                 embedding_dim, hidden_dim, output_dim,
                 num_layers,
                 is_bidirectional, 
                 dropout_rate,
                 padding_idx):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx = padding_idx)
        self.rnn = nn.LSTM(embedding_dim, hidden_dim, 
                           num_layers = num_layers,
                           bidirectional = is_bidirectional,
                           dropout = dropout_rate)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)
        self.dropout = nn.Dropout(dropout_rate)
    
    def forward(self, text, text_length):
        embedded = self.dropout(self.embedding(text))

        # Pack sequence
        packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, text_length)
        self.rnn.flatten_parameters()
        packed_output, (hidden, cell) = self.rnn(packed_embedded)
        # Unpack sequence
        output, output_length = nn.utils.rnn.pad_packed_sequence(packed_output)

        hidden = self.dropout(torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1))
        return self.fc(hidden)

