import torch
import torch.nn as nn


class RecurrentPhraseEncoder(nn.Module):

    def __init__(self, word_embedding_dim, feature_dim):
        super().__init__()
        self.feature_dim = feature_dim
        self.gru = nn.GRU(input_size=word_embedding_dim,
                          hidden_size=feature_dim // 2,
                          num_layers=1,
                          batch_first=True,
                          bidirectional=True)


    def forward(self, phrase):
        batchsize = phrase.size(0)
        h0 = torch.zeros(2, batchsize, self.feature_dim // 2)
        if torch.cuda.is_available():
            h0 = h0.cuda()
        output, hn = self.gru(phrase, h0)
        return output[:, -1, :]
   
