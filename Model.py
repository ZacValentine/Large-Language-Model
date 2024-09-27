import torch
import torch.nn as nn
import torch.nn.functional as F

from Decoder import Decoder
from PositionalEncoding import PositionalEncoding

class Model(nn.Module):
    def __init__(self, embed_dim, num_blocks, num_heads, ff_dim, dropout_rate, batch_size, vocab_size):
        super(Model, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        # self.positional_encoding = PositionalEncoding(embed_dim)
        self.positional_encoding = nn.Embedding(512, embed_dim) #< 512 is max length, add hyperparamter from config later
        self.decoder = Decoder(embed_dim, num_blocks, num_heads, ff_dim, dropout_rate, batch_size)
        self.linear = nn.Linear(embed_dim, vocab_size)
        self.dropout = nn.Dropout(dropout_rate)

        self.dropout2 = nn.Dropout(dropout_rate)                                                                                                                        # NEW HERE
        
    def forward(self, sequence, key_padding_mask, causal_mask):
        sequence = self.embedding(sequence)
        # sequence = self.positional_encoding(sequence)
        sequence += self.positional_encoding(torch.arange(0, sequence.shape[1], dtype = torch.long, device = sequence.device).unsqueeze(0))                                                  # NEW HERE
        sequence = self.dropout(sequence)

        sequence = self.decoder(sequence, key_padding_mask, causal_mask)
        logits = self.linear(sequence)
        probs = F.softmax(logits, dim=-1)

        return logits, probs