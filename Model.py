import torch.nn as nn
import torch.nn.functional as F

from Decoder import Decoder
from PositionalEncoding import PositionalEncoding

class Model(nn.Module):
    def __init__(self, embed_dim, num_blocks, num_heads, ff_dim, batch_size, vocab_size):
        super(Model, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.positional_encoding = PositionalEncoding(embed_dim)
        self.decoder = Decoder(embed_dim, num_blocks, num_heads, ff_dim, batch_size)
        self.linear = nn.Linear(embed_dim, vocab_size)
        
    def forward(self, sequence, key_padding_mask, causal_mask):
        #positional encoding here. Will be getting the tokenized data from the dataloader,
        #and the key padding mask and causal mask from datalaoder too
        sequence = self.embedding(sequence)
        sequence = self.positional_encoding(sequence)

        sequence = self.decoder(sequence, key_padding_mask, causal_mask)
        logits = self.linear(sequence)
        probs = F.softmax(logits, dim=-1)

        return logits, probs