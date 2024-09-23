import torch.nn as nn
import torch.nn.functional as F

from MultiheadAttention import MultiheadAttention

class DecoderBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim, batch_size):
        super(DecoderBlock, self).__init__()
        self.attention = MultiheadAttention(embed_dim, num_heads, batch_size)

        self.norm1 = nn.LayerNorm(embed_dim) # may be wrong

        self.ff1 = nn.Linear(embed_dim, ff_dim)
        self.ff2 = nn.Linear(ff_dim, embed_dim)

        self.norm2 = nn.LayerNorm(embed_dim) # may be wrong
        
        
    def forward(self, sequence, key_padding_mask, causal_mask):
        '''
        decoder block operations
            
        input: 
        batch_size, seq_length, embed_dim

        output: 
        same shape
        '''
        # masked self attention
        # sequence: same shape
        residual1 = sequence.clone()
        sequence = self.attention(sequence, sequence, sequence, key_padding_mask, causal_mask)
        # add & norm 1
        # sequence: same shape
        sequence = self.norm1(residual1 + sequence) # residual connection may be wrong, layernorm may be wrong
        # ffn
        # same shape
        residual2 = sequence.clone()
        sequence = self.ff1(sequence)
        sequence = F.relu(sequence)
        sequence = self.ff2(sequence)
        # add & norm 2
        sequence = self.norm2(residual2 + sequence) # residual connection may be wrong, layernorm may be wrong

        return sequence


