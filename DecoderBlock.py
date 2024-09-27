import torch.nn as nn
import torch.nn.functional as F

from MultiheadAttention import MultiheadAttention

class DecoderBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout_rate, batch_size):
        super(DecoderBlock, self).__init__()
        self.attention = MultiheadAttention(embed_dim, num_heads, dropout_rate, batch_size)

        self.dropout1 = nn.Dropout(dropout_rate)

        self.norm1 = nn.LayerNorm(embed_dim)

        self.ff1 = nn.Linear(embed_dim, ff_dim)
        self.ff2 = nn.Linear(ff_dim, embed_dim)

        self.dropout2 = nn.Dropout(dropout_rate)

        self.norm2 = nn.LayerNorm(embed_dim)
        
        
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
        # residual1 = sequence.clone()
        # sequence = self.attention(sequence, sequence, sequence, key_padding_mask, causal_mask)
        # # dropout
        # # sequence: same shape
        # # sequence = self.dropout1(sequence)
        # # add & norm 1
        # # sequence: same shape
        # sequence = self.norm1(residual1 + sequence)
        # # ffn
        # # sequence: same shape
        # residual2 = sequence.clone()
        # sequence = self.ff1(sequence)
        # sequence = F.relu(sequence)
        # sequence = self.ff2(sequence)
        # # dropout
        # # sequence: same shape
        # sequence = self.dropout2(sequence)
        # # add & norm 2
        # # sequence: same shape
        # sequence = self.norm2(residual2 + sequence)

        residual1 = sequence.clone()
        sequence = self.norm1(sequence)
        sequence = self.attention(sequence, sequence, sequence, key_padding_mask, causal_mask)
        sequence = sequence + residual1
        residual2 = sequence.clone()
        sequence = self.norm2(sequence)
        sequence = self.ff1(sequence)
        sequence = F.relu(sequence)
        sequence = self.ff2(sequence)
        sequence = self.dropout1(sequence)
        sequence = sequence + residual2



        return sequence


