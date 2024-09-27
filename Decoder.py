import torch.nn as nn

from DecoderBlock import DecoderBlock

class Decoder(nn.Module):
    def __init__(self, embed_dim, num_blocks, num_heads, ff_dim, dropout_rate, batch_size):
        super(Decoder, self).__init__()
        self.blocks = nn.ModuleList([DecoderBlock(embed_dim, num_heads, ff_dim, dropout_rate, batch_size) for _ in range(num_blocks)])
        
    def forward(self, sequence, key_padding_mask, causal_mask):
        for block in self.blocks:
            sequence = block(sequence, key_padding_mask, causal_mask)
        return sequence

