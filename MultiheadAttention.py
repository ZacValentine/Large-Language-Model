import torch
import torch.nn as nn
import math

class MultiheadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout_rate, batch_size):
        super(MultiheadAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.batch_size = batch_size
        self.d_qkv = embed_dim//num_heads

        self.w_q = nn.Linear(embed_dim, embed_dim)
        self.w_k = nn.Linear(embed_dim, embed_dim)
        self.w_v = nn.Linear(embed_dim, embed_dim)
        self.w_out = nn.Linear(embed_dim, embed_dim)

        self.dropout1 = nn.Dropout(dropout_rate)                                                                                   # NEW HERE
        self.dropout2 = nn.Dropout(dropout_rate)                                                                                                               # NEW HERE

        self.last_attention = None

    def forward(self, q, k, v, key_padding_mask=None, causal_mask=None):
        '''
        multi head attention operation
        
        input:
        q, k, v: batch_size, seq_length, embed_dim
        key_padding_mask(float): batch_size, num_heads, seq_length, seq_length
        causal_mask(float): batch_size, num_heads, seq_length, seq_length 

        output:
        attention: batch_size, seq_length, embed_dim
        '''
        if key_padding_mask is None:
            key_padding_mask = torch.zeros((1, 1, k.shape[1], k.shape[1]), dtype=torch.float32).to(q.device)
        if causal_mask is None:
            causal_mask = torch.zeros((1, 1, q.shape[1], q.shape[1]), dtype=torch.float32).to(q.device)

        # project to qkv
        #qkv: batch_size, seq_length, embed_dim
        q = self.w_q(q)
        k = self.w_k(k)
        v = self.w_v(v)
    
        # split into each head's projections
        #qkv: batch_size, seq_length, num_heads, d_qkv
        q = q.view(self.batch_size, -1, self.num_heads, self.d_qkv)
        k = k.view(self.batch_size, -1, self.num_heads, self.d_qkv)
        v = v.view(self.batch_size, -1, self.num_heads, self.d_qkv)

        # switch num_heads, seq_length for attention operation
        #qkv: batch_size, num_heads, seq_length, d_qkv
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # attention operation
        # attention: batch_size, num_heads, seq_length, d_qkv
        # matmul q, k
        attention = torch.matmul(q, k.transpose(2, 3))
        # scale
        attention = attention / math.sqrt(self.d_qkv)
        # mask
        attention = attention + key_padding_mask + causal_mask
        # softmax
        attention = torch.softmax(attention, dim=-1)

        attention = self.dropout1(attention)                                               # NEW HERE

        # save attention matrix
        self.last_attention = attention.clone().detach() # may not need clone
        # matmul attention, v
        attention = torch.matmul(attention, v)

        # concat heads
        # batch_size, seq_length, embed_dim
        # switch seq_length, num_heads for concat
        attention = attention.transpose(1, 2)
        # concat d_qkv along num_heads to embed_dim
        attention = attention.contiguous().view(self.batch_size, -1, self.embed_dim)

        # linear out
        attention = self.w_out(attention)

        attention = self.dropout2(attention)                                                                                                              # NEW HERE
        
        return attention
