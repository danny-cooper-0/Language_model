import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torchmetrics import F1Score
import torch.optim as optim
from torch.optim import Adam
from transformers import GPT2Tokenizer, AdamW

# Positional Encoding
class PositionalEncoding(nn.Module):
    def __init__(self, model_dim, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, model_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, model_dim, 2).float() * (-np.log(10000.0) / model_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0), :]

# Scaled Dot-Product Attention
def scaled_dot_product_attention(query, key, value):
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / np.sqrt(d_k)
    attn = torch.softmax(scores, dim=-1)
    output = torch.matmul(attn, value)
    return output, attn

# Multi-Head Attention
class MultiHeadAttention(nn.Module):
    def __init__(self, model_dim, num_heads):
        super(MultiHeadAttention, self).__init__()
        assert model_dim % num_heads == 0

        self.num_heads = num_heads
        self.head_dim = model_dim // num_heads

        self.query_linear = nn.Linear(model_dim, model_dim)
        self.key_linear = nn.Linear(model_dim, model_dim)
        self.value_linear = nn.Linear(model_dim, model_dim)
        self.out_linear = nn.Linear(model_dim, model_dim)

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.query_linear.weight)
        nn.init.xavier_uniform_(self.key_linear.weight)
        nn.init.xavier_uniform_(self.value_linear.weight)
        nn.init.xavier_uniform_(self.out_linear.weight)

    def forward(self, query, key, value):
        batch_size = query.size(0)
        
        query = self.query_linear(query).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        key = self.key_linear(key).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        value = self.value_linear(value).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

        scaled_scores, attn = scaled_dot_product_attention(query, key, value)

        scaled_scores = scaled_scores.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.head_dim)
        output = self.out_linear(scaled_scores)

        return output

# Transformer Encoder Layer
class TransformerEncoderLayer(nn.Module):
    def __init__(self, model_dim, num_heads, dim_feedforward, dropout):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(model_dim, num_heads)
        self.linear1 = nn.Linear(model_dim, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, model_dim)

        self.norm1 = nn.LayerNorm(model_dim)
        self.norm2 = nn.LayerNorm(model_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, src):
        src2 = self.self_attn(src, src, src)
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(F.gelu(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

# Transformer Decoder Layer
class TransformerDecoderLayer(nn.Module):
    def __init__(self, model_dim, num_heads, dim_feedforward, dropout):
        super(TransformerDecoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(model_dim, num_heads)
        self.multihead_attn = MultiHeadAttention(model_dim, num_heads)
        self.linear1 = nn.Linear(model_dim, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, model_dim)

        self.norm1 = nn.LayerNorm(model_dim)
        self.norm2 = nn.LayerNorm(model_dim)
        self.norm3 = nn.LayerNorm(model_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(self, tgt, memory):
        tgt2 = self.self_attn(tgt, tgt, tgt)
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        tgt2 = self.multihead_attn(tgt, memory, memory)
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(F.gelu(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt

# Full Transformer Model
class TransformerModel(nn.Module):
    def __init__(self, vocab_size, model_dim, num_heads, num_encoder_layers, num_decoder_layers, dim_feedforward, dropout=0.1):
        super(TransformerModel, self).__init__()
        self.model_dim = model_dim

        self.encoder_embedding = nn.Embedding(vocab_size, model_dim)
        self.decoder_embedding = nn.Embedding(vocab_size, model_dim)
        self.positional_encoding = PositionalEncoding(model_dim)

        self.encoder_layers = nn.ModuleList([
            TransformerEncoderLayer(model_dim, num_heads, dim_feedforward, dropout) for _ in range(num_encoder_layers)
        ])
        self.decoder_layers = nn.ModuleList([
            TransformerDecoderLayer(model_dim, num_heads, dim_feedforward, dropout) for _ in range(num_decoder_layers)
        ])

        self.fc = nn.Linear(model_dim, vocab_size)
        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.encoder_embedding.weight)
        nn.init.xavier_uniform_(self.decoder_embedding.weight)
        nn.init.xavier_uniform_(self.fc.weight)

    def forward(self, src, tgt):
        src = self.encoder_embedding(src) * (self.model_dim ** 0.5)
        tgt = self.decoder_embedding(tgt) * (self.model_dim ** 0.5)
        
        src = self.positional_encoding(src)
        tgt = self.positional_encoding(tgt)

        for encoder_layer in self.encoder_layers:
            src = encoder_layer(src)

        for decoder_layer in self.decoder_layers:
            tgt = decoder_layer(tgt, src)

        output = self.fc(tgt)
        return output

class TransformerLightningModel(pl.LightningModule):
    def __init__(self, model, learning_rate=0.001):
        super(TransformerLightningModel, self).__init__()
        self.model = model
        self.learning_rate = learning_rate
        self.criterion = torch.nn.CrossEntropyLoss()

    def forward(self, src, tgt):
        return self.model(src, tgt)

    def training_step(self, batch, batch_idx):
        x = batch['input_ids']  # Assuming input_ids are passed
        y = x.clone()  # For language modeling, targets are the same as inputs
        y_hat = self(x, x)  
        loss = self.criterion(y_hat.view(-1, y_hat.size(-1)), y.view(-1))
        self.log('train_loss', loss)
        return loss

    def configure_optimizers(self):
        return AdamW(self.model.parameters(), lr=self.learning_rate)
