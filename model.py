import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_length=10):
        super().__init__()

        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class MovieEmbedding(nn.Module):
    def __init__(self):
        super().__init__()

        self.l0 = nn.Linear(146036, 509)
        # self.l1 = nn.Linear(32768, 8192)
        # self.l2 = nn.Linear(8192, 2045)

        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.l0(x))
        # x = self.relu(self.l1(x))
        # x = self.relu(self.l2(x))

        return x


class FullyConnected(nn.Module):
    def __init__(self):
        super().__init__()

        self.l0 = nn.Linear(512, 87585)
        # self.l1 = nn.Linear(8192, 32768)
        # self.l2 = nn.Linear(32768, 87585)

        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.l0(x))
        # x = self.relu(self.l1(x))
        # x = self.relu(self.l2(x))
        return x

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = MovieEmbedding()
        self.positional_encoding = PositionalEncoding(512)
        self.multihead_attn = nn.MultiheadAttention(512, 8, dropout=0.1)
        self.layer_norm = nn.LayerNorm(512)
        self.dropout = nn.Dropout(0.1)
        self.fc = FullyConnected()

    def forward(self, past_data, past_rating_data):
        past_data = self.embedding(past_data)
        x = torch.cat([past_data, past_rating_data], dim=-1)
        x = self.positional_encoding(x)


        x = x.transpose(0, 1)

        attn_output, _ = self.multihead_attn(x, x, x)
        attn_output = self.dropout(attn_output)
        x = self.layer_norm(x + attn_output)

        x = x.mean(dim=0)

        output = self.fc(x)

        return output
