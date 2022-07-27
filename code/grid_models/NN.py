import torch
from torch import nn
import numpy as np


class HailNetMini(nn.Module):
    r"""
        HailNetMini - main model

        Inputs: tensor with dimension (<num_features(n)>, <width(x)>, <height(y)>)
                tensors contains climatic variables
                features order in info/feature_order.txt

        Parameters:
    """
    def __init__(self, n, x, y, lstm_num_layers, lin1_size=16, seq_len=42, units=None):
        super().__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.n = n
        self.x = x
        self.y = y
        self.lin1_size = lin1_size
        self.lstm_num_layers = lstm_num_layers
        self.seq_len = seq_len
        self.lin0 = nn.Linear(n * x * y, 256)
        self.lin1 = nn.Linear(256, 1024)
        self.conv2d1 = nn.Conv2d(in_channels=n, out_channels=n, kernel_size=3, stride=1, padding=0)
        self.conv2d2 = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, stride=1, padding=0)
        self.lin2 = nn.Linear(1024 + (self.long - 2) * (self.lat - 2) * self.n, 1024)
        self.lin3 = nn.Linear(1024, 1)
        self.lstm = nn.LSTM(
            input_size=1024,
            hidden_size=1024,
            num_layers=self.rnn_num_layers,
            batch_first=True
        )

    def forward(self, x):
        # x -> (n_batch, n_vars, x, y)
        h0 = torch.randn(self.rnn_num_layers, x.size(0), 1024).to(self.device)  # hidden cell for rnn
        c0 = torch.randn(self.rnn_num_layers, x.size(0), 1024).to(self.device)  # hidden cell for rnn
        hs1 = []

        for i in range(self.seq_len):
            t0 = x[:, i].float()
            t1 = self.lin0(t0.flatten(start_dim=1))
            t2 = torch.sigmoid(t1)
            t3 = self.lin1(t2)
            t4 = torch.sigmoid(t3)
            c2d1 = self.conv2d1(t0)
            h = torch.cat([t4, c2d1.flatten(start_dim=1)], dim=1)
            h = h.unsqueeze(dim=1)
            hs1.append(h)

        t = torch.cat(hs1, dim=1)
        t = self.lin2(t)
        out, _ = self.lstm(t, (h0, c0))

        out = out[:, -1, :]

        out = self.lin3(out)
        out = torch.sigmoid(out)

        return out