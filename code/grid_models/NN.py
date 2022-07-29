import torch
from torch import nn
import numpy as np
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader

class HailNetDots(nn.Module):
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

#####################################################
#                                                   #
#    Дописать и протестировать на данных ERA        #
#                                                   #
#####################################################


class HailNetGrid(nn.Module):
    r"""
        HailNetMini - main model

        Inputs: tensor with dimension (<num_features(n)>, <width(x)>, <height(y)>)
                tensors contains climatic variables
                features order in info/feature_order.txt

        Parameters:
    """
    def __init__(self, n: int, x: int, y: int, lstm_num_layers: int):
        super().__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.n = n
        self.x = x
        self.y = y
        self.lstm_num_layers = lstm_num_layers
        self.seq_len = n
        #self.lin0 = nn.Linear(n * x * y, 1024)
        self.lins0 = []
        for i in range(self.n):
            self.lins0.append(nn.Linear(self.x * self.y, self.x * self.y))
        #self.lin1 = nn.Linear(1024, 1024)
        self.lins1 = []
        for i in range(self.n):
            self.lins1.append(nn.Linear(self.x * self.y, self.x * self.y))
        self.conv2d1 = nn.Conv2d(in_channels=self.n, out_channels=self.n, kernel_size=3, stride=1, padding=1)
        self.conv2d2 = nn.Conv2d(in_channels=self.n, out_channels=self.n, kernel_size=5, stride=1, padding=2)
        #self.convs2d1 = []
        #for i in range(self.n):
        #    self.convs2d1.append(nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, stride=1, padding=1))
        #self.convs2d2 = []
        #for i in range(self.n):
        #    self.convs2d2.append(nn.Conv2d(in_channels=1, out_channels=1, kernel_size=5, stride=1, padding=2))

        self.lin2 = nn.Linear(3 * (self.x * self.y * self.n), 8192)
        self.lin3 = nn.Linear(8192, self.x * self.y)
        self.lstm = nn.LSTM(
            input_size=8192,
            hidden_size=8192,
            num_layers=self.lstm_num_layers,
            batch_first=True
        )
        self.to(self.device)

    def forward(self, x):
        # x -> (n_batch, n_vars, x, y)
        h0 = torch.randn(self.lstm_num_layers, 8192).to(self.device)  # hidden cell for rnn
        c0 = torch.randn(self.lstm_num_layers, 8192).to(self.device)  # hidden cell for rnn
        t4s = []

        for i in range(self.n):
            t0 = x[:, i].float()
            t1 = self.lins0[i](t0.flatten(start_dim=1))
            t2 = torch.sigmoid(t1)
            t3 = self.lins1[i](t2)
            t4 = torch.sigmoid(t3)
            t4s.append(t4)
        c2d1 = self.conv2d1(x)
        c2d2 = self.conv2d2(x)
        t = torch.cat(t4s, dim=1)
        t = torch.cat([t, c2d1, c2d2])
        print(t.shape)
        t = self.lin2(t)
        out, _ = self.lstm(t, (h0, c0))

        out = self.lin3(out)
        out = torch.sigmoid(out)
        out = out.reshape(1, self.x, self.y)

        return out

    def fit(self, x, y, optimizer, lr, batch_size, num_epochs, loss_fn):

        opt = optimizer(self.parameters(), lr=lr)
        losses = []
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        x = torch.tensor(x)
        y = torch.tensor(y)
        tds = TensorDataset(x, y)
        tdl = DataLoader(tds, batch_size)

        for epoch in range(num_epochs):
            for i, (xb, yb) in enumerate(tdl):
                xb, yb = xb.to(device), yb.to(device)
                predictions = self(xb)
                opt.zero_grad()
                loss = loss_fn(predictions, yb)
                loss.backward()
                opt.step()
            losses.append(loss.detach().item())
            print(f'Epoch: {epoch} | Loss: {loss.detach().item()}')

    def predict(self, x):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        x = torch.tensor(x)
        tds = TensorDataset(x)
        tdl = DataLoader(tds, batch_size=1)
        with torch.no_grad():
            self.eval()
            predictions = []
            for xt in tdl:
                xt = xt.to(device)
                predictions.append(self())
                break
        pass

