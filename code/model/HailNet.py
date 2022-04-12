import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F


class GNNLayer(nn.Module):
    r"""
        Graph layer for HailNet
    """
    def __init__(self, n, long, lat, output_size=16):
        super().__init__()
        self.lin1 = nn.Linear(n * long * lat, output_size*output_size)
        indices = [[], []]
        values = []
        for k in range(n):
            for j in range(long * lat, step=3):
                indices[0].append(k + j)
                indices[1].append(k + j)
                indices[0].append(k + j)
                indices[1].append(j)
                indices[1].append(k + j)
                indices[0].append(j)
                values.append(1)
                values.append(1)
                values.append(1)
            for i in range(lat + 1, (long - 1) * lat - 1, step=3):
                x = i
                for y in k + np.array([i - 1, i + 1, i + lat, i - lat, i + lat - 1,
                                       i + lat + 1, i - lat - 1, i - lat + 1]):
                    indices[0].append(x)
                    indices[1].append(y)
                for y in k + np.array([i - 1, i + 1, i + lat, i - lat, i + lat - 1,
                                       i + lat + 1, i - lat - 1, i - lat + 1]):
                    indices[0].append(y)
                    indices[1].append(x)
                for _ in range(16):
                    values.append(1)

        indices = torch.Tensor(indices)
        values = torch.Tensor(values)
        self.A1 = torch.sparse_coo_tensor(indices, values, (n * long * lat, n * long * lat))
        self.A1 = self.A1.float()

    def forward(self, x):
        x_flatten = x.flatten(start_dim=1)
        h1 = self.A1 @ x_flatten.T.float()
        h1 = h1.T
        h2 = self.lin1(h1)
        return h2


class HailNet(nn.Module):
    def __init__(self, n, long, lat, gru_hidden_size, gru_num_layers, lin1_size=16, seq_len=24, units=None):
        super().__init__()
        self.n = n
        self.long = long
        self.lat = lat
        self.gru_hidden_size = gru_hidden_size
        self.lin1_size = lin1_size
        self.gru_num_layers = gru_num_layers
        self.seq_len = seq_len
        self.units = units
        self.embedding_layer = GNNLayer(n, long, lat)
        self.lin1 = nn.Linear(self.lin1_size * self.lin1_size, self.lin1_size * self.lin1_size)
        # отредактировать и отладить сверточные слои
        self.conv2d1 = nn.Conv2d(in_channels=n, out_channels=5, kernel_size=3, stride=1, padding=0)
        self.conv2d2 = nn.Conv2d(in_channels=n, out_channels=5, kernel_size=5, stride=1, padding=0)
        self.conv2d3 = nn.Conv2d(in_channels=n, out_channels=5, kernel_size=11, stride=1, padding=0)
        self.conv3d1 = nn.Conv3d()
        # отредактировать и отладить сверточные слои
        self.gru = nn.GRU(
            input_size=self.lin1_size * self.lin1_size,
            hidden_size=self.gru_hidden_size,
            num_layers=self.gru_num_layers,
            batch_first=True
        )
        if units is None:
            self.fully_connected_layer = self.fully_connected()
        else:
            self.fully_connected_layer = self.fully_connected(self.units)

    @staticmethod
    def block1(unit):
        return nn.Sequential(nn.Linear(unit[0], unit[1]), nn.Sigmoid())

    def fully_connected(self, units: list = [16*16, 16, 16]):
        net = nn.Sequential()
        for i, unit in enumerate(units):
            if i == len(units) - 1:
                net.add_module(f'block {i}', self.block1([unit, 1]))
            else:
                net.add_module(f'block {i}', self.block1([unit, units[i + 1]]))
        return net

    def forward(self, x):
        # x -> (n_batch, seq_len, n_vars, long, lat)
        h0 = torch.randn(self.gru_num_layers, x.size(0), self.gru_hidden_size)  # hidden cell for gru
        hs1 = []
        for i in range(self.seq_len):
            t1 = self.embedding_layer(x[:, i])
            t2 = torch.sigmoid(t1)
            t3 = self.lin1(t2)
            t4 = torch.sigmoid(t3)
            t4 = t4.unsqueeze(dim=1)
            # отредактировать и отладить сверточные слои
            c3d = self.conv3d(x[:, i])
            c2d1 = self.conv2d1(x[:, i])
            c2d2 = self.conv2d2(x[:, i])
            c2d3 = self.conv2d3(x[:, i])
            h = torch.cat([t4,
                           c3d,
                           c2d1.reshape(x.shape[0], self.seq_len, -1),
                           c2d2.reshape(x.shape[0], self.seq_len, -1),
                           c2d3.reshape(x.shape[0], self.seq_len, -1)
                           ])
            # отредактировать и отладить сверточные слои
            hs1.append(h)

        t = torch.cat(hs1, dim=1)
        # t -> (bacth_size, seq_len, lin1*lin1 + c3d.shape)
        out, _ = self.gru(t, h0)
        out = out[:, -1, :]
        out = self.fully_connected_layer(out)

        return out


def train(num_epochs: int, model, loss_fn, opt, train_dl: torch.utils.data.DataLoader):
    losses = []
    for epoch in range(num_epochs):
        model.train()
        for i, (xb, yb) in enumerate(train_dl):
            pred = model(xb)
            opt.zero_grad()
            loss = loss_fn(pred, yb)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5)
            opt.step()

        losses.append(loss.detach().item())
        print(f'Epoch: {epoch} | Loss: {loss.detach().item()}')
    return losses


def test(model, test_dl: torch.utils.data.DataLoader, metrics: list, metrics_funcs: dict):
    with torch.no_grad():
        model.eval()
        predictions = []
        true_values = []
        metrics_values = {}.fromkeys(metrics)
        for xt, yt in test_dl:
            predictions.append(model(xt))
            true_values.append(yt)
    return predictions, true_values
