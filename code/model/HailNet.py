import torch
from torch import nn
import numpy as np
import glob
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torchvision import transforms

class GNNLayer(nn.Module):
    r"""
        Graph layer for HailNet
        This layer is not used right now in the HailNet
    """
    def __init__(self, n, long, lat, output_size=16):
        super().__init__()
        self.lin1 = nn.Linear(n * long * lat, output_size*output_size)
        indices = [[], []]
        values = []
        for k in range(n):
            for j in range(0, long * lat, 3):
                indices[0].append(k + j)
                indices[1].append(k + j)
                indices[0].append(k + j)
                indices[1].append(j)
                indices[1].append(k + j)
                indices[0].append(j)
                values.append(1)
                values.append(1)
                values.append(1)
            for i in range(lat + 1, (long - 1) * lat - 1, 3):
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
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.A1 = self.A1.to(self.device)

    def forward(self, x):
        x_flatten = x.flatten(start_dim=1)
        #h1 = self.A1 @ x_flatten.T.float()
        h1 = x_flatten.T
        h1 = h1.T
        h2 = self.lin1(h1)
        return h2


class HailNet(nn.Module):
    r"""
        HailNet - main model
    """
    def __init__(self, n, long, lat, rnn_hidden_size, rnn_num_layers, lin1_size=16, seq_len=24, units=None):
        super().__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.n = n
        self.long = long
        self.lat = lat
        self.rnn_hidden_size = rnn_hidden_size
        self.lin1_size = lin1_size
        self.rnn_num_layers = rnn_num_layers
        self.seq_len = seq_len
        self.lin0 = nn.Linear(n * long * lat, 256)
        self.lin1 = nn.Linear(256, 1024)
        self.conv2d1 = nn.Conv2d(in_channels=n, out_channels=n, kernel_size=50, stride=1, padding=0)
        self.lin2 = nn.Linear(1024 + (self.long - 49) * (self.lat - 49) * self.n, 1024)
        self.lin3 = nn.Linear(1024, 1)
        self.lstm = nn.LSTM(
            input_size=int(1024),
            hidden_size=1024,
            num_layers=self.rnn_num_layers,
            batch_first=True
        )

    def forward(self, x):
        # x -> (n_batch, seq_len, n_vars, long, lat)
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


def train(num_epochs: int,
          model,
          loss_fn,
          opt,
          train_dl: torch.utils.data.DataLoader):
    r"""

    Args:
        num_epochs:
        model:
        loss_fn:
        opt:
        train_dl:

    Returns:
        losses: list of losses during training

    Training cycle for HailNet

    """
    losses = []
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    for epoch in range(num_epochs):
        model.train()
        for i, (xb, yb) in enumerate(train_dl):
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb)
            opt.zero_grad()
            loss = loss_fn(pred, yb)
            loss.backward()

            opt.step()

        losses.append(loss.detach().item())
        print(f'Epoch: {epoch} | Loss: {loss.detach().item()}')
    return losses


def test(model,
         test_dl: torch.utils.data.DataLoader,
         metrics: list,
         metrics_funcs: dict):
    r"""

    Args:
        model:
        test_dl:
        metrics:
        metrics_funcs:

    Returns:
        predictions: list of predictions for testing data
        true_values: list of true labels for testing data

    Testing cycle for HailNet

    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    with torch.no_grad():
        model.eval()
        predictions = []
        true_values = []
        metrics_values = {}.fromkeys(metrics)
        for xt, yt in test_dl:
            xt, yt = xt.to(device), yt.to(device)
            predictions.append(model(xt))
            true_values.append(yt)
    return predictions, true_values


def test_lazy_load(model,
                   test_data_path: str,
                   metrics: list,
                   metrics_funcs: dict,
                   feature_names: list,
                   get_tensors: callable):
    r"""

    Args:
        model:
        test_data_path:
        metrics:
        metrics_funcs:
        feature_names:
        get_tensors:

    Returns:
        predictions: list of predictions for testing data
        true_values: list of true labels for testing data

    Testing cycle for HailNet with loading memory from hard disk

    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    with torch.no_grad():
        model.eval()
        predictions = []
        true_values = []
        metrics_values = {}.fromkeys(metrics)
        hail_path = test_data_path + "/Hail/"
        no_hail_path = test_data_path + "/No hail/"
        hail_paths = glob.glob(hail_path + "*")
        no_hail_paths = glob.glob(no_hail_path + "*")
        for i, p in tqdm(enumerate(hail_paths + no_hail_paths)):
            x = get_tensors([feature_names[0]], p + "/*")
            x = x[feature_names[0]]
            x = np.nan_to_num(x)
            x = np.expand_dims(x, axis=1)
            for feature_name in feature_names[1:]:
                numpys = get_tensors([feature_name], p + "/*")
                x = np.concatenate((x, np.expand_dims(numpys[feature_name], axis=1)), axis=1)
            x = torch.from_numpy(x)
            x = x.long().unsqueeze(dim=0).to(device)
            predictions.append(model(x))
            if i <= len(hail_paths):
                true_values.append(1)
            else:
                true_values.append(0)
    return predictions, true_values
