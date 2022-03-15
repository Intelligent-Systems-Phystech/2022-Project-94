import torch
from torch import nn
import torch.nn.functional as F

class GNNLayer(nn.Module):
    def __init__(self, n, long, lat):
        super().__init__()
        self.lin1 = nn.Linear(long * lat, 16*16)
        indecies = [[], []]
        values = []
        for j in range(long * lat):
            indecies[0].append(j)
            indecies[1].append(j)
            values.append(1)
            #self.A1[j, j] = 1
        for i in range(lat + 1, (long - 1) * lat - 1):
            indecies[0].append(i)
            indecies[1].append(i - 1)
            #self.A1[i, i - 1] = 1  # пиксель слева
            indecies[0].append(i)
            indecies[1].append(i + 1)
            #self.A1[i, i + 1] = 1  # пиксель справа
            indecies[0].append(i)
            indecies[1].append(i + lat)
            #self.A1[i, i + lat] = 1  # пиксель сверху
            indecies[0].append(i)
            indecies[1].append(i - lat)
            #self.A1[i, i - lat] = 1  # пиксель снизу
            indecies[0].append(i)
            indecies[1].append(i + lat - 1)
            #self.A1[i, i + lat - 1] = 1  # пиксель сверху слева
            indecies[0].append(i)
            indecies[1].append(i + lat + 1)
            #self.A1[i, i + lat + 1] = 1  # пиксель сверху справа
            indecies[0].append(i)
            indecies[1].append(i - lat - 1)
            #self.A1[i, i - lat - 1] = 1  # пиксель снизу слева
            indecies[0].append(i)
            indecies[1].append(i - lat + 1)
            #self.A1[i, i - lat + 1] = 1  # пиксель снизу справа
            # for symmetry
            indecies[0].append(i - 1)
            indecies[1].append(i)
            #self.A1[i - 1, i] = 1
            indecies[0].append(i + 1)
            indecies[1].append( i)
            #self.A1[i + 1, i] = 1
            indecies[0].append(i + lat)
            indecies[1].append(i)
            #self.A1[i + lat, i] = 1
            indecies[0].append(i - lat)
            indecies[1].append( i)
            #self.A1[i - lat, i] = 1
            indecies[0].append(i + lat - 1)
            indecies[1].append(i)
            #self.A1[i + lat - 1, i] = 1
            indecies[0].append(i + lat + 1)
            indecies[1].append( i)
            #self.A1[i + lat + 1, i] = 1
            indecies[0].append(i - lat - 1)
            indecies[1].append( i)
            #self.A1[i - lat - 1, i] = 1
            indecies[0].append(i - lat + 1)
            indecies[1].append(i)
            #self.A1[i - lat + 1, i] = 1
            for i in range(16):
                values.append(1)
        indecies = torch.Tensor(indecies)
        values = torch.Tensor(values)
        self.A1 = torch.sparse_coo_tensor(indecies, values, (long * lat, long * lat))
        self.A1 = self.A1.float()

    def forward(self, x):
        x_flatten = x.flatten(start_dim = 1)
        h1 = self.A1 @ x_flatten.T.float()
        h1 = h1.T
        h2 = self.lin1(h1)
        #h1 = h1.reshape(-1, 16, 16)
        return h2


class HailNet(nn.Module):
    def __init__(self, n, long, lat):
        super().__init__()
        self.embedding_layer = GNNLayer(n, long, lat)
        self.lin1 = nn.Linear(16*16, 1)

    def forward(self, x):
        h1 = self.embedding_layer(x)
        h2 = torch.sigmoid(h1)
        h3 = self.lin1(h2)
        h4 = torch.sigmoid(h3)
        return h4


def train(num_epochs, model, loss_fn, opt, train_dl):
    losses = []
    for epoch in range(num_epochs):
        model.train()
        for i, (xb, yb) in enumerate(train_dl):
            pred = model(xb)
            opt.zero_grad()
            loss = loss_fn(pred, yb)

            loss.backward()
            opt.step()
            
        losses.append(loss.detach().item())
        print(f'Epoch: {epoch} | Loss: {loss.detach().item()}')
    return losses