from equations import HeatEquation
from models import MLP

import torch 
import torch.nn as nn
from tqdm import tqdm
import os
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable


def scatter(x, y, c, title, fig, ax):
    h = ax.scatter(x, y, c=c, cmap="jet")
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(h, cax=cax, orientation='vertical')
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$")
    ax.set_title(title)

def uniform(a, b, size):
    return (b-a)*torch.rand(size=size) + a

class Normalizer:
    def __init__(self, x):
        self.min = x.min(0, keepdim=True).values
        self.max = x.max(0, keepdim=True).values
    def __call__(self, x):
        return (x-self.min)/(self.max-self.min)
    def undo(self, x):
        return x*(self.max-self.min)+self.min

def mlp_fit_heat_equation(d=1, T=1.0, n_train=1024, n_valid=1024):

    # prepare dataset
    equation = HeatEquation(d)

    # input  [n_points, 3] (x, y, mu)
    # output [n_points, 1] u(T, x, y, mu)
    train_points = uniform(-1, 1, size=(n_train, 2))
    valid_points = uniform(-1, 1, size=(n_valid, 2))
    train_input = torch.cat([train_points[:,None,:].tile(1, equation.d, 1), equation.mu.tile(n_train, 1, 1)], dim=-1).reshape([-1,3])
    valid_input = torch.cat([valid_points[:,None,:].tile(1, equation.d, 1), equation.mu.tile(n_valid, 1, 1)], dim=-1).reshape([-1,3])
    train_u = equation(T, train_points[:,0], train_points[:,1]).reshape(-1, 1)
    valid_u = equation(T, valid_points[:,0], valid_points[:,1]).reshape(-1, 1)

    # plot initial dataset
    fig, ax = plt.subplots(1, 2, figsize=(10,5))
    scatter(train_points[:,0], train_points[:,1], train_u.reshape(n_train,equation.d).sum(-1), "Train Data", fig, ax[0])
    scatter(valid_points[:,0], valid_points[:,1], valid_u.reshape(n_valid,equation.d).sum(-1), "Valid Data", fig, ax[1])
    fig.savefig("images/heat_equation_data.png", dpi=400)

    # do the normalization
    normalizer = Normalizer(train_u)
    train_u = normalizer(train_u)
    valid_u = normalizer(valid_u)

    # build up model and training utils
    model = MLP(input_size=3, output_size=1, hidden_size=32, num_layers=3)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.MSELoss()
    valid_every_eps = 100
    epoch = 10000
    losses = {
        "train":[],
        "valid":[]
    }
    best_weight, best_loss, best_epoch = None, float('inf'), None
    p = tqdm(range(epoch))
    for ep in p:
        model.train()
        optimizer.zero_grad()
        pred_u = model(train_input)
        loss = criterion(pred_u, train_u)
        loss.backward()
        optimizer.step()
        losses['train'].append((ep,loss.item()))
        p.set_postfix({'loss': loss.item()})
        if (ep+1) % valid_every_eps == 0:
            model.eval()
            with torch.no_grad():
                pred_u = model(valid_input)
                valid_loss = criterion(pred_u, valid_u)
                losses['valid'].append((ep,valid_loss.item()))
                # save best valid loss weight
                if valid_loss.item() < best_loss:
                    best_weight = model.state_dict()
                    best_loss = valid_loss.item()
                    best_epoch = ep
    if best_weight is not None:
        model.load_state_dict(best_weight)

    model.eval()

    # plot loss 
    fig, ax = plt.subplots(1, 1, figsize=(10,5))
    ax.plot([x[0] for x in losses['train']],[x[1] for x in losses['train']], label="Train Loss",  linestyle="--", alpha=0.6)
    ax.scatter([x[0] for x in losses['valid']],[x[1] for x in losses['valid']], label="Valid Loss",  alpha=0.6)
    ax.scatter(best_epoch, best_loss, label="Best Valid Loss", marker="*", s=200, c="red")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Loss")
    ax.set_yscale("log")
    ax.legend()
    fig.savefig("images/heat_equation_loss.png", dpi=400)

    # plot train and validation
    fig, ax = plt.subplots(1, 2, figsize=(10,5))
    with torch.no_grad():
        pred_train_u = model(train_input)
        pred_valid_u = model(valid_input)
    pred_train_u = normalizer.undo(pred_train_u)
    pred_valid_u = normalizer.undo(pred_valid_u)
    scatter(train_points[:,0], train_points[:,1], pred_train_u.reshape(n_train,equation.d).sum(-1), "Train Prediction", fig, ax[0])
    scatter(valid_points[:,0], valid_points[:,1], pred_valid_u.reshape(n_valid,equation.d).sum(-1), "Valid Prediction", fig, ax[1])
    fig.savefig("images/heat_equation_train_valid.png", dpi=400)

    # plot prediction
    x = torch.linspace(-1, 1, 100)
    y = torch.linspace(-1, 1, 100)
    X, Y = torch.meshgrid(x, y)
    points = torch.stack([X.reshape(-1), Y.reshape(-1)], dim=-1)
    input = torch.cat([points[:,None,:].tile(1, equation.d, 1), equation.mu.tile(100*100, 1, 1)], dim=-1).reshape([-1,3])
    with torch.no_grad():
        pred_u = model(input)
    pred_u = normalizer.undo(pred_u)
    exact_u = equation(T, points[:,0], points[:,1]).reshape(-1, 1)
    fig, ax = plt.subplots(1, 3, figsize=(15,5))
    scatter(points[:,0], points[:,1], pred_u.reshape(100*100,equation.d).sum(-1), "Prediction", fig, ax[0])
    scatter(points[:,0], points[:,1], exact_u.reshape(100*100,equation.d).sum(-1), "Exact", fig, ax[1])
    scatter(points[:,0], points[:,1], (pred_u-exact_u).reshape(100*100,equation.d).sum(-1), "Error", fig, ax[2])
    fig.savefig("images/heat_equation_prediction.png", dpi=400)

if __name__ == '__main__':
    if not os.path.exists("images"):
        os.makedirs("images")
    mlp_fit_heat_equation()

