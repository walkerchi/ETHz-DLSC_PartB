import argparse
import math
import numpy as np
import torch 
import torch.nn as nn
from tqdm import tqdm
import os
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

from equations import HeatEquation
from models import MLP, DeepONet, FNO2d, CNO2d

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

def keepdim_min(x, dims):
    if isinstance(dims, (tuple, list)) and len(dims) > 1:
        return keepdim_min(x.min(dims[0], keepdim=True).values, dims[1:])
    elif isinstance(dims, (tuple, list)) and len(dims) == 1:
        return x.min(dims[0], keepdim=True).values
    else:
        return x.min(dims, keepdim=True).values

def keepdim_max(x, dims):
    if isinstance(dims, (tuple, list)) and len(dims) > 1:
        return keepdim_max(x.max(dims[0], keepdim=True).values, dims[1:])
    elif isinstance(dims, (tuple, list)) and len(dims) == 1:
        return x.max(dims[0], keepdim=True).values
    else:
        return x.max(dims, keepdim=True).values

class Normalizer:
    def __init__(self, x, axis=0):
        self.axis = axis
        self.min = keepdim_min(x, axis)
        self.max = keepdim_max(x, axis)
    def __call__(self, x):
        return (x-self.min)/(self.max-self.min)
    def undo(self, x):
        return x*(self.max-self.min)+self.min


def plot_dataset(train_points, train_u, valid_points, valid_u, d, image_path="images/heat_equation_data.png"):
    fig, ax = plt.subplots(1, 2, figsize=(10,5))
    scatter(train_points[:,0], train_points[:,1], train_u.reshape(-1,d).sum(-1), "Train Data", fig, ax[0])
    scatter(valid_points[:,0], valid_points[:,1], valid_u.reshape(-1,d).sum(-1), "Valid Data", fig, ax[1])
    fig.savefig(image_path, dpi=400)

def fit(model, 
        train_input, train_output, 
        valid_input, valid_output,
        epoch=10000,
        lr = 1e-3,
        weight_decay=1e-4,
        eval_every_eps=100,
        loss_image_path="images/heat_equation_loss.png",
        device="cpu"):
    
    model = model.to(device)
    train_input  = train_input.to(device)
    train_output = train_output.to(device)
    valid_input  = valid_input.to(device)
    valid_output = valid_output.to(device)
    losses = {
        "train":[],
        "valid":[]
    }
    best_weight, best_loss, best_epoch = None, float('inf'), None
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.MSELoss()
    p = tqdm(range(epoch))
    
    for ep in p:
        # training step
        model.train()
        optimizer.zero_grad()

        if isinstance(train_input, (list,  tuple)):
            # if train_input is (input1, input2) or [input1, input2] format, do it as model(input1, input2)
            pred_u = model(*train_input)
        else:
            pred_u = model(train_input)

        loss = criterion(pred_u, train_output)
        loss.backward()
        optimizer.step()

        # record for display
        losses['train'].append((ep,loss.item()))
        p.set_postfix({'loss': loss.item()})

        if (ep+1) % eval_every_eps == 0:
            # validation every eval_every_eps epoch
            model.eval()

            with torch.no_grad():
                if isinstance(valid_input, (list,  tuple)):
                    # if valid_input is (input1, input2) or [input1, input2] format, do it as model(input1, input2)
                    prediction = model(*valid_input)
                else:
                    prediction = model(valid_input)
                valid_loss = criterion(prediction, valid_output)
                losses['valid'].append((ep,valid_loss.item()))
                # save best valid loss weight
                if valid_loss.item() < best_loss:
                    best_weight = model.state_dict()
                    best_loss = valid_loss.item()
                    best_epoch = ep

    # load the best recorded weight
    if best_weight is not None:
        model.load_state_dict(best_weight)

    # plot loss
    fig, ax = plt.subplots(1, 1, figsize=(10,5))
    ax.plot([x[0] for x in losses['train']],[x[1] for x in losses['train']], label="Train Loss",  linestyle="--", alpha=0.6)
    ax.scatter([x[0] for x in losses['valid']],[x[1] for x in losses['valid']], label="Valid Loss",  alpha=0.6, color="orange")
    ax.scatter(best_epoch, best_loss, label="Best Valid Loss", marker="*", s=200, c="red")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Loss")
    ax.set_yscale("log")
    ax.legend()
    fig.savefig(loss_image_path, dpi=400)

    model.eval()
    model = model.cpu()
    train_input  = train_input.cpu()
    train_output = train_output.cpu()
    valid_input  = valid_input.cpu()
    valid_output = valid_output.cpu()


def ffn_fit_heat_equation(d=1, T=1.0, n_train=1024, n_valid=1024):
    """
        g(x,y,\mu ) = u(T, x, y, \mu )
    """
    os.makedirs("images/ffn", exist_ok=True)
    os.makedirs("weights/ffn", exist_ok=True)

    # prepare dataset
    equation = HeatEquation(d)
    n_train = n_train//d
    n_valid = n_valid//d

    # input  [n_points, 3] (x, y, mu)
    # output [n_points, 1] u(T, x, y, mu)
    train_points = uniform(-1, 1, size=(n_train, 2))
    valid_points = uniform(-1, 1, size=(n_valid, 2))
    train_input = torch.cat([train_points[:,None,:].tile(1, equation.d, 1), equation.mu.tile(n_train, 1, 1)], dim=-1).reshape([-1,3])
    valid_input = torch.cat([valid_points[:,None,:].tile(1, equation.d, 1), equation.mu.tile(n_valid, 1, 1)], dim=-1).reshape([-1,3])
    train_u = equation(T, train_points[:,0], train_points[:,1]).reshape(-1, 1)
    valid_u = equation(T, valid_points[:,0], valid_points[:,1]).reshape(-1, 1)

    # plot initial dataset
    plot_dataset(train_points, train_u, valid_points, valid_u, d, image_path="images/ffn/heat_equation_data.png")

    # do the normalization
    normalizer = Normalizer(train_u)
    train_u, valid_u = normalizer(train_u), normalizer(valid_u)

    # build up model and training utils
    model = MLP(input_size=3, output_size=1, hidden_size=32, num_layers=3)
    fit(model,
        train_input, train_u,
        valid_input, valid_u,
        epoch=10000,
        lr = 1e-3,
        weight_decay=1e-4,
        eval_every_eps=100,
        loss_image_path="images/ffn/heat_equation_loss.png")
    
    torch.save(model.state_dict(), "weights/ffn/weight.pth")

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
    fig.savefig("images/ffn/heat_equation_prediction.png", dpi=400)





def deeponet_fit_heat_equation(d=1, T=1.0, n_train=1024, n_valid=1024):
    """
        G(u_0)(x,y) = u(T,x,y)
    """
    os.makedirs("images/deeponet", exist_ok=True)
    os.makedirs("weights/deeponet", exist_ok=True)

    equation = HeatEquation(d)

    # input.basis /branch  [n_basis, d] (u0_m)
    # input.weight/trunck  [n_points, 3] (T,x1,x2) 
    # output               [n_points, d] u(T, x1, x2)
    train_points = uniform(-1, 1, size=(n_train, 2))
    valid_points = uniform(-1, 1, size=(n_valid, 2))
    basis_points = uniform(-1, 1, size=(n_train, 2))
    basis_u = equation(0, basis_points[:,0], basis_points[:,1]) #[n_train, d]
    basis  = torch.cat([basis_points, basis_u], dim=-1) #[n_train, 2+d]
    train_points  = torch.cat([T*torch.ones([n_train,1]), train_points], dim=-1) #[n_train, 3]
    valid_points  = torch.cat([T*torch.ones([n_valid,1]), valid_points], dim=-1) #[n_valid, 3]
    train_u       = equation(T, train_points[:,1], train_points[:,2]) #[n_train, d]
    valid_u       = equation(T, valid_points[:,1], valid_points[:,2]) #[n_valid, d]

    # plot initial dataset
    plot_dataset(train_points[:,1:], train_u, valid_points[:,1:], valid_u, d, image_path="images/deeponet/heat_equation_data.png")

    # do the normalization
    normalizer = Normalizer(train_u)
    train_u = normalizer(train_u)
    valid_u = normalizer(valid_u)

    # build up model and training utils
    model = DeepONet(basis_size=2+equation.d, point_size=3, output_size=equation.d, hidden_size=32, num_layers=3)
    fit(model,
        [basis, train_points], train_u,
        [basis, valid_points], valid_u,
        epoch=10000,
        lr = 1e-3,
        weight_decay=1e-4,
        eval_every_eps=100,
        loss_image_path="images/deeponet/heat_equation_loss.png")

    torch.save(model.state_dict(), "weights/deeponet/weight.pth")

    # plot prediction
    X, Y = torch.meshgrid(torch.linspace(-1, 1, 100), torch.linspace(-1, 1, 100))
    points = torch.stack([T*torch.ones(100*100), X.reshape(-1), Y.reshape(-1)], dim=-1)
    with torch.no_grad():
        pred_u = model(basis, points)
    pred_u = normalizer.undo(pred_u)
    exact_u = equation(T, points[:,1], points[:,2]).reshape(-1, 1)
    fig, ax = plt.subplots(1, 3, figsize=(15,5))
    scatter(points[:,1], points[:,2], pred_u.reshape(100*100,equation.d).sum(-1), "Prediction", fig, ax[0])
    scatter(points[:,1], points[:,2], exact_u.reshape(100*100,equation.d).sum(-1), "Exact", fig, ax[1])
    scatter(points[:,1], points[:,2], (pred_u-exact_u).reshape(100*100,equation.d).sum(-1), "Error", fig, ax[2])
    fig.savefig("images/deeponet/heat_equation_prediction.png", dpi=400)
    




def fno_fit_heat_equation(d=1, T=1.0, n_train=256, n_valid=1024):
    """
        G(u_0)(x,y) = u(T,x,y)
    """
    os.makedirs("images/fno", exist_ok=True)
    os.makedirs("weights/fno", exist_ok=True)

    equation = HeatEquation(d)

    # input [1, 2+d, window_size, window_size] u0
    # output [1, d, window_size, window_size] u(T, x1, x2)

    n_train = int(math.sqrt(n_train))
    n_valid = int(math.sqrt(n_valid))
    train_x1, train_x2 = torch.meshgrid(torch.linspace(-1, 1, n_train), torch.linspace(-1, 1, n_train))
    valid_x1, valid_x2 = torch.meshgrid(torch.linspace(-1, 1, n_valid), torch.linspace(-1, 1, n_valid))
    train_points = torch.stack([train_x1, train_x2], 0)
    valid_points = torch.stack([valid_x1, valid_x2], 0)
    train_u0 = equation(0, train_x1.flatten(), train_x2.flatten()).reshape(n_train, n_train, equation.d).permute(2,0,1)
    train_uT = equation(T, train_x1.flatten(), train_x2.flatten()).reshape(n_train, n_train, equation.d).permute(2,0,1)
    valid_u0 = equation(0, valid_x1.flatten(), valid_x2.flatten()).reshape(n_valid, n_valid, equation.d).permute(2,0,1)
    valid_uT = equation(T, valid_x1.flatten(), valid_x2.flatten()).reshape(n_valid, n_valid, equation.d).permute(2,0,1)
    train_input = torch.cat([train_points,  train_u0], dim=0)[None,:]
    valid_input = torch.cat([valid_points, valid_u0], dim=0)[None,:]
    train_output = train_uT[None,:]
    valid_output = valid_uT[None,:]
    
    # plot initial dataset
    plot_dataset(train_points.T.reshape(-1,2), 
                 train_uT.reshape(d, -1).T, 
                 valid_points.T.reshape(-1,2), 
                 valid_uT.reshape(d, -1).T, 
                 d, 
                 image_path="images/fno/heat_equation_data.png")

    # do the normalization
    normalizer = Normalizer(valid_output, axis=(-2,-1))
    train_output, valid_output = normalizer(train_output), normalizer(valid_output)

    # build up model and training utils
    model = FNO2d(in_channel=2+d, out_channel=d, hidden_channel=32, num_layers=3)
    fit(model,
        train_input, train_output,
        valid_input, valid_output,
        epoch=10000,
        lr = 1e-3,
        weight_decay=1e-4,
        eval_every_eps=100,
        loss_image_path="images/fno/heat_equation_loss.png")
    
    torch.save(model.state_dict(), "weights/fno/weight.pth")

    # plot prediction
    X, Y = torch.meshgrid(torch.linspace(-1, 1, 100), torch.linspace(-1, 1, 100))
    u0 = equation(0, X.reshape(-1), Y.reshape(-1)).reshape(100,100,d).permute(2,0,1)
    input = torch.cat([torch.stack([X,Y],0),  u0], dim=0)[None,...]
    with torch.no_grad():
        pred_uT = model(input)
    pred_uT = normalizer.undo(pred_uT)[0].permute(1,2,0)
    exact_uT = equation(T, X.reshape(-1), Y.reshape(-1)).reshape(100,100,d)
    fig, ax = plt.subplots(1, 3, figsize=(15,5))
    scatter(X.flatten(), Y.flatten(), pred_uT.reshape(-1,d).sum(-1), "Prediction", fig, ax[0])
    scatter(X.flatten(), Y.flatten(), exact_uT.reshape(-1,d).sum(-1), "Exact", fig, ax[1])
    scatter(X.flatten(), Y.flatten(), (pred_uT-exact_uT).reshape(-1,d).sum(-1), "Error", fig, ax[2])
    fig.savefig("images/fno/heat_equation_prediction.png", dpi=400)





def cno_fit_heat_equation(d=1, T=1.0, n_train=256, n_valid=1024, device='cuda:0'):
    """
        G(u_0)(x,y) = u(T,x,y)
    """
    os.makedirs("images/cno", exist_ok=True)
    os.makedirs("weights/cno", exist_ok=True)

    equation = HeatEquation(d)

    # input [1, 2+d, window_size, window_size] u0
    # output [1, d, window_size, window_size] u(T, x1, x2)

    n_train = int(math.sqrt(n_train))
    n_valid = int(math.sqrt(n_valid))
    train_x1, train_x2 = torch.meshgrid(torch.linspace(-1, 1, n_train), torch.linspace(-1, 1, n_train))
    valid_x1, valid_x2 = torch.meshgrid(torch.linspace(-1, 1, n_valid), torch.linspace(-1, 1, n_valid))
    train_points = torch.stack([train_x1, train_x2], 0)
    valid_points = torch.stack([valid_x1, valid_x2], 0)
    train_u0 = equation(0, train_x1.flatten(), train_x2.flatten()).reshape(n_train, n_train, equation.d).permute(2,0,1)
    train_uT = equation(T, train_x1.flatten(), train_x2.flatten()).reshape(n_train, n_train, equation.d).permute(2,0,1)
    valid_u0 = equation(0, valid_x1.flatten(), valid_x2.flatten()).reshape(n_valid, n_valid, equation.d).permute(2,0,1)
    valid_uT = equation(T, valid_x1.flatten(), valid_x2.flatten()).reshape(n_valid, n_valid, equation.d).permute(2,0,1)
    train_input = torch.cat([train_points,  train_u0], dim=0)[None,:]
    valid_input = torch.cat([valid_points, valid_u0], dim=0)[None,:]
    train_output = train_uT[None,:]
    valid_output = valid_uT[None,:]
    
    # plot initial dataset
    plot_dataset(train_points.T.reshape(-1,2), 
                 train_uT.reshape(d, -1).T, 
                 valid_points.T.reshape(-1,2), 
                 valid_uT.reshape(d, -1).T, 
                 d, 
                 image_path="images/cno/heat_equation_data.png")

    # do the normalization
    normalizer = Normalizer(valid_output, axis=(-2,-1))
    train_output, valid_output = normalizer(train_output), normalizer(valid_output)

    # build up model and training utils
    model = CNO2d(in_channel=2+d, out_channel=d)
    fit(model,
        train_input, train_output,
        valid_input, valid_output,
        epoch=10000,
        lr = 1e-3,
        weight_decay=1e-4,
        eval_every_eps=100,
        loss_image_path="images/cno/heat_equation_loss.png",
        device=device)
    
    torch.save(model.state_dict(), "weights/cno/weight.pth")

    # plot prediction
    X, Y = torch.meshgrid(torch.linspace(-1, 1, 100), torch.linspace(-1, 1, 100))
    u0 = equation(0, X.reshape(-1), Y.reshape(-1)).reshape(100,100,d).permute(2,0,1)
    input = torch.cat([torch.stack([X,Y],0),  u0], dim=0)[None,...]
    with torch.no_grad():
        pred_uT = model(input)
    pred_uT = normalizer.undo(pred_uT)[0].permute(1,2,0)
    exact_uT = equation(T, X.reshape(-1), Y.reshape(-1)).reshape(100,100,d)
    fig, ax = plt.subplots(1, 3, figsize=(15,5))
    scatter(X.flatten(), Y.flatten(), pred_uT.reshape(-1,d).sum(-1), "Prediction", fig, ax[0])
    scatter(X.flatten(), Y.flatten(), exact_uT.reshape(-1,d).sum(-1), "Exact", fig, ax[1])
    scatter(X.flatten(), Y.flatten(), (pred_uT-exact_uT).reshape(-1,d).sum(-1), "Error", fig, ax[2])
    fig.savefig("images/cno/heat_equation_prediction.png", dpi=400)




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", type=str, default="ffn", choices=["ffn", "deeponet", "fno","cno"])
    parser.add_argument("--d", type=int, default=1)
    parser.add_argument("--T", type=float, default=1.0)
    parser.add_argument("--n_train", type=int, default=256)
    parser.add_argument("--n_valid", type=int, default=1024)
    args = parser.parse_args()

    {
        "ffn":ffn_fit_heat_equation,
        "deeponet":deeponet_fit_heat_equation,
        "fno":fno_fit_heat_equation,
        "cno":cno_fit_heat_equation
    }[args.model](args.d, args.T, args.n_train, args.n_valid)


