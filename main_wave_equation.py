#%% import packages
import argparse
import math
from operator import eq
import numpy as np
import torch 
import torch.nn as nn
from tqdm import tqdm
import os
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

from equations import HeatEquation, WaveEquation
from main import deeponet_fit_heat_equation
from models import MLP, DeepONet, FNO2d, CNO2d

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.manual_seed(1466)

#%% define util functions

def scatter(x, y, c, title, fig, ax):
    h = ax.scatter(x, y, c=c, cmap="jet")
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(h, cax=cax, orientation='vertical')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
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


def plot_dataset(train_points, train_u, valid_points, valid_u, image_path="images/wave_equation_data.png"):
    fig, ax = plt.subplots(1, 2, figsize=(10,5))
    scatter(train_points[:,0], train_points[:,1], train_u, "Train Data", fig, ax[0])
    scatter(valid_points[:,0], valid_points[:,1], valid_u, "Valid Data", fig, ax[1])
    fig.savefig(image_path, dpi=400)

#%% predefine training process

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
    if isinstance(train_input, (list,  tuple)):
        train_input = [x.to(device) for x in train_input]
    else:
        train_input  = train_input.to(device)
    if isinstance(valid_input, (list,  tuple)):
        valid_input = [x.to(device) for x in valid_input]
    else:
        valid_input  = valid_input.to(device)

    train_output = train_output.to(device)
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

    if isinstance(train_input, (list,  tuple)):
        train_input = [x.cpu() for x in train_input]
    else:
        train_input  = train_input.cpu()
    if isinstance(valid_input, (list,  tuple)):
        valid_input = [x.cpu() for x in valid_input]
    else:
        valid_input  = valid_input.cpu()

    train_output = train_output.cpu()
    valid_output = valid_output.cpu()

#%%  define different kinds of training process
def ffn_fit_wave_equation(K = 1, T=1.0, r:float = 0.85, c:float = 0.1, n_train_pro_K = 64, n_valid_pro_K = 64, n_train=4096, n_valid=4096):
    """
        g(x,y,a) = u(T, x, y, a)
    """
    os.makedirs("images/ffn", exist_ok=True)
    os.makedirs("weights/ffn", exist_ok=True)

    # prepare dataset
    equation = WaveEquation(K=K, r=r, c=c)
    n_samples_train = n_train//n_train_pro_K
    n_samples_valid = n_valid//n_valid_pro_K

    # input  [n_points, 2 + K**2] (x, y, a)
    # output [n_points, 1] u(T, x, y, a)

    a_train_list = []
    a_valid_list = []
    train_input_list = []
    valid_input_list = []

    for _ in range(n_samples_train):
        a_train_list.append(equation.a.reshape(1, -1).tile((n_train_pro_K, 1)))
        train_input_list.append(uniform(0, 1, size=(n_train_pro_K, 2)))
        equation.reset_a()
    a_train_input = torch.cat(a_train_list, dim = 0)
    train_input = torch.cat(train_input_list, dim = 0)
        
    for _ in range(n_samples_valid):
        a_valid_list.append(equation.a.reshape(1, -1).tile((n_valid_pro_K, 1)))
        valid_input_list.append(uniform(0, 1, size=(n_train_pro_K, 2)))
        equation.reset_a()  # generate random a for each profile

    a_valid_input = torch.cat(a_valid_list, dim = 0)
    valid_input = torch.cat(valid_input_list, dim = 0)

    train_input = torch.cat((train_input,a_train_input), dim = 1)
    valid_input = torch.cat((valid_input,a_valid_input), dim = 1)
    
    K = equation.K
    # vectorization of a 
    train_u = equation(T, train_input[:,0], train_input[:,1], a = train_input[:,2:].reshape(-1, K, K)).reshape(-1, 1)
    valid_u = equation(T, valid_input[:,0], valid_input[:,1], a = valid_input[:,2:].reshape(-1 ,K, K)).reshape(-1, 1)

    # plot initial dataset
    plot_dataset(train_input, train_u, valid_input, valid_u, image_path="images/ffn/wave_equation_data.png")

    # do the normalization
    normalizer = Normalizer(train_u)
    train_u, valid_u = normalizer(train_u), normalizer(valid_u)

    # build up model and training utils
    model = MLP(input_size=2 + K**2, output_size=1, hidden_size=32, num_layers=3)
    fit(model,
        train_input, train_u,
        valid_input, valid_u,
        epoch=20000,
        lr = 1e-3,
        weight_decay=1e-4,
        eval_every_eps=100,
        loss_image_path="images/ffn/wave_equation_loss.png",
        device=device)
    
    torch.save(model.state_dict(), "weights/ffn/weight.pth")

    # plot prediction with only one K
    # Use 100 points for the illustration
    x = torch.linspace(0, 1, 100)
    y = torch.linspace(0, 1, 100)
    X, Y = torch.meshgrid(x, y)
    a = uniform(-1, 1, size=(K, K))
    a_input = a.reshape(1, -1).tile((X.reshape(-1).shape[0], 1))
    print(a_input.shape)
    print(X.reshape(-1,1).shape, Y.shape)
    input = torch.cat([X.reshape(-1,1), Y.reshape(-1,1), a_input], dim=1)
    with torch.no_grad():
        pred_u = model(input)
    pred_u = normalizer.undo(pred_u)
    exact_u = equation(T, input[:,0], input[:,1], a=a).reshape(-1, 1)
    fig, ax = plt.subplots(1, 3, figsize=(15,5))
    scatter(input[:,0], input[:,1], pred_u, "Prediction", fig, ax[0])
    scatter(input[:,0], input[:,1], exact_u, "Exact", fig, ax[1])
    scatter(input[:,0], input[:,1], (pred_u-exact_u), "Error", fig, ax[2])
    fig.savefig("images/ffn/wave_equation_prediction.png", dpi=400)


def deeponet_fit_wave_equation(T:float = 1.0, K:int = 1, r:float = 0.85, c:float = 0.1,n_samples = 3,n_points_pro_init = 64, n_train:int = 4096, n_valid:int = 4096):
    os.makedirs("images/deeponet", exist_ok=True)
    os.makedirs("weights/deeponet", exist_ok=True)

    equation = WaveEquation(K=K, r=r, c=c)
    
    samples = torch.linspace(0, 1, n_samples)
    x_samples, y_samples = torch.meshgrid(samples, samples, indexing="ij")
    x_samples = x_samples.reshape(-1)
    y_samples = y_samples.reshape(-1)
    sample_points = torch.stack([0 * torch.ones(n_samples**2), x_samples, y_samples], dim=-1)

    n_init_train = n_train//n_points_pro_init
    n_init_valid = n_valid//n_points_pro_init
    
    a_train_list = []
    a_valid_list = []
    branch_input_train_list = []
    branch_input_valid_list = []

    for _ in range(n_init_train):
        a_train_list.append(equation.a[None, :].tile((n_points_pro_init, 1, 1)))  # save the parameters for each initial profile
        branch_input_train_list.append(equation(0, sample_points[:,1], sample_points[:, 2])[None, :].tile((n_points_pro_init, 1)))
        equation.reset_a()
    a_train = torch.cat(a_train_list, dim = 0) #[n_points_pro_init * n_init , K, K]
    branch_input_train = torch.cat(branch_input_train_list, dim = 0) #[n_points_pro_init * n_init , n_samples**2]
    trunck_input_train = uniform(0, 1, size=(n_points_pro_init * n_init_train, 2)) #[n_points_pro_init * n_init, 2]   ignore [T]
    u_train = equation(T, trunck_input_train[:,0], trunck_input_train[:,1], a = a_train)[:, None] #[n_points_pro_init * n_init, 1]

    for _ in range(n_init_valid):
        a_valid_list.append(equation.a)
        branch_input_valid_list.append(equation(0, sample_points[:,1], sample_points[:, 2])[None, :].tile((n_points_pro_init, 1)))
        equation.reset_a()
    a_valid = torch.cat(a_valid_list, dim = 0) #[n_points_pro_init * n_init , K, K]
    branch_input_valid = torch.cat(branch_input_valid_list, dim = 0) #[n_points_pro_init * n_init , n_samples**2]
    trunck_input_valid = uniform(0, 1, size=(n_points_pro_init * n_init_valid, 2)) #[n_points_pro_init * n_init, 2]   ignore [T]
    u_valid = equation(T, trunck_input_valid[:,0], trunck_input_valid[:,1], a = a_valid)[:, None] #[n_points_pro_init * n_init, 1]

    plot_dataset(trunck_input_train, u_train, trunck_input_valid, u_valid, image_path="images/deeponet/wave_equation_data.png")
    normalizer = Normalizer(u_train)
    u_train = normalizer(u_train)
    u_valid = normalizer(u_valid)

    model = DeepONet(basis_size=n_samples**2, point_size=2, output_size=1, hidden_size=32, num_layers=3)
    fit(model,
        [branch_input_train, trunck_input_train], u_train,
        [branch_input_valid, trunck_input_valid], u_valid,
        epoch=100000,
        lr = 1e-5,
        weight_decay=1e-4,
        eval_every_eps=100,
        loss_image_path="images/deeponet/wave_equation_loss.png",
        device=device)
    
    torch.save(model.state_dict(), "weights/deeponet/weight.pth")
    
    # plot prediction
    X, Y = torch.meshgrid(torch.linspace(0, 1, 100), torch.linspace(0, 1, 100))
    trunck_input_pred = torch.stack([X.reshape(-1), Y.reshape(-1)], dim=-1)
    equation.reset_a()
    branch_input_pred = equation(0, sample_points[:,1], sample_points[:, 2])[None, :].tile((100*100, 1))
    with torch.no_grad():
        pred_u = model(branch_input_pred, trunck_input_pred)
    pred_u = normalizer.undo(pred_u)
    exact_u = equation(T, trunck_input_pred[:,0], trunck_input_pred[:,1])[:, None]

    fig, ax = plt.subplots(1, 3, figsize=(15,5))
    # assume d = 1
    scatter(trunck_input_pred[:,0], trunck_input_pred[:,1], pred_u.squeeze(), "Prediction", fig, ax[0])
    scatter(trunck_input_pred[:,0], trunck_input_pred[:,1], exact_u.squeeze(), "Exact", fig, ax[1])
    scatter(trunck_input_pred[:,0], trunck_input_pred[:,1], (pred_u-exact_u).squeeze(), "Error", fig, ax[2])
    
    fig.savefig("images/deeponet/wave_equation_prediction.png", dpi=400)


# FNO meshgrid input 
# TODO: train with more initial profiles (must be implemented with a dataloader)
def fno_fit_wave_equation(T:float = 1.0, K:int = 1, r:float = 0.85, c:float = 0.1,n_init = 4, n_train:int = 256, n_valid:int = 512):
    os.makedirs("images/fno", exist_ok=True)
    os.makedirs("weights/fno", exist_ok=True)

    equation = WaveEquation(K=K, r=r, c=c)
    
    train_x, train_y = torch.meshgrid(torch.linspace(0, 1, n_train), torch.linspace(0, 1, n_train))
    valid_x, valid_y = torch.meshgrid(torch.linspace(0, 1, n_valid), torch.linspace(0, 1, n_valid))
    
    train_input = []
    train_output = []
    valid_input = []
    valid_output = []
    
    for _ in range(n_init):
        train_x = uniform(0, 1, size=(n_train, n_train))
        train_y = uniform(0, 1, size=(n_train, n_train))
        train_u0 = equation(0, train_x.reshape(-1), train_y.reshape(-1)).reshape(n_train, n_train)[None, :]
        train_u = equation(T, train_x.reshape(-1), train_y.reshape(-1)).reshape(n_train, n_train)[None, :]
        train_input.append(torch.cat([train_x[None,:], train_y[None, :], train_u0], dim=0))
        train_output.append(train_u)
        equation.reset_a()

    for _ in range(n_init):
        valid_x = uniform(0, 1, size=(n_valid, n_valid))
        valid_y = uniform(0, 1, size=(n_valid, n_valid))
        valid_u0 = equation(0, valid_x.reshape(-1), valid_y.reshape(-1)).reshape(n_valid, n_valid)[None, :]
        valid_u = equation(T, valid_x.reshape(-1), valid_y.reshape(-1)).reshape(n_valid, n_valid)[None, :]
        valid_input.append(torch.cat([valid_x[None,:], valid_y[None, :], valid_u0], dim=0))
        valid_output.append(valid_u)

    train_input = torch.stack(train_input, dim=0)
    train_output = torch.stack(train_output, dim=0)
    valid_input = torch.stack(valid_input, dim=0)
    valid_output = torch.stack(valid_output, dim=0)
    
    normalizer = Normalizer(train_output, axis=(0, -2,-1))
    train_output = normalizer(train_output)
    valid_output = normalizer(valid_output)
    # print(train_input.shape)
    # print(train_output.shape)

    # plot_dataset(train_points.T.reshape(-1,2), 
    #              train_uT.reshape(d, -1).T, 
    #              valid_points.T.reshape(-1,2), 
    #              valid_uT.reshape(d, -1).T, 
    #              d, 
    #              image_path="images/fno/wave_equation_data.png")
    
    
    model = FNO2d(in_channel=3, out_channel=1, hidden_channel=32, num_layers=3)
    fit(model,
        train_input, train_output,
        valid_input, valid_output,
        epoch=10000,
        lr = 1e-3,
        weight_decay=1e-4,
        eval_every_eps=100,
        loss_image_path="images/fno/wave_equation_loss.png",
        device=device)
    
    torch.save(model.state_dict(), "weights/fno/weight.pth")

    # plot prediction
    X, Y = torch.meshgrid(torch.linspace(0, 1, 100), torch.linspace(0, 1, 100)) 
    equation.reset_a()
    pred_u0  = equation(T, X.reshape(-1), Y.reshape(-1)).reshape(100, 100)
    pred_input = torch.stack([X, Y, pred_u0], dim=0)[None, :]
    with torch.no_grad():
        pred_output = model(pred_input)
    pred_output = normalizer.undo(pred_output).reshape(-1)
    exact_output = equation(T, X.reshape(-1), Y.reshape(-1))
    fig, ax = plt.subplots(1, 3, figsize=(15,5))
    scatter(X.flatten(), Y.flatten(), pred_output, "Prediction", fig, ax[0])
    scatter(X.flatten(), Y.flatten(), exact_output, "Exact", fig, ax[1])
    scatter(X.flatten(), Y.flatten(), (pred_output - exact_output), "Error", fig, ax[2])
    fig.savefig("images/fno/wave_equation_prediction.png", dpi=400)


def cno_fit_wave_equation(T:float = 1.0, K:int = 1, r:float = 0.85, c:float = 0.1,n_init = 4, n_train:int = 256, n_valid:int = 512):
    os.makedirs("images/cno", exist_ok=True)
    os.makedirs("weights/cno", exist_ok=True)

    equation = WaveEquation(K=K, r=r, c=c)
    
    train_x, train_y = torch.meshgrid(torch.linspace(0, 1, n_train), torch.linspace(0, 1, n_train))
    valid_x, valid_y = torch.meshgrid(torch.linspace(0, 1, n_valid), torch.linspace(0, 1, n_valid))
    
    train_input = []
    train_output = []
    valid_input = []
    valid_output = []
    
    for _ in range(n_init):
        train_x = uniform(0, 1, size=(n_train, n_train))
        train_y = uniform(0, 1, size=(n_train, n_train))
        train_u0 = equation(0, train_x.reshape(-1), train_y.reshape(-1)).reshape(n_train, n_train)[None, :]
        train_u = equation(T, train_x.reshape(-1), train_y.reshape(-1)).reshape(n_train, n_train)[None, :]
        train_input.append(torch.cat([train_x[None,:], train_y[None, :], train_u0], dim=0))
        train_output.append(train_u)
        equation.reset_a()

    for _ in range(n_init):
        valid_x = uniform(0, 1, size=(n_valid, n_valid))
        valid_y = uniform(0, 1, size=(n_valid, n_valid))
        valid_u0 = equation(0, valid_x.reshape(-1), valid_y.reshape(-1)).reshape(n_valid, n_valid)[None, :]
        valid_u = equation(T, valid_x.reshape(-1), valid_y.reshape(-1)).reshape(n_valid, n_valid)[None, :]
        valid_input.append(torch.cat([valid_x[None,:], valid_y[None, :], valid_u0], dim=0))
        valid_output.append(valid_u)

    train_input = torch.stack(train_input, dim=0)
    train_output = torch.stack(train_output, dim=0)
    valid_input = torch.stack(valid_input, dim=0)
    valid_output = torch.stack(valid_output, dim=0)
    
    normalizer = Normalizer(train_output, axis=(0, -2,-1))
    train_output = normalizer(train_output)
    valid_output = normalizer(valid_output)

    model = CNO2d(in_channel=3, out_channel=1, hidden_size=32, num_layers=3)
    # hidden_size = hidden channel

    fit(model,
        train_input, train_output,
        valid_input, valid_output,
        epoch=10000,
        lr = 1e-3,
        weight_decay=1e-4,
        eval_every_eps=100,
        loss_image_path="images/cno/wave_equation_loss.png",
        device=device)
    
    torch.save(model.state_dict(), "weights/cno/weight.pth")

    # plot prediction
    X, Y = torch.meshgrid(torch.linspace(0, 1, 100), torch.linspace(0, 1, 100)) 
    equation.reset_a()
    pred_u0  = equation(T, X.reshape(-1), Y.reshape(-1)).reshape(100, 100)
    pred_input = torch.stack([X, Y, pred_u0], dim=0)[None, :]
    with torch.no_grad():
        pred_output = model(pred_input)
    pred_output = normalizer.undo(pred_output).reshape(-1)
    exact_output = equation(T, X.reshape(-1), Y.reshape(-1))
    fig, ax = plt.subplots(1, 3, figsize=(15,5))
    scatter(X.flatten(), Y.flatten(), pred_output, "Prediction", fig, ax[0])
    scatter(X.flatten(), Y.flatten(), exact_output, "Exact", fig, ax[1])
    scatter(X.flatten(), Y.flatten(), (pred_output - exact_output), "Error", fig, ax[2])
    fig.savefig("images/cno/wave_equation_prediction.png", dpi=400)
    



    pass
    

# %%
if __name__ == "__main__":
    # ffn_fit_wave_equation()
    # deeponet_fit_wave_equation()
    # fno_fit_wave_equation()
    cno_fit_wave_equation()
    