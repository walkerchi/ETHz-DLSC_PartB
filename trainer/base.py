import os
import pandas as pd
import numpy as np
import torch 
import torch.nn as nn
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import random
from itertools import chain
from tqdm import tqdm
from torch.utils.data import DataLoader, TensorDataset
from mpl_toolkits.axes_grid1 import make_axes_locatable
from itertools import product


from equations import EquationLookUp
from models import ModelLookUp
from config import MODELS,  EQUATION_KEY, EQUATION_VALUES
def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

def to_device(x,  device):
    if isinstance(x, torch.Tensor):
        return x.to(device, non_blocking=True) if x.device != device is not None else x
    elif isinstance(x, (list,  tuple)):
        return [to_device(x_, device) for x_ in x]
    elif isinstance(x, dict):
        return {k:to_device(v, device) for k,v in x.items()}
    else:
        raise NotImplementedError(f"to_device not implemented for {type(x)}")
    
def general_call(func, args):
    if isinstance(args, (list, tuple)):
        return func(*args)
    elif isinstance(args, dict):
        return func(**args)
    else:
        return func(args)

def scatter(x, y, c, title, fig, ax, xlims, cmap="jet"):
    h = ax.scatter(x, y, c=c, cmap=cmap)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(h, cax=cax, orientation='vertical')
    ax.set_xlim(xlims[0][0], xlims[0][1])
    ax.set_ylim(xlims[1][0], xlims[1][1])
    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$")
    ax.set_title(title)

def scatter_error2d(x, y, prediction, exact, image_path, xlims, **kwrags):
    os.makedirs(image_path, exist_ok=True)

    fig, ax = plt.subplots(1, 3, figsize=(15,5))
    scatter(x, y, prediction.flatten(), "Prediction", fig, ax[0], xlims)
    scatter(x, y, exact.flatten(), "Exact", fig, ax[1], xlims)
    scatter(x, y, (prediction-exact).flatten(), "Error", fig, ax[2], xlims)
    fig.savefig(os.path.join(image_path, "comparison.png"), dpi=400)
    plt.close(fig=fig)

    fig, ax = plt.subplots(figsize=(6, 6))
    scatter(x, y, prediction.flatten(), "Prediction", fig, ax, xlims)
    fig.savefig(os.path.join(image_path, "prediction.png"), dpi=400)
    fig.savefig(os.path.join(image_path, "prediction.pdf"), dpi=400)
    plt.close(fig=fig)

    fig, ax = plt.subplots(figsize=(6, 6))
    scatter(x, y, exact.flatten(), "Exact", fig, ax, xlims)
    fig.savefig(os.path.join(image_path, "uT.png"), dpi=400)
    fig.savefig(os.path.join(image_path, "uT.pdf"), dpi=400)
    plt.close(fig=fig)

    fig, ax = plt.subplots(figsize=(6, 6))
    scatter(x, y, (prediction - exact).flatten(), "Error", fig, ax, xlims, cmap="seismic")
    fig.savefig(os.path.join(image_path, "error.png"), dpi=400)
    fig.savefig(os.path.join(image_path, "error.pdf"), dpi=400)
    plt.close(fig=fig)

    for k,v in kwrags.items():
        if  isinstance(v, (tuple,  list)):
            x_local, y_local, v = v
        else:
            x_local, y_local = x, y
        fig, ax = plt.subplots(figsize=(6, 6))
        scatter(x_local, y_local, v.flatten(), k, fig, ax, xlims)
        fig.savefig(os.path.join(image_path, f"{k}.png"), dpi=400)
        fig.savefig(os.path.join(image_path, f"{k}.pdf"), dpi=400)
        plt.close(fig=fig)



class SpatialSampler:
    """
        Sample spatial points from domain
        Equation:  Equation class
        sampler: "mesh" or "sobol" or "uniform"

        Usage:
            >>> sampler = SpatialSampler(HeatEquation, sampler="mesh")
            >>> points = sampler(100, flattern=False)
            >>> points.shape
            torch.Size([100, 100, 2])
            >>> points = sampler(100, flattern=True)
            >>> points.shape
            torch.Size([10000, 2])
    """
    def __init__(self, Equation, sampler="mesh"):

        assert sampler  in ["mesh", "sobol", "uniform"]
        dimension = Equation.x_domain.shape[0]
        self.dimension = dimension
        self.sampler = sampler 
        self.domain  = Equation.x_domain

        if sampler == "sobol":
            self.sample_engine = torch.quasirandom.SobolEngine(dimension, scramble=True)

    def __call__(self,n_points,flatten=False):
        """
            Parameters:
            -----------
                n_points: int
                flattern: bool, if True, and sampler is "mesh" return shape=(n_points*dimension, dimension)
            Returns:
            --------
                points: torch.Tensor, 
                    if sampler is "mesh" and not flatten
                        shape=(axis_0, ... , axis_{dimension-1} , dimension)
                    elif sampler is "mesh" and flattern
                        shape=(n_points, dimension)
                    elif sampler is "sobol" or "uniform"
                        shape=(n_points, dimension)
        """

        if self.sampler == "mesh":
            axis_n_points = int(np.power(n_points, 1/self.dimension))
            points = torch.meshgrid(*[torch.linspace(*self.domain[i], axis_n_points) for i in range(self.dimension)])
            points = torch.stack(points, dim=-1)
            if flatten:
                points = points.reshape(-1, self.dimension)

        elif self.sampler == "sobol":
            points = self.sample_engine.draw(n_points)
            points = points * (self.domain[:,1]-self.domain[:,0]) + self.domain[:,0]

        elif self.sampler == "uniform":
            points = torch.rand(size=(n_points, self.dimension)) * (self.domain[:,1]-self.domain[:,0]) + self.domain[:,0]
        
        return points


class DatasetGeneratorBase:
    def __call__(self):
        raise NotImplementedError()
    def load(self,  path):
        pass
    def save(self, path):
        pass

class NormalizerBase:
    @staticmethod
    def init():
        raise NotImplementedError()
    def __call__(self):
        raise NotImplementedError()
    def norm_input(self):
        raise NotImplementedError()
    def unorm_output(self):
        raise NotImplementedError()
    @staticmethod
    def load():
        raise NotImplementedError()
    def save(self):
        raise NotImplementedError()

class DataLoaderBase(DataLoader):
    def __init__(self, input, output, batch_size, device, shuffle=False, **kwargs):
        self.device = device
        self.input  = to_device(input, device) 
        self.output = to_device(output, device)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.counter = None
    def __iter__(self):
        if self.shuffle:
            index = torch.randperm(self.input.shape[0])
            self.input  = self.input[index]
            self.output = self.output[index]
        self.counter = 0
        return self
    def __next__(self):
        if self.counter >= self.input.shape[0]:
            raise StopIteration
        else:
            batch_input  = self.input[self.counter:self.counter+self.batch_size]
            batch_output = self.output[self.counter:self.counter+self.batch_size]
            self.counter += self.batch_size
            return batch_input, batch_output
         
            
class TrainerBase:
    DataLoader = None 
    Normalizer = None

    def __init__(self, config):
        self.config           = config

        self.xlims            = None  # define at initialization

        self.model            = None   # define at initialization
        self.dataset_generator = None  # define at initialization
        
        self.image_path       = None   # define at initialization
        self.weight_path      = None   # define at initialization
        self.table_path       = None   # define at initialization

        self.normalizer       = None   # define at fit

        raise  NotImplementedError()

    def to(self,device):
        self.model = self.model.to(device)

    def fit(self):

        os.makedirs(self.image_path, exist_ok=True)
        os.makedirs(self.weight_path, exist_ok=True)

        config = self.config

        if config.pin_memory:
            kwargs = {
                "pin_memory":True,
            }
        else:
            kwargs = {}
    
        
        train_dataset     = self.dataset_generator(config.n_train_sample, config.n_train_spatial)
        valid_dataset     = self.dataset_generator(config.n_valid_sample, config.n_valid_spatial)
        normalizer        = self.Normalizer.init(*train_dataset)
        self.normalizer   = normalizer
        train_dataset     = normalizer(*train_dataset)
        valid_dataset     = normalizer(*valid_dataset)
        train_dataloader  = self.DataLoader(*train_dataset,
                                    batch_size=config.batch_size, 
                                    shuffle=True,
                                    device = config.device,
                                    **kwargs)
        valid_dataloader  = self.DataLoader(*valid_dataset,
                                    batch_size=config.batch_size,
                                    shuffle=True,
                                    device = config.device,
                                     **kwargs)
        
        model = self.model.to(config.device)
        losses = {"train":[],"valid":[]}
        best_weight, best_loss, best_epoch = None, float('inf'), None
        optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
        criterion = nn.MSELoss()
        
        p = tqdm(range(config.epoch))
    
        for ep in p:
            # training step
            model.train()
            
            iteration_losses = []
            for input_batch, output_batch in train_dataloader:

                input_batch, output_batch = to_device([input_batch, output_batch], config.device)

                optimizer.zero_grad()
        
                prediction = general_call(model, input_batch)

                loss = criterion(prediction, output_batch)
                loss.backward()
                optimizer.step()

                iteration_losses.append(loss.item())

            iteration_loss = np.mean(iteration_losses)

            # record for display
            losses['train'].append((ep,iteration_loss))
            p.set_postfix({'loss': iteration_loss})

            if (ep+1) % config.eval_every_eps == 0:
                # validation every eval_every_eps epoch
                model.eval()

                with torch.no_grad():

                    iteration_losses = []
                    for input_batch, output_batch in valid_dataloader:

                        input_batch, output_batch = to_device([input_batch, output_batch], config.device)
                        
                        prediction = general_call(model, input_batch)
                        
                        iteration_losses.append(criterion(prediction, output_batch).item())

                    valid_loss = np.mean(iteration_losses)
                    losses['valid'].append((ep,valid_loss))

                    # save best valid loss weight
                    if valid_loss < best_loss:
                        best_weight = model.state_dict()
                        best_loss = valid_loss.item()
                        best_epoch = ep

        # load the best recorded weight
        if best_weight is not None:
            model.load_state_dict(best_weight)

        model.eval()
        model = model.cpu()

        # plot loss
        self.plot_loss(losses['train'], losses['valid'], best_epoch, best_loss)

        self.model = model
    
    def eval(self):
        """
            Returns:
            --------
                position:  torch.Tensor, shape=(n_eval_sample, n_eval_spatial, 2)
                l2_errors: torch.Tensor, shape=(n_eval_sample, n_eval_spatial)
        """
        raise NotImplementedError()

    def plot_loss(self, train_losses, valid_losses, best_epoch, best_loss):

        fig, ax = plt.subplots(1, 1, figsize=(10,5))
        ax.plot([x[0] for x in train_losses],[x[1] for x in train_losses], label="Train Loss",  linestyle="--", alpha=0.6)
        ax.scatter([x[0] for x in valid_losses],[x[1] for x in valid_losses], label="Valid Loss",  alpha=0.6, color="orange")
        ax.scatter(best_epoch, best_loss, label="Best Valid Loss", marker="*", s=200, c="red")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_title("Loss")
        ax.set_yscale("log")
        ax.legend()
        fig.savefig(os.path.join(self.image_path, "loss.png"), dpi=400)
        plt.close(fig=fig)
        
    def plot_varying(self, eval_results, **kwargs):
        """
            eval_results: list of (position, prediction, output)
            kwargs: {varying_key: varying_values}
        """
        sns.set_theme()
        
        assert len(kwargs) == 1
        k, vs = list(kwargs.items())[0]
        df = {k:[], "relative error":[], "l2 error":[]}
        for v, (position, prediction, output) in zip(vs,eval_results):
            prediction = prediction.numpy()
            output     = output.numpy()
            df[k].append(np.full([len(prediction)], v))
            df["relative error"].append((np.abs(prediction-output)/np.abs(output)).mean(-1))
            df["l2 error"].append(np.sqrt(np.mean(np.square(prediction-output), -1)))
        for key, value in df.items():
            df[key] = np.concatenate(value)
        df = pd.DataFrame.from_dict(df)
  
        path = f"images/{self.config.equation}/{self.config.model}"
      
        fig = sns.lmplot(x=k, y="l2 error", data=df, y_jitter=.02, logistic=True, truncate=False)
       
        os.makedirs(path, exist_ok=True)
        fig.savefig(os.path.join(path, f"varying.png"), dpi=300)
        fig.savefig(os.path.join(path, f"varying.pdf"), dpi=300)

    def plot_varying_together(self, predictions, outputs):
        sns.set(font_scale=1.5)
        predictions = torch.stack(predictions, 0).reshape(len(EQUATION_VALUES), len(MODELS), self.config.n_eval_sample, self.config.n_eval_spatial) # [n_values, n_model, n_sample, n_spatial]
        outputs     = torch.stack(outputs,     0).reshape(len(EQUATION_VALUES), len(MODELS), self.config.n_eval_sample, self.config.n_eval_spatial) # [n_values, n_model, n_sample, n_spatial]

        key    = EQUATION_KEY[self.config.equation]
        errors = ((predictions-outputs)**2).mean(-1).numpy() # [n_values, n_model, n_sample]
        labels = np.meshgrid(EQUATION_VALUES, MODELS, np.arange(self.config.n_eval_sample))
     
        df = pd.DataFrame.from_dict({
            key:labels[0].flatten(),
            "Model":labels[1].flatten(),
            "Sample":labels[2].flatten(),
            "L2 Error":errors.flatten()
        })

        fig = sns.lmplot(x=key, y="L2 Error", hue="Model", data=df, 
                         scatter = False,
                         y_jitter=.02, logistic=True, truncate=False)
        
        path = f"images/{self.config.equation}"

        os.makedirs(path, exist_ok=True)
        fig.savefig(os.path.join(path, f"varying.png"), dpi=300)
        fig.savefig(os.path.join(path, f"varying.pdf"), dpi=300)
        

    def table_varying_together(self, predictions, outputs):
        predictions = torch.stack(predictions, 0).reshape(len(EQUATION_VALUES), len(MODELS), self.config.n_eval_sample, self.config.n_eval_spatial) # [n_values, n_model, n_sample, n_spatial]
        outputs     = torch.stack(outputs,     0).reshape(len(EQUATION_VALUES), len(MODELS), self.config.n_eval_sample, self.config.n_eval_spatial) # [n_values, n_model, n_sample, n_spatial]

        key    = EQUATION_KEY[self.config.equation]
        errors = ((predictions-outputs)**2).mean(-1).numpy() # [n_values, n_model, n_sample]
        errors_mean = errors.mean(-1) # [n_values, n_model]
        errors_std  = errors.std(-1)  # [n_values, n_model]
        data       = np.zeros_like(errors_mean, dtype=str)
        for i,(mean,std) in enumerate(zip(errors_mean.flat, errors_std.flat)):
            data.flat[i] = f"{mean:.2e} ($\pm$ {std:.2e})"
        df = pd.DataFrame(data=data, columns=MODELS, index=EQUATION_VALUES)
        df.index.name = key

        path = f"tables/{self.config.equation}"

        df.to_latex(os.path.join(path, "varying.tex"))
        df.to_markdown(os.path.join(path, "varying.md"))
        df.to_csv(os.path.join(path, "varying.csv"))

    def plot_prediction_together(self, u0s, predictions,  uTs, plot_error=True):
        values = list(u0s.keys())
        nrow = len(values) 
        ncol = 1 + 1 + len(MODELS) # u0, uT, models
        
        predictions = list(chain.from_iterable(predictions.values()))
        uTs         = list(chain.from_iterable(uTs.values()))
        u0s         = list(chain.from_iterable(u0s.values()))
        # breakpoint()
        prediction = torch.stack(predictions, dim=0).reshape(nrow, len(MODELS), -1) # [nrow, n_model, n_spatial]
        uT = torch.stack([uTs[i*len(MODELS)+2] for i in range(nrow)], dim=0).reshape(nrow, -1) # get the uT for cno as col 1 [nrow, n_spatial]
        u0 = torch.stack([u0s[i*len(MODELS)+2] for i in range(nrow)], dim=0).reshape(nrow, -1) # get the u0 for cno as col 0 [nrow, n_spatial]
        error  = (prediction - uT[:, None, :])**2

        mesh_axis = int(np.sqrt(self.config.n_eval_spatial))
        assert mesh_axis * mesh_axis == self.config.n_eval_spatial
        prediction = prediction.reshape(nrow, len(MODELS), mesh_axis, mesh_axis)
        uT         = uT.reshape(nrow, mesh_axis, mesh_axis)
        u0         = u0.reshape(nrow, mesh_axis, mesh_axis)
        error      = error.reshape(nrow, len(MODELS), mesh_axis, mesh_axis)

        if  plot_error:
            nrow *= 2
            row_iter = range(0, nrow, 2)
        else:
            row_iter = range(nrow)
        fig, axes = plt.subplots(nrows=nrow, ncols=ncol, figsize=(2*ncol, 2*nrow))
        for ax in axes.flat:
            ax.axis("off")
        

        def add_right_cax(ax):
            divider = make_axes_locatable(ax)
            cax = divider.append_axes('right', size='5%', pad=0.05)
            return cax 
        
        def add_left_cax(ax):
            divider = make_axes_locatable(ax)
            cax = divider.append_axes('left', size='5%', pad=0.05)
            return cax


        # u0
        
        for i,irow in enumerate(row_iter):
            vmin,vmax = u0[i].min(), u0[i].max()
            im = axes[irow, 0].imshow(u0[i], vmin=vmin, vmax=vmax, cmap="jet")
            cax = add_left_cax(axes[irow, 0])
            cbar = fig.colorbar(im, cax=cax)
            cbar.ax.yaxis.set_ticks_position('left')

        
        for i,irow in enumerate(row_iter):
            vmin_uT, vmax_uT = uT[i].min(), uT[i].max()
            vmin_pr, vmax_pr = prediction[i].min(), prediction[i].max()
            vmin, vmax = min(vmin_uT, vmin_pr), max(vmax_uT, vmax_pr)

            # uT 
            im = axes[irow, 1].imshow(uT[i], vmin=vmin, vmax=vmax, cmap="jet")
            # prediction
            for j,icol in enumerate(range(2,ncol)):
                im = axes[irow, icol].imshow(prediction[i, j], vmin=vmin, vmax=vmax, cmap="jet")
        
            cax = add_right_cax(axes[irow, -1])
            fig.colorbar(im, cax=cax)
       
        if plot_error:
            # error
            
            for i,irow  in enumerate(range(1, nrow, 2)):
                vmin, vmax = error[i].min(), error[i].max()
                for j,icol in enumerate(range(2,ncol)):
                    im = axes[irow, icol].imshow(error[i, j], vmin=vmin, vmax=vmax, cmap="seismic")
                   
                cax = add_right_cax(axes[irow, -1])
                fig.colorbar(im, cax=cax)
        
        # set column name
        axes[0, 0].set_title("$u_0$", fontsize=18)
        axes[0, 1].set_title("$u_T$", fontsize=18)
        for icol, model in enumerate(MODELS):
            axes[0, icol+2].set_title(model, fontsize=18)

        # set row name
        for i,irow in enumerate(row_iter):
            axes[irow, 0].axis('on')
            axes[irow, 0].set_xticks([])
            axes[irow, 0].set_yticks([])
            axes[irow, 0].set_ylabel(f"{EQUATION_KEY[self.config.equation]} = {values[i]}", rotation=0, labelpad=60, fontsize=16)
        if plot_error:
            for i, irow in enumerate(range(1, nrow, 2)):
                axes[irow, 2].axis('on')
                axes[irow, 2].set_xticks([])
                axes[irow, 2].set_yticks([])
                axes[irow, 2].set_ylabel(f"error", rotation=0, labelpad=30, fontsize=16)

        # fig.tight_layout()
       

        path = f"images/{self.config.equation}"
       
        os.makedirs(path, exist_ok=True)

        if EQUATION_VALUES == values:
            fig.savefig(os.path.join(path, f"predict.png"), dpi=300)
            fig.savefig(os.path.join(path, f"predict.pdf"), dpi=300)
        else:
            assert len(values) == 1
            fig.savefig(os.path.join(path, f"predict_{EQUATION_KEY[self.config.equation]}={values[0]}.png"), dpi=300)
            fig.savefig(os.path.join(path, f"predict_{EQUATION_KEY[self.config.equation]}={values[0]}.pdf"), dpi=300)

        plt.close(fig=fig)

    def table_prediction_together(self, predictions,  uTs):

        prediction = torch.stack(predictions, dim=0).reshape(len(EQUATION_VALUES), len(MODELS), -1) # [n_value, n_model, n_spatial]
        uT         = torch.stack(uTs,         dim=0).reshape(len(EQUATION_VALUES), len(MODELS), -1) # [n_value, n_model, n_spatial]
        error      = (prediction - uT)**2 # [n_value, n_model, n_spatial]
        error      = error.mean(-1) # [n_value, n_model]

        df = pd.DataFrame(error.numpy(), columns=MODELS, index=EQUATION_VALUES)
        df.index.name = EQUATION_KEY[self.config.equation]

        path = f"tables/{self.config.equation}"
        os.makedirs(path, exist_ok=True)
       
        df.to_latex(os.path.join(path, "predict.tex"), float_format="%.2e")
        df.to_markdown(os.path.join(path, "predict.md"))
        df.to_csv(os.path.join(path, "predict.csv"))



    def save(self):
        self.normalizer.save(os.path.join(self.weight_path, "normalizer.pth"))
        self.dataset_generator.save(os.path.join(self.weight_path, "dataset_generator.pth"))
        torch.save(self.model.state_dict(), os.path.join(self.weight_path, "model.pth"))

    def load(self):
        self.normalizer = self.Normalizer.load(os.path.join(self.weight_path, "normalizer.pth"))
        self.dataset_generator.load(os.path.join(self.weight_path, "dataset_generator.pth"))
        self.model.load_state_dict(torch.load(os.path.join(self.weight_path, "model.pth")))

