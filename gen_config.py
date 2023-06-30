import os 
import torch
import toml
from itertools import product
from tqdm import tqdm

from config import EQUATIONS, EQUATION_T, EQUATION_KEYS, EQUATION_VALUES, MODELS

def gen_model(model):
    return f"""
# for model 
model          = "{model}"
num_hidden     = 64
num_layers     = 4
activation     = "relu"
    """

def gen_equation(equation, T, **kwargs):
    if  len(kwargs) == 1:
        k, v = list(kwargs.items())[0]
        return f"""
# for equation
equation       = "{equation}"
{k}              = {v}      # source component dimension
T              = {T}    # prediction time
    """ 
    elif len(kwargs) == 0:
        return f"""
# for equation
equation       = "{equation}"
T              = {T}    # prediction time
    """
    else:
        raise NotImplementedError()

def gen_task(task):
    return f"""
# for task
task           = "{task}"
    """

def gen_training(epoch = 1000, 
                 sampler = "mesh", 
                 batch_size = 64, 
                 n_train_sample = 64, 
                 n_valid_sample = 64, 
                 n_eval_sample = 64,
                 n_train_spatial = 4096, 
                 n_valid_spatial = 4096, 
                 n_eval_spatial = 4096):
    return f"""
# for training
epoch          = {epoch}
sampler        = "{sampler}"
batch_size     = {batch_size}
n_train_sample = {n_train_sample}     # how many mu to sample for training
n_valid_sample = {n_valid_sample}
n_eval_sample  = {n_eval_sample}
n_train_spatial = {n_train_spatial}   # how many spatial points to sample for training
n_valid_spatial = {n_valid_spatial}
n_eval_spatial  = {n_eval_spatial}
    """

def gen_predict(sampler = "mesh",
             n_eval_spatial = 4096):
    return f"""
# for evaluation
sampler        = "{sampler}"
n_eval_spatial = {n_eval_spatial}    
"""

def gen_eval(sampler = "mesh",
             batch_size = 64,
             n_eval_sample = 64,
             n_eval_spatial = 4096):
    return f"""
# for evaluation
sampler        = "{sampler}"
batch_size     = {batch_size}
n_eval_sample  = {n_eval_sample}
n_eval_spatial = {n_eval_spatial}
"""

def gen_device():
    use_cuda = torch.cuda.is_available()
    return f"""
# for device
cuda            = {"true" if use_cuda else "false"} # use CPU
    """

def gen_train_config(batch_size=64):
    n_train_spatial = 4096
    for equation, v, model in tqdm(product(EQUATIONS, EQUATION_VALUES, MODELS), desc="generating training config"):
        config = gen_task("train"
                )+ gen_model(model
                ) + gen_equation(equation, 
                    EQUATION_T[equation],
                    **{EQUATION_KEYS[equation]:v}
                ) + gen_training(
                    batch_size = batch_size * n_train_spatial if model == "ffn" 
                            else int(batch_size / 4) if model  == "cno" 
                            else batch_size,
                    n_train_spatial = n_train_spatial
                ) + gen_device()
        dirpath = f"config/train/{equation}_{EQUATION_KEYS[equation]}={v}"
        os.makedirs(dirpath, exist_ok=True)
        with open(os.path.join(dirpath, f"{model}.toml"), "w") as f:
            f.write(config)

def gen_predict_config(batch_size=64):
    for equation,v, model in tqdm(product(EQUATIONS, EQUATION_VALUES, MODELS), desc="generating predict config"):
        config = gen_task("predict"
                )+ gen_model(model
                ) + gen_equation(equation, 
                    EQUATION_T[equation],
                    **{EQUATION_KEYS[equation]:v}
                ) + gen_predict(
                ) + gen_device()
        dirpath = f"config/predict/{equation}_{EQUATION_KEYS[equation]}={v}"
        os.makedirs(dirpath, exist_ok=True)
        with open(os.path.join(dirpath, f"{model}.toml"), "w") as f:
            f.write(config)

def gen_varying_config(batch_size=64):
    for equation, model in tqdm(product(EQUATIONS, MODELS), desc="generating varying config"):
        config = gen_task("predict"
                    )+ gen_model(model
                    ) + gen_equation(equation, 
                        EQUATION_T[equation]
                    ) + gen_eval(
                    ) + gen_device()
        dirpath = f"config/varying/{equation}"
        os.makedirs(dirpath, exist_ok=True)
        with open(os.path.join(dirpath, f"{model}.toml"), "w") as f:
            f.write(config)

def gen_config(batch_size=64):
    """
        batch_size  = 64 for 2GB memory
    """
    gen_train_config(batch_size=batch_size)
    gen_predict_config(batch_size=batch_size)
    gen_varying_config(batch_size=batch_size)

if __name__ == '__main__':
    gen_config()