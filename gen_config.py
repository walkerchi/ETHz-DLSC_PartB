import os 
import torch
import toml
import argparse
from itertools import product
from tqdm import tqdm


from config import EQUATIONS, EQUATION_T, EQUATION_KEY, EQUATION_VALUES, MODELS, SPATIAL_SAMPLINGS

SEED = 1234
CNO_BATCH_SIZE_SCALE = 1
CNO_N_LAYER = 2
CNO_N_HIDDEN = 8
CNO_JIT     = True
FFN_N_HIDDEN = 64
DEFAULT_N_LAYER = 4
DEFAULT_N_HIDDEN = 64

"""
    config generaton helper functions
"""

def gen_model(model, n_layers=4, n_hidden=64):
    return f"""
# for model 
model          = "{model}"
num_hidden     = {n_hidden}
num_layers     = {n_layers}
activation     = "relu"
jit            = {"true" if CNO_JIT else "false"}
    """

def gen_equation(equation, T, **kwargs):
    if  len(kwargs) == 1:
        k, v = list(kwargs.items())[0]
        return f"""
# for equation
equation = "{equation}"
{k}      = {v}      # source component dimension
T        = {T}    # prediction time
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
seed           = {SEED}
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
pin_memory      = true
    """

"""
    config generaton main functions
"""

def gen_train_config(batch_size=64):
    n_spatial = 4096
    n_sample  = 64
    for equation, v, model in tqdm(product(EQUATIONS, EQUATION_VALUES, MODELS), total=len(EQUATIONS) *  len(MODELS) * len(EQUATION_VALUES), desc="generating training config"):
        config = gen_task("train"
                )+ gen_model(model,
                    n_layers = CNO_N_LAYER if model == "cno" or model == "unet" 
                                else DEFAULT_N_LAYER,
                    n_hidden = CNO_N_HIDDEN if model == "cno" or model == "unet"
                                else FFN_N_HIDDEN if model == "ffn"
                                else DEFAULT_N_HIDDEN
                ) + gen_equation(equation, 
                    EQUATION_T[equation],
                    **{EQUATION_KEY[equation]:v}
                ) + gen_training(
                    n_train_sample= n_sample, 
                    n_valid_sample= n_sample, 
                    n_eval_sample = n_sample, 
                    n_train_spatial=n_spatial, 
                    n_valid_spatial=n_spatial, 
                    n_eval_spatial=n_spatial, 
                    batch_size = batch_size * n_spatial if model == "ffn" 
                            else int(batch_size * CNO_BATCH_SIZE_SCALE) if model  == "cno" or model == "unet"
                            else batch_size,
                ) + gen_device()
        dirpath = f"config/train/{equation}_{EQUATION_KEY[equation]}={v}"
        os.makedirs(dirpath, exist_ok=True)
        with open(os.path.join(dirpath, f"{model}.toml"), "w") as f:
            f.write(config)

def gen_predict_config():
    for equation,v,model in tqdm(product(EQUATIONS,EQUATION_VALUES, MODELS), total=len(EQUATIONS) *  len(MODELS) * len(EQUATION_VALUES), desc="generating predict config"):
        config = gen_task("predict"
                )+ gen_model(model,
                    n_layers = CNO_N_LAYER if model == "cno" or model == "unet"
                                else DEFAULT_N_LAYER,
                    n_hidden = CNO_N_HIDDEN if model == "cno" or model == "unet"
                                else FFN_N_HIDDEN if model == "ffn"
                                else DEFAULT_N_HIDDEN
                ) + gen_equation(equation, 
                    EQUATION_T[equation],
                     **{EQUATION_KEY[equation]:v}
                ) + gen_predict(
                ) + gen_device()
        dirpath = f"config/predict/{equation}_{EQUATION_KEY[equation]}={v}"
        os.makedirs(dirpath, exist_ok=True)
        with open(os.path.join(dirpath, f"{model}.toml"), "w") as f:
            f.write(config)

def gen_varying_config(batch_size=64):
    n_spatial = 4096
    for equation, v, model in tqdm(product(EQUATIONS, EQUATION_VALUES, MODELS),total=len(EQUATIONS) *  len(MODELS),  desc="generating varying config"):
        config = gen_task("varying"
                    )+ gen_model(model,
                        n_layers =CNO_N_LAYER if model == "cno" or model == "unet" 
                                else DEFAULT_N_LAYER,
                        n_hidden = CNO_N_HIDDEN if model == "cno" or model == "unet"
                                else FFN_N_HIDDEN if model == "ffn"
                                else DEFAULT_N_HIDDEN
                    ) + gen_equation(equation, 
                        EQUATION_T[equation],
                         **{EQUATION_KEY[equation]:v}
                    ) + gen_eval(
                            n_eval_sample= batch_size, 
                            n_eval_spatial=n_spatial, 
                            batch_size = batch_size * n_spatial if model == "ffn" 
                            else int(batch_size * CNO_BATCH_SIZE_SCALE) if model  == "cno" or model == "unet"
                            else batch_size,
                    ) + gen_device()
        dirpath = f"config/varying/{equation}_{EQUATION_KEY[equation]}={v}"
        os.makedirs(dirpath, exist_ok=True)
        with open(os.path.join(dirpath, f"{model}.toml"), "w") as f:
            f.write(config)

def gen_config(batch_size=64):
    """
        batch_size  = 64 for 2GB memory
    """
    gen_train_config(batch_size=batch_size)
    gen_predict_config()
    gen_varying_config(batch_size=batch_size)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-m","--memory", default=2, type=int, help="memory size in GB")
    parser.add_argument("-b","--batch_size", default=None, type=int, help="batch size")
    args = parser.parse_args()
    if args.batch_size is None:
        args.batch_size = int(32 * args.memory)
    gen_config(args.batch_size)