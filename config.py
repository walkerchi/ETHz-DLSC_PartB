import os
import argparse
import torch
import toml 
import json
import yaml

TASKS = ["train","predict","varying"]
SAMPLERS = ["mesh", "sobol", "uniform"]
MODELS = ["cno","ffn","deeponet","fno","kno","unet"]
EQUATIONS = ["heat","wave","poisson"]
EQUATION_KEY = {
    "heat":"d",
    "wave":"K",
    "poisson":"K"
}
EQUATION_VALUES = [1, 2, 4, 8, 16]
SPATIAL_SAMPLINGS = [16 * 16, 32 * 32, 64 * 64]
EQUATION_T = {
    "heat":0.005,
    "wave":5,
    "poisson":1 # it doesn't matter, should be none zero
}
D = [16, 32, 64]

"""
    API for config
    1. use_cmd_config()
        get config from command line
        # in xxx.py
        >>> config = use_cmd_config()
        # in terminal
        >>> python xxx.py --argument1 value1 --argument2 value2
    2. use_file_config()
        get config from toml/json/yaml file
        # in xxx.py
        >>> config = use_file_config("config.toml")
        OR
        >>> config = use_file_config("config.json")
        OR
        >>> config = use_file_config("config.yaml")
    3. use_kwarg_config(**kwargs)
        get config from dictionary
        # in xxx.py
        >>> config = use_kwarg_config(argument1=value1, argument2=value2)
"""

def add_arguments(parser):
    """
        default configuration,
        do the customization here!
    """
    parser.add_argument("-t","--task",type=str, default="train", choices=TASKS)
    parser.add_argument("-e","--equation", type=str,default="heat", choices=EQUATIONS)
    parser.add_argument("-m", "--model", type=str, default="ffn", choices=MODELS)
    parser.add_argument("--d", type=int, default=1, help="parameter of the heat equation")
    parser.add_argument("--K", type=int, default=1, help="parameter of the wave equation")
    parser.add_argument("--T", type=float, default=1.0)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--n_train_sample", type=int, default=64, help="number of training samples, normally how many sets of d to sample")
    parser.add_argument("--n_valid_sample", type=int, default=64, help="number of validation samples,  normally how many sets of d to sample")
    parser.add_argument("--n_train_spatial", type=int, default=256)
    parser.add_argument("--n_valid_spatial", type=int, default=1024)
    parser.add_argument("--n_eval_sample", type=int, default=64)
    parser.add_argument("--n_eval_spatial", type=int, default=10000)
    parser.add_argument("--num_hidden", type=int, default=64)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("-act","--activation", type=str, default="relu")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--eval_every_eps", type=int, default=100)
    parser.add_argument("--epoch", type=int, default=1000)
    parser.add_argument("--cuda", action="store_true", help="whether to use cuda")
    parser.add_argument("--sampler", type=str, default="mesh",  choices=SAMPLERS)
    parser.add_argument("--pin_memory",action="store_true", help="whether to use pin_memory in dataloader")
    parser.add_argument("--seed", type=int, default=1234, help="random seed")
    # only for cno
    parser.add_argument("--jit", action="store_true", help="whether to use jit in cno")
    # only for fno
    parser.add_argument("--modes", type=int, default=None, help="number of modes for fno")
    # only for deeponet
    parser.add_argument("--n_basis_spatial", type=int, default=1024, help="especially for deeponet")
    parser.add_argument("--branch_sampler", type=str, default="mesh", choices=SAMPLERS, help = "only for deeponet")
    parser.add_argument("--trunk_sampler", type=str, default="mesh", choices=SAMPLERS, help="only for deeponet")
    parser.add_argument("--branch_arch", type=str, default="mlp", choices=["mlp", "resnet", "fno"], help="architecture of branch network for deeponet")
    parser.add_argument("--trunk_arch", type=str, default="mlp", choices=["mlp", "resnet", "fno"], help="architecture of trunk network for deeponet")


"""
    Configuartion Classes
    not  important
"""
class Config(object):
    def __getitem__(self, key):
        return getattr(self, key)
    def __setitem__(self, key, value):
        return setattr(self, key, value)


class ConfigItem:
    def __init__(self, key, type=str, action=None, default=None, choices=None, **kwargs):
        if action == "store_true":
            type    = bool
            default = False
        if default  is not None  and choices is not None:
            assert default in choices, f"{default} is not in {choices}"
        self.key = key
        self.type = type
        self.default = type(default) if default is not None else default
        self.choices = choices

    def assert_within_choices(self, value):
        if self.choices is not None:
            assert value in self.choices, f"{value} is not in {self.choices}"


class KwargsParser:
    def __init__(self):
        self.arguments = []
    def add_argument(self, *args, **kwargs):
        for arg in args:
            if arg.startswith("--"):
                self.arguments.append(ConfigItem(arg[2:], **kwargs))
    def parse_args(self, **kwargs):
        config  = Config()
        for arg in self.arguments:
            if arg.key in kwargs:
                arg.assert_within_choices(kwargs[arg.key])
                setattr(config,  arg.key, arg.type(kwargs[arg.key]))
            else:
                setattr(config, arg.key, arg.default)
        return config

class TomlParser(KwargsParser):
    def parse_args(self, config_path):
        with open(config_path, "r") as f:
            toml_config = toml.load(f)
        return super().parse_args(**toml_config)
    
class JsonParser(KwargsParser):
    def parse_args(self, config_path):
        with open(config_path, "r") as f:
            json_config = json.load(f)
        return  super().parse_args(**json_config)

class YamlParser(KwargsParser):
    def parse_args(self, config_path):
        with open(config_path, "r") as f:
            yaml_config = yaml.load(f)
        return super().parse_args(**yaml_config)





def adjust_config(config):
    """
        adjust config, do some post-processing
    """
    if config.cuda:
        config.device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    else:
        config.device = torch.device("cpu")
    return config


"""

    API for config

"""

def use_cmd_config():
    cmd_parser = argparse.ArgumentParser()
    add_arguments(cmd_parser)
    config = cmd_parser.parse_args()
    config = adjust_config(config)
    return config

def use_file_config(filepath):

    if filepath.endswith(".toml"):
        parser = TomlParser()
    elif filepath.endswith(".json"):
        parser = JsonParser()
    elif filepath.endswith(".yaml") or filepath.endswith(".yml"):
        parser = YamlParser()

    add_arguments(parser)

    filepath = os.path.abspath(filepath)
    config = parser.parse_args(filepath)
    config = adjust_config(config)

    return config

def use_kwarg_config(**kwargs):
    parser = KwargsParser()
    add_arguments(parser)
    config = parser.parse_args(**kwargs)
    config = adjust_config(config)
    return config