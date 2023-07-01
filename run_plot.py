
import os
import re
import argparse
from config import use_file_config
from run_cmd import main
from run_folder import walk_config


def run_predict(model = None):
    config_paths = walk_config("config/predict")
    if model is not None:
        config_paths = [config_path for config_path in config_paths if config_path.endswith(f"{model}.toml")]
    image_paths  = [config_path.replace("config/predict","images").replace(".toml","/error.pdf") for config_path in config_paths]
    for i,config_path in enumerate(config_paths):
        print(f"==================runnig '{config_path}' {i+1}/{len(config_paths)}===================")
        if os.path.exists(image_paths[i]):
            print(f"skipping '{config_path}' because '{image_paths[i]}' already exists")
            continue
        config  = use_file_config(config_path)
        main(config)

def run_varying(model = None):
    config_paths = walk_config("config/varying")
    if model is not None:
        config_paths = [config_path for config_path in config_paths if config_path.endswith(f"{model}.toml")]
    varying_paths  = [config_path.replace("config/varying","images").replace(".toml","/varying.pdf") for config_path in config_paths]
    for i,config_path in enumerate(config_paths):
        print(f"==================runnig '{config_path}' {i+1}/{len(config_paths)}===================")
        if os.path.exists(varying_paths[i]):
            print(f"skipping '{config_path}' because '{varying_paths[i]}' already exists")
            continue
        config  = use_file_config(config_path)
        main(config)

def run_plot(model = None):
    run_predict(model)
    run_varying(model)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", default=None, type=str, help="model name")
    args = parser.parse_args()
    run_plot(args.model)
    