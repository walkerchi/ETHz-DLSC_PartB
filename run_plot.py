
import os
import re

from config import use_file_config
from main import main
from run_folder import walk_config


def run_predict():
    config_paths = walk_config("config/predict")
    image_paths  = [config_path.replace("config/predict").replace(".toml","/error.pdf") for config_path in config_paths]
    for i,config_path in enumerate(config_paths):
        print(f"==================runnig '{config_path}' {i+1}/{len(config_paths)}===================")
        if os.path.exists(image_paths[i]):
            print(f"skipping '{config_path}' because '{image_paths[i]}' already exists")
            continue
        config  = use_file_config(config_path)
        main(config)

def run_varying():
    config_paths = walk_config("config/varying")
    varying_paths  = [config_path.replace("config/varying").replace(".toml","/varying.pdf") for config_path in config_paths]
    for i,config_path in enumerate(config_paths):
        print(f"==================runnig '{config_path}' {i+1}/{len(config_paths)}===================")
        if os.path.exists(varying_paths[i]):
            print(f"skipping '{config_path}' because '{varying_paths[i]}' already exists")
            continue
        config  = use_file_config(config_path)
        main(config)

if __name__ == '__main__':
    run_predict()
    run_varying()
    