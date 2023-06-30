
import os
import re

from config import use_file_config
from run_folder import walk_config
from run_cmd import main

def run_train():
    config_paths = walk_config("config/train")
   
    weights_paths = [config_path.replace("config/train", "weights").replace(".toml","/model.pth") for config_path in config_paths]
    for i,config_path in enumerate(config_paths):
        print(f"==================runnig '{config_path}' {i+1}/{len(config_paths)}===================")
        if os.path.exists(weights_paths[i]):
            print(f"skipping '{config_path}' because '{weights_paths[i]}' already exists")
            continue
        config  = use_file_config(config_path)
        main(config)

if __name__ == '__main__':
    run_train()
    