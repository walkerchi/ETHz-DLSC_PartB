
import os
import re
import argparse
from config import use_file_config
from run_folder import walk_config
from run_cmd import main

def run_train(model=None,  overwrite = False):
    config_paths = walk_config("config/train")

    if model is not None:
        config_paths = [config_path for config_path in config_paths if config_path.endswith(f"{model}.toml")]
        
    weights_paths = [config_path.replace("config/train", "weights").replace(".toml","/model.pth") for config_path in config_paths]
    for i,config_path in enumerate(config_paths):
        print(f"\n==================runnig '{config_path}' {i+1}/{len(config_paths)}===================\n")
        if not overwrite and os.path.exists(weights_paths[i]):
            print(f"skipping '{config_path}' because '{weights_paths[i]}' already exists")
            continue
        config  = use_file_config(config_path)
        main(config)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", default=None, type=str, help="model name")
    parser.add_argument("-f", "--force", action="store_true", help="force overwrite")
    args = parser.parse_args()
    run_train(args.model, overwrite=args.force)
    