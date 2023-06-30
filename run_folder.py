import argparse
import os
import re

from config import use_file_config
from run_cmd import main

def walk_config(folder):
    config_paths = []
    for dirpath, dirnames, filenames in os.walk(folder):
        for filename in filenames:
            if re.match(r".*\.(toml|json|yaml|yml)", filename):
                config_paths.append(os.path.join(dirpath, filename))
    return config_paths


def run_folder(folder):
    config_paths = walk_config(folder)
        
    print(config_paths)
    for i,config_path in enumerate(config_paths):
        print(f"==================runnig '{config_path}' {i+1}/{len(config_paths)}===================")
        config  = use_file_config(config_path)
        main(config)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("folder", type=str, help="path to config folder(.toml, .json, .yaml)")
    args = parser.parse_args()

    run_folder(args.folder)
    