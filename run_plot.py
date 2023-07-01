
import os
import re
import argparse
from config import use_file_config
from run_cmd import main
from run_folder import walk_config


def run_predict(model = None, overwrite = False):
    config_paths = walk_config("config/predict")
    if model is not None:
        config_paths = [config_path for config_path in config_paths if config_path.endswith(f"{model}.toml")]
    image_paths  = [config_path.replace("config/predict","images").replace(".toml","/error.pdf") for config_path in config_paths]
    for i,config_path in enumerate(config_paths):
        print(f"\n==================runnig '{config_path}' {i+1}/{len(config_paths)}===================")
        if not overwrite and os.path.exists(image_paths[i]):
            print(f"skipping '{config_path}' because '{image_paths[i]}' already exists")
            continue
        config  = use_file_config(config_path)
        main(config)

def run_varying(model = None, overwrite = False):
    config_paths = walk_config("config/varying")
    if model is not None:
        config_paths = [config_path for config_path in config_paths if config_path.endswith(f"{model}.toml")]
    varying_paths  = [config_path.replace("config/varying","images").replace(".toml","/varying.pdf") for config_path in config_paths]
    
    for i,config_path in enumerate(config_paths):
        print(f"\n==================runnig '{config_path}' {i+1}/{len(config_paths)}===================")
        if not overwrite and os.path.exists(varying_paths[i]):
            print(f"skipping '{config_path}' because '{varying_paths[i]}' already exists")
            continue
        config  = use_file_config(config_path)
        main(config)

def run_plot(model = None, overwrite = False):
    run_predict(model, overwrite)
    run_varying(model, overwrite)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", default=None, type=str, help="model name")
    parser.add_argument("--only_varying", action="store_true", help="only run varying")
    parser.add_argument("--only_predict", action="store_true", help="only run predict")
    parser.add_argument("-f", "--force", action="store_true", help="force overwrite")
    args = parser.parse_args()

    if args.only_varying:
        run_varying(args.model, overwrite=args.force)
    elif args.only_predict:
        run_predict(args.model, overwrite=args.force)
    else:
        run_plot(args.model, overwrite=args.force)
    