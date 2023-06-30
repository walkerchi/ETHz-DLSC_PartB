import argparse

from config import use_file_config
from run_cmd import main


def run_file(filepath):
    config  = use_file_config(filepath)
    main(config)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("filepath", type=str, help="path to config file(.toml, .json, .yaml)")
    args = parser.parse_args()
    
    run_file(args.filepath)
    