import argparse

from config import use_file_config
from main import main


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-c","--config", type=str, help="path to config file(.toml, .json, .yaml)")
    args = parser.parse_args()
    config  = use_file_config(args.config)

    main(config)
    