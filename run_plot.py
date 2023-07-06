
import os
import re
import argparse
from tqdm import tqdm
from itertools import product
from collections import OrderedDict
from config import use_file_config, EQUATIONS, EQUATION_VALUES, EQUATION_KEY, MODELS
from run_cmd import main, build_trainer
from run_folder import walk_config


def run_plot_predict_together(model = None, overwrite = False,  value=None):
    """
        one predict together corresponding to multiple predict config for different equation values
    """
    for i,equation in enumerate(EQUATIONS):
        print(f"\n==================runnig 'plot predict {equation}' {i+1}/{len(EQUATIONS)}===================")
        if value is None:
            image_path  = f"images/{equation}/predict.pdf"
        else:
            image_path  = f"images/{equation}/predict_{EQUATION_KEY[equation]}={value}.pdf"
        if not overwrite and os.path.exists(image_path):
            print(f"skipping '{equation}' because '{image_path}' already exists")
            continue
        
        values = EQUATION_VALUES if value is None else [value]

        p = tqdm(total = len(values) *  len(MODELS), desc="plot prediction")
        u0s, predictions, uTs = [OrderedDict([(k,[]) for k in values]) for _ in range(3)]
        for val, model in product(values, MODELS):
           
            config_path = f"config/predict/{equation}_{EQUATION_KEY[equation]}={val}/{model}.toml"
            config  = use_file_config(config_path)
            trainer = build_trainer(config)
            trainer.load()
            _, u0, prediction, uT = trainer.predict(config.n_eval_spatial)
            u0s[val].append(u0)
            predictions[val].append(prediction)
            uTs[val].append(uT)
            p.update(1)

        trainer.plot_prediction_together(u0s, predictions, uTs)
      

def run_table_predict_together(model = None, overwrite = False):
    """
        one predict together corresponding to multiple predict config for different equation values
    """
    for i,equation in enumerate(EQUATIONS):
        print(f"\n==================runnig 'table predict {equation}' {i+1}/{len(EQUATIONS)}===================")
        table_path  = f"tables/{equation}/predict.csv"
    
        if not overwrite and os.path.exists(table_path):
            print(f"skipping '{equation}' because '{table_path}' already exists")
            continue
        
        predictions, uTs =  [], []
        for val, model in tqdm(product(EQUATION_VALUES, MODELS), total=len(EQUATION_VALUES) *  len(MODELS), desc="plot prediction"):
            config_path = f"config/predict/{equation}_{EQUATION_KEY[equation]}={val}/{model}.toml"
            config  = use_file_config(config_path)
            trainer = build_trainer(config)
            trainer.load()
            _, _, prediction, uT = trainer.predict(config.n_eval_spatial)
            predictions.append(prediction)
            uTs.append(uT)
        trainer.table_prediction_together(predictions, uTs)
       
def run_plot_varying_together(model = None, overwrite = False):
    for i,equation in enumerate(EQUATIONS):
        print(f"\n==================runnig 'plot varying {equation}' {i+1}/{len(EQUATIONS)}===================")
        image_path  = f"images/{equation}/varying.pdf"

        if not overwrite and os.path.exists(image_path):
            print(f"skipping '{equation}' because '{image_path}' already exists")
            continue
    
        predictions, uTs = [], []
        for val, model in tqdm(product(EQUATION_VALUES, MODELS), total=len(EQUATION_VALUES) *  len(MODELS), desc="plot prediction"):
            config_path = f"config/varying/{equation}_{EQUATION_KEY[equation]}={val}/{model}.toml"
            config  = use_file_config(config_path)
            trainer = build_trainer(config)
            trainer.load()
            _, prediction, uT = trainer.eval()
            predictions.append(prediction)
            uTs.append(uT)
        trainer.plot_varying_together(predictions, uTs)

def run_table_varying_together(model = None, overwrite = False):
    for i,equation in enumerate(EQUATIONS):
        print(f"\n==================runnig 'table varying {equation}' {i+1}/{len(EQUATIONS)}===================")
        table_path  = f"tables/{equation}/varying.csv"

        if not overwrite and os.path.exists(table_path):
            print(f"skipping '{equation}' because '{table_path}' already exists")
            continue
    
        predictions, uTs = [], []
        for val, model in tqdm(product(EQUATION_VALUES, MODELS), total=len(EQUATION_VALUES) *  len(MODELS), desc="plot prediction"):
            config_path = f"config/varying/{equation}_{EQUATION_KEY[equation]}={val}/{model}.toml"
            config  = use_file_config(config_path)
            trainer = build_trainer(config)
            trainer.load()
            _, prediction, uT = trainer.eval()
            predictions.append(prediction)
            uTs.append(uT)
        trainer.table_varying_together(predictions, uTs)

def run_plot(model = None, overwrite = False):
    run_plot_predict_together(model, overwrite)
    run_table_predict_together(model, overwrite)
    run_plot_varying_together(model, overwrite)
    run_table_varying_together(model, overwrite)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", default=None, type=str, help="model name")
    parser.add_argument("--only_varying", action="store_true", help="only run varying")
    parser.add_argument("--only_predict", action="store_true", help="only run predict")
    parser.add_argument("--only_table",  action="store_true", help="only run table")
    parser.add_argument("--only_plot",   action="store_true", help="only run plot")
    parser.add_argument("-v","--value", type=int, default=None, help="value of parameter for the equation")
    parser.add_argument("-f", "--force", action="store_true", help="force overwrite")
    args = parser.parse_args()

    if args.only_varying:
        run_plot_varying_together(args.model, overwrite=args.force)
        run_table_varying_together(args.model, overwrite=args.force)
    elif args.only_predict:
        run_plot_predict_together(args.model, overwrite=args.force, value=args.value)
        run_table_varying_together(args.model, overwrite=args.force)
    elif args.only_table:
        run_table_predict_together(args.model, overwrite=args.force)
        run_table_varying_together(args.model, overwrite=args.force)
    elif args.only_plot:
        run_plot_predict_together(args.model, overwrite=args.force,  vlaue=args.value)
        run_plot_varying_together(args.model, overwrite=args.force)
    else:
        run_plot(args.model, overwrite=args.force)
    