from itertools import product
from trainer import FFNTrainer, DeepONetTrainer, MeshNeuralOperatorTrainer
from config import use_cmd_config, EQUATION_VALUES, EQUATION_KEYS, MODELS


def build_trainer(config):
    if config.model == "ffn":
        trainer = FFNTrainer(config)
    elif config.model == "deeponet":
        trainer = DeepONetTrainer(config)
    else:
        trainer = MeshNeuralOperatorTrainer(config)
    return trainer

def main(config):

    if config.task == "train":
        trainer = build_trainer(config)
        trainer.fit()
        trainer.save()
        trainer.plot_prediction(config.n_eval_spatial)
    elif config.task == "predict":
        trainer = build_trainer(config)
        trainer.load()
        trainer.predict(config.n_eval_spatial)
    elif config.task == "plot_varying":
        eval_results = []
        k = EQUATION_KEYS[config.equation]
        for v in EQUATION_VALUES:
            config[k] = v
            trainer = build_trainer(config)
            trainer.load()
            eval_result = trainer.eval()
            eval_results.append(eval_result)
        trainer.plot_varying(eval_results, **{k:EQUATION_VALUES})
    else:
        raise NotImplementedError()

if __name__ == '__main__':
    config  = use_cmd_config()
    main(config)
   