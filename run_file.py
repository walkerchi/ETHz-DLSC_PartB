from trainer import FFNTrainer, DeepONetTrainer, MeshNeuralOperatorTrainer
from config import use_file_config


if __name__ == '__main__':
    config  = use_file_config()
    
    if config.model == "ffn":
        trainer = FFNTrainer(config)
    elif config.model == "deeponet":
        trainer = DeepONetTrainer(config)
    else:
        trainer = MeshNeuralOperatorTrainer(config)

    if config.task == "train":
        trainer.fit()
        trainer.save()
    elif config.task == "predict":
        trainer.load()
        trainer.plot_prediction(config.n_eval_spatial)
    elif config.task == "varying":
        trainer.load()
        trainer.plot_varying()
    else:
        raise NotImplementedError()