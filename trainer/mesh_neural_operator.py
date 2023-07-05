import os
import numpy as np
import torch 
from torch.utils.data import DataLoader, TensorDataset
from models import FNO2d,CNO2d,UNet2d,KNO2d, ModelLookUp
from .base import   SpatialSampler,\
                    DatasetGeneratorBase,\
                    NormalizerBase,\
                    DataLoaderBase,\
                    TrainerBase,\
                    general_call,\
                    scatter_error2d,\
                    to_device,\
                    set_seed,\
                    EquationLookUp
from config import EQUATION_KEY


class MeshNeuralOperatorDatasetGenerator(DatasetGeneratorBase):
    def __init__(self, T, Equation,seed=1234, **kwargs):
        self.kwargs = kwargs
        self.T = T
        self.Equation = Equation
        self.seed = seed 
        set_seed(seed)

    def __call__(self, n_sample, n_points, **kwargs):
        """
            can only use mesh sampler
            inputs  [n_samples, 7, H, W] (x, y, u0)
            outputs [n_samples, 1, H, W] u(T, x, y, mu)
        """
        inputs = []
        outputs= []
        points_sampler = SpatialSampler(self.Equation, sampler="mesh")
        points      = points_sampler(n_points).permute(2, 0, 1)  # [2, H, W]
        x_dim = self.Equation.x_domain.shape[0]

        for _ in range(n_sample):

            equation    = self.Equation(**self.kwargs)
            u0          = equation(0,      *[points[i] for i in range(x_dim)])[None, ...] # [1, H, W]
            uT          = equation(self.T, *[points[i] for i in range(x_dim)])[None, ...] # [1, H, W]  
            input       = torch.cat([points,
                                      torch.cos(2 * np.pi * points),
                                      torch.sin(2 * np.pi * points),
                                      u0], dim=0) #[3, H, W]
            output      = uT
            inputs.append(input)
            outputs.append(output)

        inputs  = torch.stack(inputs, dim=0)  # [n_samples, 7, H, W]
        outputs = torch.stack(outputs, dim=0) # [n_samples, 1, H, W]
       
        return inputs, outputs  
    
class MeshNeuralOperatorNormalizer:
    def __init__(self, inputs_min, inputs_max, outputs_min, outputs_max):
        self.inputs_min = inputs_min
        self.inputs_max = inputs_max
        self.outputs_min = outputs_min
        self.outputs_max = outputs_max
    
    @classmethod
    def init(cls, inputs, outputs):
        """
            inputs  [n_samples, 3, H, W] (x, y, u0)
            outputs [n_samples, 1, H, W] u(T, x, y, mu)
        """
        inputs     = inputs.permute(1,0,2,3)
        inputs_min = inputs.reshape(inputs.shape[0], -1).min(1).values[None, :, None, None]
        inputs_max = inputs.reshape(inputs.shape[0], -1).max(1).values[None, :, None, None]
        outputs_min = outputs.min()
        outputs_max = outputs.max()
        return cls(inputs_min, inputs_max, outputs_min, outputs_max)
    
    def __call__(self, inputs, outputs):
        inputs_min, inputs_max, outputs_min, outputs_max = to_device([self.inputs_min, self.inputs_max, self.outputs_min, self.outputs_max],inputs.device)
        # normalize input
        inputs = (inputs-inputs_min)/(inputs_max-inputs_min)
        # normalize output
        outputs = (outputs-outputs_min)/(outputs_max-outputs_min)
        return inputs, outputs
    
    def norm_input(self, input):
        inputs_min, inputs_max = to_device([self.inputs_min, self.inputs_max],input.device)
        return (input - inputs_min) / (inputs_max-inputs_min)
    
    def unorm_output(self, output):
        outputs_min, outputs_max = to_device([self.outputs_min, self.outputs_max],output.device)
        return output*(outputs_max-outputs_min)+outputs_min
    
    def save(self, path):
        torch.save({'inputs_min':self.inputs_min,'inputs_max':self.inputs_max,'outputs_min':self.outputs_min,'outputs_max':self.outputs_max}, path)

    @classmethod
    def load(cls, path):
        data = torch.load(path)
        inputs_min = data['inputs_min']
        inputs_max = data['inputs_max']
        outputs_min = data['outputs_min']
        outputs_max = data['outputs_max']
        return cls(inputs_min, inputs_max, outputs_min, outputs_max)

class MeshNeuralOperatorDataLoader(DataLoaderBase):
    pass
    

class MeshNeuralOperatorTrainer(TrainerBase):
    """
        G(u0,x1,x2)
    """
    DataLoader = MeshNeuralOperatorDataLoader
    Normalizer = MeshNeuralOperatorNormalizer
    def __init__(self, config):
        self.config = config
        Equation = EquationLookUp[config.equation]
        equation_kwargs = {EQUATION_KEY[config.equation]:config[EQUATION_KEY[config.equation]]}
        # equation_kwargs = {k:config[k] for k in EquationKwargsLookUp[config.equation]}
        self.xlims = Equation.x_domain

        Model                  = ModelLookUp[config.model]
        x_dim                  = Equation.x_domain.shape[0]

        self.dataset_generator = MeshNeuralOperatorDatasetGenerator(config.T, Equation, seed=self.config.seed, **equation_kwargs)

        if Model == FNO2d:
            model_kwargs       = {"modes":config.modes}
        elif Model == CNO2d:
            model_kwargs       = {"jit":config.jit}
        else:
            model_kwargs       = {}
        self.model             = Model(in_channel=x_dim*3 +1, out_channel=1, hidden_channel=config.num_hidden, num_layers=config.num_layers, activation=config.activation,
                                       **model_kwargs)
   
        self.weight_path       = f"weights/{config.equation}_{'_'.join([f'{k}={v}' for k,v in equation_kwargs.items()])}/{config.model}"
        self.image_path        = f"images/{config.equation}_{'_'.join([f'{k}={v}' for k,v in equation_kwargs.items()])}/{config.model}"
      
    def to(self, device):
        self.model = self.model.to(device)
        return self

    def eval(self):
        """
            Returns:
            --------
                position:  torch.Tensor, shape=(n_eval_sample, n_eval_spatial, 2)
                predictions: torch.Tensor, shape=(n_eval_sample, n_eval_spatial)
                outputs: torch.Tensor, shape=(n_eval_sample, n_eval_spatial)
        """
        self.to(self.config.device)
        config            = self.config
        x_dim             = self.xlims.shape[0]
        dataset           = self.dataset_generator(config.n_eval_sample, config.n_eval_spatial) # input (x,y,u0), output (u)
        points            = dataset[0] # [n_eval_sample, 3, H, W]
        points            = points[:, :x_dim] # [n_eval_sample, 2, H, W]
        points            = points.permute(0, 2, 3, 1) # [n_eval_sample, H, W, 2]
        points            = points.reshape(config.n_eval_sample, -1, 2) # [n_eval_sample, n_eval_spatial, 2]
        dataset           = self.normalizer(*dataset)
        dataloader        = self.DataLoader(*dataset, batch_size=config.batch_size, device=config.device, shuffle=True)

        predictions = []
        outputs     = []
    
        with torch.no_grad():
            for input_batch, output_batch in dataloader:       
                prediction   = general_call(self.model, input_batch) #[batch_size, 1, H, W] 
                prediction   = self.normalizer.unorm_output(prediction)
                output_batch = self.normalizer.unorm_output(output_batch)
                prediction   = prediction.reshape([-1, config.n_eval_spatial]) # [batch_size, n_eval_spatial]
                output_batch = output_batch.reshape([-1, config.n_eval_spatial])
                predictions.append(prediction.cpu())
                outputs.append(output_batch.cpu())

        predictions = torch.cat(predictions, dim=0) # [n_eval_sample, n_eval_spatial]
        outputs     = torch.cat(outputs, dim=0)     # [n_eval_sample, n_eval_spatial]

        return points, predictions, outputs


    def predict(self, n_eval_spatial):
        """
            Parameters:
            -----------
                n_eval_spatial: int, number of spatial points to evaluate

            Returns:
            --------
                position:  torch.Tensor, shape=(n_eval_spatial, 2)
                u0:      torch.Tensor, shape=(n_eval_spatial)
                predictions: torch.Tensor, shape=(n_eval_spatial)
                uT:     torch.Tensor, shape=(n_eval_spatial)
        """
        self.to(self.config.device)
        set_seed(self.config.seed)
        input, output = self.dataset_generator(1, n_eval_spatial) # input (1, 3, H, W), output (1, 1, H, W)
        points = input[0, :2] # [2, H, W]
        points = points.reshape(2,-1)
        u0     = input[0, -1] # [H, W]
        input = self.normalizer.norm_input(input)
        input = to_device(input, self.config.device)

        with torch.no_grad():
            prediction = self.model(input)
        prediction = self.normalizer.unorm_output(prediction) # [1, 1, H, W]
        prediction = prediction.cpu().flatten() # [H*W]
        output     = output.cpu().flatten()     # [H*W]

        return points.T, u0.flatten().cpu(), prediction, output



    def plot_prediction(self, n_eval_spatial):
        points, u0, prediction, output = self.predict(n_eval_spatial)

        scatter_error2d(points[:,0], points[:,1], prediction, output, self.image_path, xlims=self.xlims,
                        u0 = u0)
