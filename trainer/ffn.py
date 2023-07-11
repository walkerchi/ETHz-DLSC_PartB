
import torch 

from models import FFN
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
                    
class FFNDatasetGenerator(DatasetGeneratorBase):
    def __init__(self, T, Equation, seed=1234, **kwargs):
        self.T        = T
        self.kwargs   = kwargs
        self.Equation = Equation 
        self.seed     = seed 
        set_seed(seed)

    def __call__(self, n_sample, n_points, sampler="mesh"):
        """
            inputs [n_samples * n_points, 2+d] (x, y, mu)
            outputs [n_samples * n_points, 1] u(T, x, y, mu)
        """
        inputs = []
        outputs= []
        u0s    = []
        points_sampler = SpatialSampler(self.Equation, sampler=sampler)
        
        x_dim = self.Equation.x_domain.shape[0]
        
        for _ in range(n_sample):
            points   = points_sampler(n_points, flatten=True)
            equation = self.Equation(**self.kwargs)
            input    = torch.cat([points, equation.variable.flatten()[None,:].tile(points.shape[0],1)], -1) #[n_points, 2 + d]
            output   = equation(self.T, *[points[:,i] for i in range(x_dim)])[:, None] # [n_points, 1]
            u0       = equation(0, *[points[:,i] for i in range(x_dim)]) # [n_points]
            u0s.append(u0)
            inputs.append(input)
            outputs.append(output)
     
        self.u0s = torch.stack(u0s, dim=0) # [n_samples * n_points]
        inputs  = torch.cat(inputs, dim=0)  # [n_samples * n_points, 2+d]
        outputs = torch.cat(outputs, dim=0) # [n_samples * n_points, 1]
       
        return inputs, outputs
    
class FFNNormalizer(NormalizerBase):
    def __init__(self, inputs_min, inputs_max, outputs_min, outputs_max):
        self.inputs_min = inputs_min
        self.inputs_max = inputs_max
        self.outputs_min = outputs_min
        self.outputs_max = outputs_max

    @classmethod
    def init(cls, inputs, outputs):
        inputs_min = inputs.min(0).values[None, ...]
        inputs_max = inputs.max(0).values[None,...]
        outputs_min = outputs.min(0).values[None,...]
        outputs_max = outputs.max(0).values[None,...]
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
    
class FFNDataLoader(DataLoaderBase):
    pass

class FFNTrainer(TrainerBase):
    """
        g(x,y,\mu ) = u(T, x, y, \mu )
    """
    DataLoader = FFNDataLoader
    Normalizer = FFNNormalizer
    def __init__(self, config):
        self.config = config
        Equation = EquationLookUp[config.equation]
        equation_kwargs = {EQUATION_KEY[config.equation]:config[EQUATION_KEY[config.equation]]}
        # equation_kwargs = {k:config[k] for k in EquationKwargsLookUp[config.equation]}
        self.xlims = Equation.x_domain

        self.dataset_generator = FFNDatasetGenerator(config.T, Equation, seed=self.config.seed, **equation_kwargs)
        self.model             = FFN(input_size=2+Equation.degree_of_freedom(**equation_kwargs), output_size=1, hidden_size=config.num_hidden, num_layers=config.num_layers, activation=config.activation)
   
        self.weight_path       = f"weights/{config.equation}_{'_'.join([f'{k}={v}' for k,v in equation_kwargs.items()])}/ffn"
        self.image_path        = f"images/{config.equation}_{'_'.join([f'{k}={v}' for k,v in equation_kwargs.items()])}/ffn"
       
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
        dataset           = self.dataset_generator(config.n_eval_sample, config.n_eval_spatial, sampler=config.sampler) # input (x, y, ...mu), output
        points            = dataset[0][:, :x_dim].reshape(config.n_eval_sample, config.n_eval_spatial, x_dim) # [n_eval_sample, n_eval_spatial, 2]
        dataset           = self.normalizer(*dataset)
        dataloader        = self.DataLoader(*dataset, batch_size=config.batch_size, device=config.device, shuffle=True)

        predictions = []
        outputs     = []

        with torch.no_grad():
            for input_batch, output_batch in dataloader:  
                prediction = general_call(self.model, input_batch) #[batch_size*n_eval_spatial, 1] 
                prediction = self.normalizer.unorm_output(prediction).reshape([-1, config.n_eval_spatial]) # [batch_size, n_eval_spatial]
                output_batch = self.normalizer.unorm_output(output_batch).reshape([-1, config.n_eval_spatial])
                predictions.append(prediction.cpu())
                outputs.append(output_batch.cpu())

        predictions = torch.cat(predictions, dim=0) # [n_eval_sample, n_eval_spatial]
        outputs     = torch.cat(outputs, dim=0) # [n_eval_sample, n_eval_spatial]

        return points, predictions, outputs

    def predict(self, n_eval_spatial):
        """
            Parameters:
            -----------
                n_eval_spatial: int, number of spatial points to evaluate

            Returns:
            --------
                points: torch.Tensor, shape=(n_eval_spatial, 2)
                u0: torch.Tensor, shape=(n_eval_spatial)
                prediction: torch.Tensor, shape=(n_eval_spatial)
                uT: torch.Tensor, shape=(n_eval_spatial)
        """
        self.to(self.config.device)
        set_seed(self.config.seed)
        input, output = self.dataset_generator(1, n_eval_spatial, sampler="mesh")
        points = input[:, :2]
        input = self.normalizer.norm_input(input)
        input = to_device(input, self.config.device)
        with torch.no_grad():
            prediction = self.model(input)
        prediction = self.normalizer.unorm_output(prediction).cpu()

        return points.cpu(), self.dataset_generator.u0s.flatten(), prediction.flatten(), output.flatten().cpu()

    def plot_prediction(self, n_eval_spatial):
        points, u0, prediction, uT = self.predict(n_eval_spatial)

        scatter_error2d(points[:,0], points[:,1], prediction, uT, self.image_path, self.xlims, u0=u0)
