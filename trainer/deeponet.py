import torch 
import os
from models import DeepONet
from .base import   SpatialSampler,\
                    DatasetGeneratorBase,\
                    NormalizerBase,\
                    DataLoaderBase,\
                    TrainerBase,\
                    general_call,\
                    scatter_error2d,\
                    to_device,\
                    EquationLookUp,\
                    EquationKwargsLookUp


class DeepONetDatasetGenerator(DatasetGeneratorBase):
    """
        if branch_sampler is "mesh" and branch_arch is ["fno", "resnet"]
            branch shape=(n_sample, H, W)
        else
            branch shape=(n_sample, n_basis)

        if trunk_sampler is mesh and trunk_arch is ["fno", "resnet"]
            trunk shape=(2, H, W)
            output shape=(n_sample, H, W)
        else
            trunk shape=(n_points, 2)
            output shape=(n_sample, n_points)

    """
    def __init__(self, T, n_basis, Equation, 
                 branch_arch="mlp", 
                 trunk_arch="mlp", 
                 branch_sampler="mesh",
                 trunk_sampler="uniform",
                 **kwargs):
        
        self.kwargs   = kwargs
        self.T        = T
        self.Equation =  Equation
        self.branch_arch = branch_arch
        self.trunk_arch  = trunk_arch

        # when branch architecure is resnet or fno, branch_sampler must be mesh
        if branch_arch in ["resnet", "fno"]:
            assert branch_sampler == "mesh", f"branch_sampler must be mesh when branch_arch is {branch_arch}"

        # sample basis
        basis_sampler  = SpatialSampler(self.Equation(**self.kwargs), sampler=branch_sampler )
        basis_points   = basis_sampler(n_basis,flatten = (branch_arch=="mlp"))
        self.basis_points = basis_points # [n_basis, 2] or [H, W, 2]

        self.trunk_sampler = trunk_sampler

    def load(self, path):
        self.basis_points = torch.load(path)
    
    def save(self, path):
        torch.save(self.basis_points, path)
        
    def __call__(self, n_sample, n_points):
        # when trunk architecure is resnet or fno, trunk_sampler must be mesh
        if self.trunk_arch in ["resnet", "fno"]:
            assert self.trunk_sampler == "mesh", f"trunk_sampler must be mesh when trunk_arch is {self.trunk_arch}"
        
        trunk_sampler = SpatialSampler(self.Equation, sampler=self.trunk_sampler)
        x_dim         = self.Equation.x_domain.shape[0]
        trunk_points  = trunk_sampler(n_points, flatten=(self.trunk_arch=="mlp")) # [n_points, 2] or [H, W, 2]
        branch = []
        output  = []
        for _ in range(n_sample):
            equation= self.Equation(**self.kwargs)
            basis_u0   = equation(0,      *[self.basis_points[...,i] for i in range(x_dim)])      # u0 [n_basis] or [H, W]
            query_uT   = equation(self.T, *[trunk_points[...,i] for i in range(x_dim)])           # uT [n_points] or [H, W]
            branch.append(basis_u0)
            output.append(query_uT)
      
        branch = torch.stack(branch, 0)   #[n_sample, n_basis] or [n_sample, H, W]
        trunk  = trunk_points if trunk_points.dim() == 2 else trunk_points.permute(2, 0, 1)      #[n_points, 2]  or [2, H, W]      
        output  = torch.stack(output, 0)  #[n_sample, n_points] or [n_sample, H, W]
        
        return branch,trunk,output
    
class DeepONetNormalizer(NormalizerBase):
    def __init__(self, branch_min, branch_max, trunk_min, trunk_max, output_min, output_max):
        self.branch_min = branch_min
        self.branch_max = branch_max
        self.trunk_min  = trunk_min
        self.trunk_max  = trunk_max
        self.output_min = output_min
        self.output_max = output_max

    @classmethod
    def init(cls, branch, trunk, output):
        branch_min = branch.min()
        branch_max = branch.max()
        trunk_min  = trunk.min()
        trunk_max  = trunk.max()
        output_min = output.min()
        output_max = output.max()
        return cls(branch_min, branch_max, trunk_min, trunk_max, output_min, output_max)
    
    def __call__(self, branch, trunk, output):
        branch_min, branch_max, trunk_min, trunk_max, output_min, output_max = to_device([self.branch_min, self.branch_max, self.trunk_min, self.trunk_max, self.output_min, self.output_max], branch.device)
        # normalize input
        branch = (branch-branch_min)/(branch_max-branch_min)
        trunk  = (trunk-trunk_min)/(trunk_max-trunk_min)
        # normalize output
        output = (output-output_min)/(output_max-output_min)
        return branch, trunk, output
    
    def norm_input(self, branch, trunk):
        branch_min, branch_max, trunk_min, trunk_max = to_device([self.branch_min, self.branch_max, self.trunk_min, self.trunk_max], branch.device)
        branch = (branch-branch_min)/(branch_max-branch_min)
        trunk  = (trunk-trunk_min)/(trunk_max-trunk_min)
        return branch, trunk
    
    def unorm_output(self, output):
        output_min, output_max = to_device([self.output_min, self.output_max], output.device)
        return output*(output_max-output_min)+output_min
    
    def save(self, path):
        torch.save({'branch_min':self.branch_min,'branch_max':self.branch_max,'trunk_min':self.trunk_min,'trunk_max':self.trunk_max,'output_min':self.output_min,'output_max':self.output_max}, path)
    @classmethod
    def load(cls, path):
        data = torch.load(path)
        branch_min = data['branch_min']
        branch_max = data['branch_max']
        trunk_min = data['trunk_min']
        trunk_max = data['trunk_max']
        output_min = data['output_min']
        output_max = data['output_max']
        return cls(branch_min, branch_max, trunk_min, trunk_max, output_min, output_max)

class DeepONetDataLoader(DataLoaderBase):
    def __init__(self, branch, trunk, output, **kwargs):
        self.branch = branch 
        self.trunk  = trunk
        self.output = output
        super().__init__(torch.arange(len(self.branch)), collate_fn=self.collate_fn, **kwargs)
    def collate_fn(self, items):
        index = torch.stack(items)
        batch_branch = self.branch[index]
        batch_output = self.output[index]
        return (batch_branch, self.trunk), batch_output

class DeepONetTrainer(TrainerBase):
    """
        G(u0)(x1,x2)
    """
    DataLoader = DeepONetDataLoader
    Normalizer = DeepONetNormalizer
    def __init__(self, config):
        self.config = config
        Equation = EquationLookUp[config.equation]
        equation_kwargs = {k:config[k] for k in EquationKwargsLookUp[config.equation]}
        self.xlims = Equation.x_domain

        self.dataset_generator = DeepONetDatasetGenerator(config.T, config.n_basis_spatial, Equation, **equation_kwargs)
        self.model             = DeepONet(branch_size    = config.n_basis_spatial, 
                                        trunk_size      = Equation.x_domain.shape[0], 
                                        hidden_size     = config.num_hidden, 
                                        num_layers      = config.num_layers,
                                        activation      = config.activation,
                                        branch_arch     = config.branch_arch,
                                        trunk_arch      = config.trunk_arch)
   
        self.weight_path       = f"weights/{config.equation}_{'_'.join([f'{k}={v}' for k,v in equation_kwargs.items()])}/deeponet"
        self.image_path        = f"images/{config.equation}_{'_'.join([f'{k}={v}' for k,v in equation_kwargs.items()])}/deeponet"

    def plot_prediction(self, n_eval_spatial):

        branch, trunk, output = self.dataset_generator(1, n_eval_spatial) # branch, trunk, output
        if trunk.dim() == 3: # [2, H, W]
            points = trunk.reshape([2, -1]).T # [n_eval_spatial, 2]
        else:                # [n_eval_spatial, 2]
            points = trunk
        input = self.normalizer.norm_input(branch, trunk)
        input = to_device(input, self.config.device)
        with torch.no_grad():
            prediction = self.model(*input)
        prediction = self.normalizer.unorm_output(prediction).cpu()
        if prediction.dim() == 3: #[1, H, W]
            prediction = prediction.flatten() #[H*W]
            output     = output.flatten()     #[H*W]

        scatter_error2d(points[:,0], points[:,1], prediction, output, os.path.join(self.image_path, "prediction.png"), xlims=self.xlims)

    def eval(self):
        """
            Returns:
            --------
                position:  torch.Tensor, shape=(n_eval_sample, n_eval_spatial, 2)
                predictions: torch.Tensor, shape=(n_eval_sample, n_eval_spatial)
                outputs:     torch.Tensor, shape=(n_eval_sample, n_eval_spatial)
        """
        config            = self.config
        dataset           = self.dataset_generator(config.n_eval_sample, config.n_eval_spatial) # branch, trunk, output
        points            = dataset[1] # trunk [2, H, W] or [n_eval_spatial, 2]
        xdim              = self.xlims.shape[0]
        if points.dim() == 3: # [2, H, W]
            points = points.reshape([xdim, -1]).T[None,:,:].tile([config.n_eval_sample, 1, 1]) # [n_eval_sample, n_eval_spatial, 2]
        dataset           = self.normalizer(*dataset)
        dataloader        = self.DataLoader(*dataset, batch_size=config.batch_size * config.n_eval_spatial, shuffle=True)

        predictions = []
        outputs     = []

        with torch.no_grad():
            for batch_input, batch_output in dataloader:    
                batch_input, batch_output = to_device(batch_input, batch_output, config.device)       
                prediction = general_call(self.model, batch_input) #[batch_size, n_eval_spatial] or [batch_size, H, W] 
                prediction = self.normalizer.unorm_output(prediction)
                batch_output = self.normalizer.unorm_output(batch_output)
                if prediction.dim() == 3: #[batch_size, H, W]
                    prediction = prediction.reshape([-1, config.n_eval_spatial])
                    batch_output = batch_output.reshape([-1, config.n_eval_spatial])
                predictions.append(prediction.cpu())
                outputs.append(batch_output.cpu())

        predictions = torch.cat(predictions, 0) #[n_eval_sample, n_eval_spatial]
        outputs     = torch.cat(outputs, 0)     #[n_eval_sample, n_eval_spatial]                

        return points, predictions, outputs
