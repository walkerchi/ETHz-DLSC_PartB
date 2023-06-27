import torch 
import torch.nn as nn 

from .ffn import MLP
from .cno import ResBlock2d
from .fno import FNO2d


class DeepONet(nn.Module):
    """
    DeepONet
        Branch: u0
            if branch_arch is mlp
                self.branch [n_samples, n_basis] -> [n_samples, hidden_size] 
            else trunk_arch is resnet or fno
                self.branch [n_sample, 1, H, W] -> [n_sample, hidden_size]
        Trunk: x
            if trunk_arch is mlp
                self.trunk  [n_points, 2] -> [n_points, hidden_size]
            else
                self.trunk  [2, H, W] -> [hidden_size, H, W]
       
        basis = self.branch(branch)
        query = self.trunk(trunk)

        if trunk_arch is mlp
            output  [n_sample, n_points] uT
        else:
            output  [n_sample, H, W] uT
        Parameters:
        -----------
            branch_size: int, n_points
            trunk_size: int, n_basis
            hidden_size: int, hidden_size
            num_layers: int, num_layers
            branch_arch: str, branch_arch one of ["mlp", "resnet", "fno"], default "mlp"
            trunk_arch: str, trunk_arch one of ["mlp", "resnet", "fno"], default  "mlp"
    """
    def __init__(self,
                 branch_size,
                 trunk_size,
                 hidden_size,
                 num_layers,
                 activation="relu",
                 branch_arch="mlp",
                 trunk_arch="mlp"):
        super().__init__()
        assert branch_arch in ["mlp", "resnet", "fno"]
        assert trunk_arch in ["mlp", "resnet", "fno"]
        self.branch_size = branch_size
        self.trunk_size = trunk_size
        self.hidden_size = hidden_size
        self.branch_arch = branch_arch
        self.trunk_arch = trunk_arch

        # basis = branch, weight = trunk
        MeshNeuralOperator = {
            "resnet": ResBlock2d,
            "fno": FNO2d
        }
        
        if branch_arch == "mlp":
            self.branch  = MLP(branch_size, hidden_size, hidden_size, num_layers, activation=activation)
        else:
            BranchNet = MeshNeuralOperator[branch_arch]
            self.branch  = nn.Sequential(
                BranchNet(branch_size, hidden_size, hidden_size, num_layers, activation=activation),
                nn.Flatten(),
                nn.LazyLinear(hidden_size)
            )
        
        if trunk_arch == "mlp":
            self.trunk = MLP(trunk_size, hidden_size, hidden_size, num_layers, activation=activation)
        else:
            TrunkNet = MeshNeuralOperator[trunk_arch]
            self.trunk = TrunkNet(trunk_size, hidden_size, hidden_size, num_layers, activation=activation)
    
        self.bias = nn.Parameter(torch.zeros(1), requires_grad=True)

        self.reset_parameters()

    def reset_parameters(self):
        self.branch.reset_parameters()
        self.trunk.reset_parameters()
        nn.init.zeros_(self.bias)

    def forward(self, branch, trunk):
        """
            Parameters:
            -----------
                basis/branch: torch.Tensor, shape=(n_sample, n_basis) (u0) for mlp basis arch 
                                        or  shape=(n_sample, H, W) for resnet/fno basis arch
                points/trunck: torch.Tensor, shape=(n_basis, 2) (x1,x2)
                                        or shape=(2, H, W) (x1,x2) for resnet/fno trunk arch
            Returns:
            --------
                if trunk_arch is mlp:
                    output: torch.Tensor, shape=(n_samples, n_points) u(T, x1, x2)
                else:
                    output: torch.Tensor, shape=(n_samples, H, W) u(T, x1, x2)
        """
        if self.branch_arch == "mlp":
            basis = self.branch(branch)                 # [n_samples, n_basis] -> [n_samples, hidden_size]
        else:
            basis = self.branch(branch[:, None, :, :])  # [n_samples, H, W] -> [n_samples, hidden_size]
            
        if self.trunk_arch == "mlp":
            query = self.trunk(trunk)               # [n_points, 2] -> [n_points, hidden_size] 
            output = basis @ query.T                # [n_samples, n_points]
        else:
            query = self.trunk(trunk[None,...])[0]  # [2, H, W] -> [hidden_size, H, W]
            output = torch.einsum("ni,ihw->nhw", basis, query) # [n_samples, H, W]

        output += self.bias
        
        return output