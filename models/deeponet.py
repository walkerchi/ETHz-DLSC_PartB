import torch 
import torch.nn as nn 

from .mlp import MLP

class DeepONet(nn.Module):
    def __init__(self,
                 basis_size,
                 point_size,
                 output_size,
                 hidden_size,
                 num_layers):
        super().__init__()
        self.basis_size = basis_size
        self.point_size = point_size
        self.output_size = output_size 
        self.hidden_size = hidden_size
        self.Basis = MLP(basis_size, hidden_size*output_size, hidden_size*output_size, num_layers)
        self.Weight = MLP(point_size, hidden_size*output_size, hidden_size*output_size, num_layers)
        self.reset_parameters()

    def reset_parameters(self):
        self.Basis.reset_parameters()
        self.Weight.reset_parameters()

    def forward(self, basis, points):
        """
            Parameters:
            -----------
                basis/branch: torch.Tensor, shape=(n_basis, 2 + d) (x1,x2,u0)
                points/trunck: torch.Tensor, shape=(n_points, 3) (T,x1,x2)
            Returns:
            --------
                output: torch.Tensor, shape=(n_points, d) u(T, x1, x2)
        """
        basis = self.Basis(basis).reshape(-1, self.output_size, self.hidden_size)
        weight = self.Weight(points).reshape(-1, self.output_size, self.hidden_size)
        output = torch.einsum("ioh,joh->jo", basis, weight)
        return output