import torch 
import torch.nn as nn 
import torch.nn.functional as F
from tqdm import tqdm

class MLP(nn.Module):
    """
    Multi-Layer Perceptron

    Usage:
        >>> mlp = MLP(2, 1, 32, 3)
        >>> x = torch.randn(10, 2)
        >>> y = mlp(x)
        >>> y.shape
        torch.Size([10, 1])
    """
    def __init__(self,
                 input_size,
                 output_size,
                 hidden_size,
                 num_layers,
                 activation='relu',
                 reset_parameters=True):
        super(MLP, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.activation = activation

        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(self.input_size, self.hidden_size))
        for _ in range(self.num_layers - 2):
            self.layers.append(nn.Linear(self.hidden_size, self.hidden_size))
        self.layers.append(nn.Linear(self.hidden_size, self.output_size))

        if self.activation in ['tanh', 'sigmoid']:
            self.activation_fn = getattr(torch, self.activation)
        else:
            self.activation_fn = getattr(F, self.activation)

        if reset_parameters:
            self.reset_parameters()

    def reset_parameters(self):
        for layer in self.layers:
            layer.reset_parameters()

    def forward(self, x):
        """
            Parameters:
            -----------
                x: torch.Tensor, shape=(batch_size, input_size)
            Returns:
            --------
                output: torch.Tensor, shape=(batch_size, output_size)
        """
        for layer in self.layers[:-1]:
            x = self.activation_fn(layer(x))
        x = self.layers[-1](x)
        return x





class FFN(MLP):
    pass

