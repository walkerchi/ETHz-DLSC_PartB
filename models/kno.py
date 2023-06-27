import torch 
import torch.nn as nn
import torch.nn.functional as F 

from .fno import SpectralConv2d

class Koopman2d(SpectralConv2d):
    def __init__(self, hidden_channel, modes=None):
        super().__init__(hidden_channel, hidden_channel, modes)

class KNO2d(nn.Module):
    def __init__(self, in_channel, out_channel, hidden_channel, num_layers=6, modes=4, **kwargs):
        super().__init__()

        self.order = num_layers
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channel, hidden_channel, kernel_size=1),
            nn.Tanh()
        )
        self.koopman = Koopman2d(hidden_channel, modes)
        self.decoder = nn.Sequential(
            nn.Conv2d(hidden_channel, hidden_channel, kernel_size=1),
            nn.Tanh(),
            nn.Conv2d(hidden_channel, out_channel, kernel_size=1)
        )
        self.reset_parameters()

    def reset_parameters(self):
        for layer in self.encoder:
            if isinstance(layer, nn.Conv2d):
                layer.reset_parameters()
        for layer in self.decoder:
            if isinstance(layer, nn.Conv2d):
                layer.reset_parameters()
        self.koopman.reset_parameters()

    def forward(self, x):
        """
            Parameters:
            -----------
                x: torch.Tensor, shape=(batch_size,  channel, H, W)
            Returns:
            --------
                y: torch.Tensor, shape=(batch_size,  channel, H, W)
        """
        x = self.encoder(x)

        skip = x 

        for _ in range(self.order):
            x = x + self.koopman(x)
          
        x = self.decoder(x + skip)

        return x