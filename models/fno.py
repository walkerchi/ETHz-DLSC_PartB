import torch 
import torch.nn as nn 
import torch.nn.functional as F



class SpectralConv2d(nn.Module):
    def __init__(self, 
                 input_channel,
                 output_channel):
        super().__init__()
        self.weight_real = nn.Parameter(torch.randn(input_channel, output_channel))
        self.weight_imag = nn.Parameter(torch.randn(input_channel, output_channel))
        self.bias_real = nn.Parameter(torch.randn(output_channel))
        self.bias_imag = nn.Parameter(torch.randn(output_channel))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight_real, a=5**0.5)
        nn.init.kaiming_uniform_(self.weight_imag, a=5**0.5)
        nn.init.zeros_(self.bias_real)
        nn.init.zeros_(self.bias_imag)

    def forward(self, x):
        """
            Parameters:
            -----------
                x: torch.Tensor, shape=(batch_size, input_channel, window_size, window_size)
            Returns:
            --------
                y: torch.Tensor, shape=(batch_size, output_channel, window_size, window_size)
        """
        spectral = torch.fft.rfft2(x) # spectral [batch_size, input_channel, window_size, window_size//2+1]
        spectral = torch.complex(
            torch.einsum("bimn,io->bomn", spectral.real, self.weight_real) + self.bias_real[None,:,None,None],
            torch.einsum("bimn,io->bomn", spectral.imag, self.weight_imag) + self.bias_imag[None,:,None,None]
        )
        output   = torch.fft.irfft2(spectral, s=x.shape[-2:]) # output [batch_size, output_channel, window_size, window_size]
        return output
    

class UnitConv2d(nn.Conv2d):
    def __init__(self, 
                input_channel,
                output_channel):
        super().__init__(input_channel, output_channel, kernel_size=1, stride=1, padding=0, bias=True)


class Permute(nn.Module):
    def __init__(self, *dims):
        super().__init__()
        self.dims = dims

    def forward(self, x):
        return x.permute(*self.dims)
    
class Activation(nn.Module):
    def __init__(self, activation):
        super().__init__()
        if activation in ['tanh', 'sigmoid']:
            self.activation_fn = getattr(torch, activation)
        else:
            self.activation_fn = getattr(F, activation)

    def forward(self, x):
        return self.activation_fn(x)


class FNO2d(nn.Module):
    def __init__(self,
                 input_size,
                 output_size,
                 hidden_size,
                 num_layers,
                 activation='relu'):
        super().__init__()
        self.spectral_convs  = nn.ModuleList()
        self.spatial_convs   = nn.ModuleList()

        self.input_transform = nn.Sequential(
            Permute(0, 2, 3, 1),
            nn.Linear(input_size, hidden_size),
            Activation(activation),
            Permute(0, 3, 1, 2)
        )
        for _ in range(num_layers):
            self.spectral_convs.append(SpectralConv2d(hidden_size, hidden_size))
            self.spatial_convs.append(UnitConv2d(hidden_size, hidden_size))
        self.output_transform = nn.Sequential(
            Permute(0, 2, 3, 1),
            nn.Linear(hidden_size, output_size),
            Activation(activation),
            Permute(0, 3, 1, 2)
        )

        if activation in ['tanh', 'sigmoid']:
            self.activation_fn = getattr(torch, activation)
        else:
            self.activation_fn = getattr(F, activation)

        self.reset_parameters()

    def reset_parameters(self):
        for spectral_conv, spatial_conv in zip(self.spectral_convs, self.spatial_convs):
            spectral_conv.reset_parameters()
            spatial_conv.reset_parameters()
        for module in self.input_transform:
            if isinstance(module, nn.Linear):
                module.reset_parameters()
        for module in self.output_transform:
            if isinstance(module, nn.Linear):
                module.reset_parameters()

    def forward(self, x):
        """
            Parameters:
            -----------
                x: torch.Tensor, shape=(batch_size, input_channel, window_size, window_size)
            Returns:
            --------
                y: torch.Tensor, shape=(batch_size, output_channel, window_size, window_size)
        """
        x = self.input_transform(x)
        for spectral_conv, spatial_conv in zip(self.spectral_convs, self.spatial_convs):
            x = self.activation_fn(spatial_conv(x) + spectral_conv(x))
        x = self.output_transform(x)
        return x