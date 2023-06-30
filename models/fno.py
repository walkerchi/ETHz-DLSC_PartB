import torch 
import torch.nn as nn 
import torch.nn.functional as F

def parse_int2(x):
    if isinstance(x, (tuple, list)):
        assert len(x) == 2, "x must be a tuple of length 2"
        return int(x[0]), int(x[1])
    else:
        return int(x), int(x)

class SpectralConv2d(nn.Module):
    def __init__(self, 
                 in_channel,
                 out_channel,
                 modes=None):
        super().__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        if modes is None:
            self.modes_x, self.modes_y = None, None
            self.weight_real = nn.Parameter(torch.randn(in_channel, out_channel))
            self.weight_imag = nn.Parameter(torch.randn(in_channel, out_channel))
        else:
            self.modes_x, self.modes_y = parse_int2(modes)
            self.weight_real = nn.Parameter(torch.randn(in_channel, out_channel, self.modes_x, self.modes_y))
            self.weight_imag = nn.Parameter(torch.randn(in_channel, out_channel, self.modes_x, self.modes_y))
        
        self.bias_real = nn.Parameter(torch.randn(out_channel)[None,:,None,None])
        self.bias_imag = nn.Parameter(torch.randn(out_channel)[None,:,None,None])
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.uniform_(self.weight_real, 0, 1 / (self.in_channel * self.out_channel))
        nn.init.uniform_(self.weight_imag, 0, 1 / (self.in_channel * self.out_channel))
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

        if self.modes_x is None:
            spectral = torch.complex(
                torch.einsum("bixy,io->boxy", spectral.real, self.weight_real) + self.bias_real,
                torch.einsum("bixy,io->boxy", spectral.imag, self.weight_imag) + self.bias_imag
            )
            output   = torch.fft.irfft2(spectral, s=x.shape[-2:]) # output [batch_size, output_channel, window_size, window_size]
        else:
            assert self.modes_y <= (x.shape[-1] // 2 + 1)/2, f"modes_y must be less than or equal to (window_size // 2 + 1)/2({(x.shape[-1]//2 + 1)/2}), got {self.modes_y}"
            spectral = torch.zeros(spectral.shape, dtype=spectral.dtype, device=spectral.device)
            B, Ci, H, W = spectral.shape
            Co = self.out_channel
            output_spectral = torch.zeros([B, Co, H, W], dtype=spectral.dtype, device=spectral.device)
            output_spectral[:, :, :self.modes_x, :self.modes_y] = torch.complex(
                torch.einsum("bixy,ioxy->boxy", spectral[:, :, :self.modes_x, :self.modes_y].real, self.weight_real) + self.bias_real,
                torch.einsum("bixy,ioxy->boxy", spectral[:, :, :self.modes_x, :self.modes_y].imag, self.weight_imag) + self.bias_imag
            )
            output_spectral[:, :, -self.modes_x:, -self.modes_y:] = torch.complex(
                torch.einsum("bixy,ioxy->boxy", spectral[:, :, -self.modes_x:, -self.modes_y:].real, self.weight_real) + self.bias_real,
                torch.einsum("bixy,ioxy->boxy", spectral[:, :, -self.modes_x:, -self.modes_y:].imag, self.weight_imag) + self.bias_imag
            )
            output   = torch.fft.irfft2(output_spectral, s=x.shape[-2:]) # output [batch_size, output_channel, window_size, window_size]
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
                 in_channel,
                 out_channel,
                 hidden_channel,
                 num_layers,
                 activation='gelu',
                 modes=None,
                 io_transform=False):
        super().__init__()
        self.spectral_convs  = nn.ModuleList()
        self.spatial_convs   = nn.ModuleList()

        if io_transform:
            self.input_transform = nn.Sequential(
                Permute(0, 2, 3, 1),
                nn.Linear(in_channel, hidden_channel),
                Activation(activation),
                Permute(0, 3, 1, 2)
            )

            for _ in range(num_layers):
                self.spectral_convs.append(SpectralConv2d(hidden_channel, hidden_channel, modes=modes))
                self.spatial_convs.append(UnitConv2d(hidden_channel, hidden_channel))

            self.output_transform = nn.Sequential(
                Permute(0, 2, 3, 1),
                nn.Linear(hidden_channel, out_channel),
                Activation(activation),
                Permute(0, 3, 1, 2)
            )
        else:
            self.input_transform = None 
            self.output_transform = None
            
            self.spectral_convs.append(SpectralConv2d(in_channel, hidden_channel, modes=modes))
            self.spatial_convs.append(UnitConv2d(in_channel, hidden_channel))
            
            for _ in range(num_layers - 2):
                self.spectral_convs.append(SpectralConv2d(hidden_channel, hidden_channel, modes=modes))
                self.spatial_convs.append(UnitConv2d(hidden_channel, hidden_channel))
            
            self.spectral_convs.append(SpectralConv2d(hidden_channel, out_channel, modes=modes))
            self.spatial_convs.append(UnitConv2d(hidden_channel, out_channel))

        if activation in ['tanh', 'sigmoid']:
            self.activation_fn = getattr(torch, activation)
        else:
            self.activation_fn = getattr(F, activation)

        self.reset_parameters()

    def reset_parameters(self):
        for spectral_conv, spatial_conv in zip(self.spectral_convs, self.spatial_convs):
            spectral_conv.reset_parameters()
            spatial_conv.reset_parameters()
        
        if self.input_transform is not None:
            for module in self.input_transform:
                if isinstance(module, nn.Linear):
                    module.reset_parameters()
        
        if self.output_transform is not None:
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
        if self.input_transform is not None:
            x = self.input_transform(x)
        
        for spectral_conv, spatial_conv in zip(self.spectral_convs, self.spatial_convs):
            x = self.activation_fn(spatial_conv(x) + spectral_conv(x))
        
        if self.output_transform is not None:
            x = self.output_transform(x)
        
        return x