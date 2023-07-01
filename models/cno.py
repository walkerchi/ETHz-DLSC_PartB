import torch 
import torch.nn as nn 
import torch.nn.functional as F
import numpy as np
import scipy.signal
import warnings
from functools import partial
from extension.filtered_lrelu import filtered_lrelu


def grid_pool2d(x, kernel_size=2):
    return x[:,:,::kernel_size,::kernel_size]

class GridPool2d(nn.Module):
    def __init__(self, kernel_size=2):
        super().__init__()
        self.kernel_size = kernel_size
    def forward(self, x):
        """
            Parameters:
            -----------
                x: torch.Tensor, shape=(batch_size, channel, H, W)
            Returns:
            --------
                output: torch.Tensor, shape=(batch_size, channel, H//kernel_size, W//kernel_size)
        """
        return grid_pool2d(x, self.kernel_size)

class Conv2d(nn.Module):
    """
        Conv2d with activation function
    """
    def __init__(self, 
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride = 1,
                 transpose = False,
                 padding="same",
                 activation="relu"):
        super().__init__()
        if transpose:
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=padding)
            if stride == 1:
                self.sampling = lambda x:x 
            elif stride > 1:
                self.sampling = nn.Upsample(scale_factor=stride, mode='nearest')
            else:
                raise ValueError(f"stride must be greater than 1, but got {stride}")
        else:
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=padding)
            if stride == 1:
                self.sampling = lambda x:x
            elif stride > 1:
                self.sampling = nn.AvgPool2d(stride)
            else:
                raise ValueError(f"stride must be greater than 1, but got {stride}")

        if activation is None:
            self.activation_fn = None
        elif activation in ["tanh", "sigmoid"]:
            self.activation_fn = getattr(torch, activation)
        else:
            self.activation_fn = getattr(F, activation)

        self.reset_parameters()
    
    def reset_parameters(self):
        self.conv.reset_parameters()
    
    def forward(self, x):
        """
            Parameters:
            -----------
                x: torch.Tensor, shape=(batch_size, in_channel, H, W)
            Returns:
            --------
                output: torch.Tensor, shape=(batch_size, out_channel, H, W)
        """
        x = self.conv(x)
        x = self.sampling(x)
        if self.activation_fn is not None:
            x = self.activation_fn(x)
        return x

class UpFIRDn2d(nn.Module):
    """
        Upsampling and downsampling with FIR filter
        Usage:

        >>> upfirdn2d = UpFIRDn2d(3, 32, 16)
        >>> x = torch.randn(1, 3, 16, 16)
        >>> y = upfirdn2d(x)
        >>> y.shape
        torch.Size([1, 3, 16, 16])
    """
    def __init__(self, 
                channel,
                lrelu_upsampling=2,
                filter_size=6,
                gain=np.sqrt(2),
                slope=0.2):
        super().__init__()
        self.filter_size = filter_size
        self.lrelu_upsampling = lrelu_upsampling

        self.bias = nn.Parameter(torch.zeros(1, channel, 1, 1).float())

        # x > 0: gain *x ; x <=0 : gain * slope * x
        self.activation_fn  = lambda x: F.leaky_relu(x, negative_slope=slope) * gain

        self.reset_parameters()

    @property
    def device(self):
        return self.bias.device

    def init_kernel(self, in_size, out_size):
        self.in_size  = in_size
        self.out_size = out_size

        up_factor           = int(np.ceil(max(in_size,out_size) * self.lrelu_upsampling/ in_size))
        dn_factor           = int(np.ceil(max(in_size,out_size) * self.lrelu_upsampling/ out_size))
        self.up_factor      = up_factor
        self.dn_factor      = dn_factor

        self.up_sample      = nn.Upsample(scale_factor=up_factor, mode='nearest')
        self.dn_sample      = GridPool2d(dn_factor)

        uptaps              = self.filter_size * self.up_factor
        dntaps              = self.filter_size * self.dn_factor
      
        up_filter           = torch.tensor(scipy.signal.firwin(numtaps=uptaps, cutoff=in_size/2.0001, width=in_size, fs=in_size))
        dn_filter           = torch.tensor(scipy.signal.firwin(numtaps=dntaps, cutoff=out_size/2.0001, width=out_size, fs=out_size))
        # 1d kernel to 2d kernel by outer product [kernel_size] -> [1, 1, kernel_size, kernel_size]
        up_filter           = torch.outer(up_filter, up_filter)[None, None, :, :].float()
        dn_filter           = torch.outer(dn_filter, dn_filter)[None, None, :, :].float()

        self.register_buffer('up_filter', up_filter, persistent=False)
        self.register_buffer('dn_filter', dn_filter, persistent=False)

        self.up_filter      = self.up_filter.to(self.device)
        self.dn_filter      = self.dn_filter.to(self.device)
        
        self.conv_up        = lambda  x: F.conv2d(x, self.up_filter, padding="same")
        self.conv_dn        = lambda  x: F.conv2d(x, self.dn_filter, padding="same")

    def reset_parameters(self):
        nn.init.zeros_(self.bias)

    def forward(self, x:torch.Tensor):
        """
            Upsampling -> FIR filter -> activation -> FIR filter -> downsampling
            Parameters:
            -----------
                x: torch.Tensor, shape=(batch_size, channel, in_size, in_size)
            Returns:
            --------
                output: torch.Tensor, shape=(batch_size, channel, out_size, out_size)
        """
        B,C,H,W = x.shape
        assert H == W and H == self.in_size, f"input tensor size({H}) must be equal to in_size({self.in_size})"
        

        x = x + self.bias
        
        x = x.reshape(B*C, 1, H, W)
        
        x = self.up_sample(x)

        x = self.conv_up(x)

        x = self.activation_fn(x)

        x = self.conv_dn(x)

        x = self.dn_sample(x)

        _, _, H, W = x.shape
        assert H == W and H == self.out_size, "output tensor size must be equal to out_size"

        x = x.reshape(B, C, self.out_size, self.out_size)

        return x


class UpFIRDn2dJIT(nn.Module):
    """
        Upsampling and downsampling with FIR filter
        Usage:

        >>> upfirdn2d = UpFIRDn2dJIT(3, 32, 16)
        >>> x = torch.randn(1, 3, 16, 16)
        >>> y = upfirdn2d(x)
        >>> y.shape
        torch.Size([1, 3, 16, 16])
    """
    def __init__(self, 
                channel,
                lrelu_upsampling=2,
                filter_size=6,
                gain=np.sqrt(2),
                slope=0.2,
                kernel_size=3):
        super().__init__()
        self.gain = gain 
        self.slope = slope 
        self.kernel_size = kernel_size
        self.filter_size = filter_size
        self.lrelu_upsampling = lrelu_upsampling

        self.bias = nn.Parameter(torch.zeros(channel).float())

        self.in_size   = None 
        self.out_size  = None 
        self.up_factor = None
        self.dn_factor = None
        self.padding   = None

    @property
    def device(self):
        return self.bias.device

    def init_kernel(self, in_size, out_size):
        self.in_size  = in_size
        self.out_size = out_size

        up_factor           = int(np.ceil(max(in_size,out_size) * self.lrelu_upsampling/ in_size))
        dn_factor           = int(np.ceil(max(in_size,out_size) * self.lrelu_upsampling/ out_size))
        self.up_factor      = up_factor
        self.dn_factor      = dn_factor

        uptaps              = self.filter_size * self.up_factor
        dntaps              = self.filter_size * self.dn_factor

        up_filter           = scipy.signal.firwin(numtaps=uptaps, cutoff=in_size/2.0001, width=in_size, fs=in_size)
        dn_filter           = scipy.signal.firwin(numtaps=dntaps, cutoff=out_size/2.0001, width=out_size, fs=out_size)
        up_filter           = torch.as_tensor(up_filter, dtype=torch.float32, device=self.device)
        dn_filter           = torch.as_tensor(dn_filter, dtype=torch.float32, device=self.device)

        self.register_buffer('up_filter', up_filter, persistent=False)
        self.register_buffer('dn_filter', dn_filter, persistent=False)

        in_size   = np.broadcast_to(np.asarray(in_size), [2])
        out_size  = np.broadcast_to(np.asarray(out_size), [2])
        pad_total = (out_size - 1) * self.dn_factor + 1 # Desired output size before downsampling
        pad_total -= (in_size + self.kernel_size - 1) * self.up_factor # Input size after upsampling.
        pad_total += uptaps + dntaps - 2 # Size reduction caused by the filters.
        pad_lo = (pad_total + self.up_factor) // 2 # Shift sample locations according to the symmetric interpretation (Appendix C.3).
        pad_hi = pad_total - pad_lo
        self.padding = [int(pad_lo[0]), int(pad_hi[0]), int(pad_lo[1]), int(pad_hi[1])]

    def reset_parameters(self):
        nn.init.zeros_(self.bias)

    def forward(self, x):
        if x.device.type == 'cuda':
            x = filtered_lrelu.filtered_lrelu(x=x, fu=self.up_filter, fd=self.dn_filter, b=self.bias.to(x.dtype),
            up=self.up_factor, down=self.dn_factor, padding=self.padding, gain=self.gain, slope=self.slope)
        else:
            warnings.warn("filtered_lrelu_jit is not supported on CPU. Use filtered_lrelu instead.")
            
            crop  = self.kernel_size-1 - int(self.kernel_size/2)
            x = x[:,:,crop:-crop,crop:-crop]
            
            B,C,H,W = x.shape
            assert H == W and H == self.in_size, f"input tensor size({H},{W}) must be equal to in_size({self.in_size})"

            x += self.bias[None, :, None, None]
            x = x.reshape(B*C, 1, H, W)
            
            x = F.upsample(x, scale_factor=self.up_factor, mode='nearest')
            x = F.conv2d(x, torch.outer(self.up_filter,self.up_filter)[None, None, :, :], padding="same")
            x = F.leaky_relu(x, negative_slope=self.slope) * self.gain
            x = F.conv2d(x, torch.outer(self.dn_filter,self.dn_filter)[None, None, :, :], padding="same")
            x = grid_pool2d(x, self.dn_factor)

            _, _, H, W = x.shape
            assert H == W and H == self.out_size, "output tensor size must be equal to out_size"

            x = x.reshape(B, C, self.out_size, self.out_size)
            
        
        
        return x


class UpFIRDnConv2d(nn.Module):
    """
        Upsampling and downsampling with FIR filter
    """
    def __init__(self, 
                 in_channel,
                 out_channel,
                 kernel_size,
                 stride = 1,
                 transpose = False,
                 lrelu_upsampling=2,
                 filter_size=6,
                 jit=True,
                 **kwargs):
        super().__init__()

        self.out_channel    = out_channel
        self.jit            = jit 
        self.kernel_size    = kernel_size
        self.scale          = stride if transpose else 1 / stride
        self.filter_size    = filter_size
        self.lrelu_upsampling = lrelu_upsampling
        self.in_size        = None

        if jit:
            self.conv = nn.Conv2d(in_channel, out_channel, kernel_size, stride=1, padding=(kernel_size-1,kernel_size-1))
            self.upfirdn2d = UpFIRDn2dJIT(
                self.out_channel, 
                lrelu_upsampling=self.lrelu_upsampling, 
                filter_size=self.filter_size,
                kernel_size=self.kernel_size
            )
        else:
            self.conv = nn.Conv2d(in_channel, out_channel, kernel_size, stride=1, padding="same")
            self.upfirdn2d = UpFIRDn2d(
                self.out_channel, 
                lrelu_upsampling=self.lrelu_upsampling, 
                filter_size=self.filter_size,
            )

        self.reset_parameters()

    def reset_parameters(self):
        self.conv.reset_parameters()
        if self.upfirdn2d is not None:
            self.upfirdn2d.reset_parameters()
    
    @property
    def device(self):
        return next(iter(self.parameters())).device
    
    @property
    def dtype(self):
        return next(iter(self.parameters())).dtype

    def update_upfirdn2d_filter(self, in_size):
        """
            Parameters:
            -----------
                in_size: int, input tensor size
        """
        out_size = int( in_size * self.scale )
        self.upfirdn2d.init_kernel(in_size, out_size)
        
    def forward(self, x):
        """
            Parameters:
            -----------
                x: torch.Tensor, shape=(batch_size, in_channel, H, W)
            Returns:
            --------
                output: torch.Tensor, shape=(batch_size, out_channel, H, W)
        """

        _, _, H, W = x.shape 
        assert H == W 
        if self.in_size is None or H != self.in_size:
            # first time init or different size of input received
            self.in_size = H
            self.update_upfirdn2d_filter(self.in_size)
        
        x = self.conv(x)
        
        x = self.upfirdn2d(x)

        return x
        
class ResBlock2d(nn.Module):
    """
        Residual block for 2d convolution
        H_out = Conv2d(H_in) + Conv2d(\sigma(Conv2d^{num\_layers-1}(H_in)))
    """
    def __init__(self,
                in_channel,
                out_channel,
                num_layers = 2,
                kernel_size=3,
                activation='relu',
                stable_weight=0.9,
                Conv2d = Conv2d):
        super().__init__()
        assert num_layers >= 2, "num_layers must be greater than 2"
        self.stable_weight = stable_weight
        self.convs = nn.ModuleList([Conv2d(in_channel, out_channel, kernel_size, activation=activation)])
        for _ in range(num_layers-2):
            self.convs.append(Conv2d(out_channel, out_channel, kernel_size, activation=activation))
        self.convs.append(Conv2d(out_channel, out_channel, kernel_size, activation=None))
        
    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, x):
        """
            Parameters:
            -----------
                x: torch.Tensor, shape=(batch_size, in_channel, H, W)
            Returns:
            --------
                output: torch.Tensor, shape=(batch_size, out_channel, H, W)
        """
        out = x
        for conv in self.convs:
            out = conv(out)
        out = self.stable_weight * x + (1-self.stable_weight) * out
        return out

class DownBlock2d(nn.Module):
    def __init__(self, 
                 in_channel,
                 kernel_size=3,
                 scale_factor=2,
                 activation='relu',
                 Conv2d=UpFIRDnConv2d):
        super().__init__()
        self.output_channel = in_channel * (scale_factor**2)
        self.scale_factor = scale_factor
        self.bn_next   = nn.BatchNorm2d(in_channel)
        self.bn_down   = nn.BatchNorm2d(in_channel)
        self.conv_next = Conv2d(in_channel, in_channel, kernel_size, activation=activation)
        self.conv_down = Conv2d(in_channel, self.output_channel, kernel_size, stride=scale_factor, activation=activation)
        self.reset_parameters()

    def reset_parameters(self):
        self.conv_next.reset_parameters()
        self.conv_down.reset_parameters()
        self.bn_next.reset_parameters()
        self.bn_down.reset_parameters()

    def forward(self, x):
        """
            Parameters:
            -----------
                x: torch.Tensor, shape=(batch_size, in_channel, H, W)
            Returns:
            --------
                unscale_output: torch.Tensor, shape=(batch_size, in_channel, H, W)
                scale_output: torch.Tensor, shape=(batch_size, in_channel*(scale_factor^2), H//scale_factor, W//scale_factor)
        """
        x = self.bn_next(x)
        x = self.conv_next(x)
        x_next = x
        x = self.bn_down(x)
        x = self.conv_down(x)
        x_down = x
        return x_next, x_down

class UpBlock2d(nn.Module):
    def __init__(self, 
                 out_channel,
                 kernel_size=3,
                 scale_factor=2,
                 activation='relu',
                 jit=True,
                 Conv2d=UpFIRDnConv2d):
        super().__init__()
        self.input_channel = out_channel * (scale_factor**2)
        self.scale_factor = scale_factor
        self.bn_up     = nn.BatchNorm2d(self.input_channel)
        self.bn_next   = nn.BatchNorm2d(2 * out_channel)
        self.conv_up   = Conv2d(self.input_channel, out_channel, kernel_size, stride=scale_factor, transpose=True, activation=activation)
        self.conv_next = Conv2d(2*out_channel, out_channel, kernel_size, activation=activation)
        self.reset_parameters()
    
    def reset_parameters(self):
        self.conv_next.reset_parameters()
        self.conv_up.reset_parameters()
        self.bn_next.reset_parameters()
        self.bn_up.reset_parameters()

    def forward(self, x_up, x_skip):
        """
            Parameters:
            -----------
                x_up: torch.Tensor, shape=(batch_size, out_channel*(scale_factor^2), H//scale_factor, W//scale_factor)
                x_skip: torch.Tensor, shape=(batch_size, out_channel, H, W)
            Returns:
            --------
                x: torch.Tensor, shape=(batch_size, out_channel, H, W)
        """
        x_next = self.bn_up(x_up)
        x_next = self.conv_up(x_next)

        x  = torch.cat([x_next, x_skip], dim=1)

        x = self.bn_next(x)
        x = self.conv_next(x)

        return x

class GeneralUNet2d(nn.Module):
    """
                             |<---        num layers             --->|
        input - Conv - Conv - ResBlock*4 ---------------------------- Conv - Conv - output    /\
                        |                                              |                      |
                       ConvS2 - Conv - ResBlock*4 ------------ Conv - TConvS2                depth
                                  |                              |                            |
                                 ConvS2 - Conv - ResBlock*4 - TConvS2                         \/
                                
    """
    def __init__(self, 
                in_channel,
                out_channel,
                hidden_channel=16,
                num_layers=4,
                kernel_size=3,
                depth=3,
                activation="relu",
                Conv2d = UpFIRDnConv2d):
        super().__init__()
        self.depth = depth
        self.down_blocks = nn.ModuleList([])
        self.up_blocks   = nn.ModuleList([])
        self.res_blocks  = nn.ModuleList([])
        self.input_transform  = Conv2d(in_channel, hidden_channel, kernel_size, activation=activation) # lift
        self.output_transform = Conv2d(hidden_channel, out_channel, kernel_size, activation=None)      # project
          
        for d in range(depth-1):
            channel = hidden_channel * 4 ** d
            self.down_blocks.append(DownBlock2d(channel, kernel_size, scale_factor=2, activation=activation, Conv2d=Conv2d))            
            self.up_blocks.append(UpBlock2d(channel,  kernel_size, scale_factor=2, activation=activation, Conv2d=Conv2d))
        for d in range(depth):
            channel = hidden_channel * 4 ** d
            self.res_blocks.append(nn.Sequential(
                nn.BatchNorm2d(channel),
                *[ResBlock2d(channel, channel, num_layers=num_layers, kernel_size=kernel_size, activation=activation, Conv2d=Conv2d)
            ]))
            
        self.reset_parameters()

    def reset_parameters(self):
        self.input_transform.reset_parameters()
        self.output_transform.reset_parameters()
        for up_block in self.up_blocks:
            up_block.reset_parameters()
        for down_block in self.down_blocks:
            down_block.reset_parameters()
        for res_block in self.res_blocks:
            for layer in res_block:
                layer.reset_parameters()

    def forward(self, x):
        """
            UNet architecture
            Parameters:
            -----------
                x: torch.Tensor, shape=(batch_size, in_channel, H, W) 
            Returns:
            --------
                output: torch.Tensor, shape=(batch_size, out_channel, H, W)
        """
        down_feature = self.input_transform(x)
  
        skips = []
        for d in range(self.depth-1):
            next_feature, down_feature = self.down_blocks[d](down_feature)
            res_feature = self.res_blocks[d](next_feature)
            skips.append(res_feature)
    
        up_feature = self.res_blocks[-1](down_feature)
        
        for d in range(self.depth-2, -1, -1):
            up_feature = self.up_blocks[d](up_feature, skips[d])
        
        x = self.output_transform(up_feature)

        return x
        
class UNet2d(GeneralUNet2d):
    def __init__(self, 
                in_channel,
                out_channel,
                hidden_channel=16,
                num_layers=4,
                kernel_size=3,
                depth=3,
                activation="relu"):
        super().__init__(in_channel, out_channel, hidden_channel, num_layers, kernel_size, depth, activation, Conv2d=Conv2d)

class CNO2d(GeneralUNet2d):
    def __init__(self, 
                in_channel,
                out_channel,
                hidden_channel=16,
                num_layers=4,
                kernel_size=3,
                depth=3,
                jit=True,
                activation=None):
        super().__init__(in_channel, out_channel, hidden_channel, num_layers, kernel_size, depth, activation=None, 
                         Conv2d=partial(UpFIRDnConv2d,  jit=jit) ) 

