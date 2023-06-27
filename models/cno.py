import torch 
import torch.nn as nn 
import torch.nn.functional as F
import numpy as np
import scipy.signal

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
            self.sampling = nn.Upsample(scale_factor=stride, mode='nearest')
        else:
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=padding)
            self.sampling = nn.AvgPool2d(stride)

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
                in_size,
                out_size, 
                lrelu_upsampling=2,
                filter_size=6,
                gain=np.sqrt(2),
                slope=0.2):
        super().__init__()
        self.filter_size = filter_size
        self.lrelu_upsampling = lrelu_upsampling

        self.bias = nn.Parameter(torch.zeros(1, channel, 1, 1).float())

        up_factor           = int(np.ceil(max(in_size,out_size) * self.lrelu_upsampling/ in_size))
        dn_factor           = int(np.ceil(max(in_size,out_size) * self.lrelu_upsampling/ out_size))
        self.up_factor      = up_factor
        self.dn_factor      = dn_factor
             
        self.up_sample      = nn.Upsample(scale_factor=up_factor, mode='nearest')
        self.dn_sample      = nn.AvgPool2d(dn_factor)
        # x > 0: gain *x ; x <=0 : gain * slope * x
        self.activation_fn  = lambda x: F.leaky_relu(x, negative_slope=slope) * gain

        self.init_kernel(in_size, out_size)
        self.reset_parameters()

    @property
    def device(self):
        return self.bias.device

    def init_kernel(self, in_size, out_size):
        self.in_size  = in_size
        self.out_size = out_size

        uptaps              = self.filter_size * self.up_factor
        dntaps              = self.filter_size * self.dn_factor
      
        up_filter           = torch.tensor(scipy.signal.firwin(numtaps=uptaps, cutoff=in_size/2.0001, width=in_size, fs=in_size))
        dn_filter           = torch.tensor(scipy.signal.firwin(numtaps=dntaps, cutoff=out_size/2.0001, width=out_size, fs=out_size))
        # 1d kernel to 2d kernel by outer product [kernel_size] -> [1, 1, kernel_size, kernel_size]
        up_filter           = torch.outer(up_filter, up_filter)[None, None, :, :].float()
        dn_filter           = torch.outer(dn_filter, dn_filter)[None, None, :, :].float()

        self.register_buffer('up_filter', up_filter)
        self.register_buffer('dn_filter', dn_filter)

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
                 padding:str="same",
                 lrelu_upsampling=2,
                 filter_size=6,
                 **kwargs):
        super().__init__()

        self.out_channel = out_channel

        self.conv = nn.Conv2d(in_channel, out_channel, kernel_size, stride=1, padding=padding)
        
        self.scale = stride if transpose else 1 / stride
        self.filter_size= filter_size
        self.lrelu_upsampling = lrelu_upsampling
        self.in_size        = None
        self.upfirdn2d      = None

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

    def init_upfirdn2d_filter(self, in_size):
        """
            Parameters:
            -----------
                in_size: int, input tensor size
        """
        out_size = int( in_size * self.scale )
        self.upfirdn2d = UpFIRDn2d(self.out_channel, in_size, out_size, lrelu_upsampling=self.lrelu_upsampling, filter_size=self.filter_size)
        self.upfirdn2d.to(self.device)

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
        if self.in_size is None:
            # first time init
            self.in_size = H
            self.init_upfirdn2d_filter(self.in_size)
        elif H != self.in_size:
            # different size of input received
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
                **kwargs):
        super().__init__(in_channel, out_channel, hidden_channel, num_layers, kernel_size, depth, activation=None, Conv2d=UpFIRDnConv2d)


