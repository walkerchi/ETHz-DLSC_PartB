import torch 
import torch.nn as nn 
import torch.nn.functional as F


class Conv2d(nn.Conv2d):
    """
        Conv2d with activation function
    """
    def __init__(self, 
                 in_channels,
                 out_channels,
                 kernel_size,
                 padding="same",
                 activation="relu"):
        assert padding in ["same", "valid"]
        padding = {"same": kernel_size//2, "valid": 0}[padding]
        super().__init__(in_channels, out_channels, kernel_size, padding=padding)
        if activation is None:
            self.activation_fn = None
        elif activation in ["tanh", "sigmoid"]:
            self.activation_fn = getattr(torch, activation)
        else:
            self.activation_fn = getattr(F, activation)
    def forward(self, x):
        """
            Parameters:
            -----------
                x: torch.Tensor, shape=(batch_size, in_channel, H, W)
            Returns:
            --------
                output: torch.Tensor, shape=(batch_size, out_channel, H, W)
        """
        x = super().forward(x)
        if self.activation_fn is not None:
            x = self.activation_fn(x)
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
                use_res  = True,
                kernel_size=3,
                activation='relu'):
        super().__init__()

        self.convs = nn.ModuleList([Conv2d(in_channel, out_channel, kernel_size, activation=activation)])
        for _ in range(num_layers-2):
            self.convs.append(Conv2d(out_channel, out_channel, kernel_size, activation=activation))
        self.convs.append(Conv2d(out_channel, out_channel, kernel_size, activation=None))

        if use_res:
            self.res_conv = Conv2d(in_channel, out_channel, kernel_size, activation=None)
        else:
            self.res_conv = None
        
    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        if  self.res_conv is not None:
            self.res_conv.reset_parameters()

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
        if self.res_conv is not None:
            out = out + self.res_conv(x)
        return out

class DownBlock2d(nn.Module):
    def __init__(self, 
                 in_channel,
                 kernel_size=3,
                 num_layers=2,
                 scale_factor=2,
                 activation='relu'):
        super().__init__()
        self.output_channel = in_channel * (scale_factor**2)
        self.scale_factor = scale_factor
        self.convs = nn.ModuleList([])
        for _ in range(num_layers-1):
            self.convs.append(Conv2d(in_channel, in_channel, kernel_size, activation=activation))
        self.scale_conv = nn.Conv2d(in_channel, in_channel*(scale_factor**2), kernel_size, stride=scale_factor, padding=kernel_size//2)
        self.reset_parameters()

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        self.scale_conv.reset_parameters()

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
        for conv in self.convs:
            x = conv(x)
        unscale_output = x
        scale_output = self.scale_conv(x)
        return unscale_output, scale_output


class UpBlock2d(nn.Module):
    def __init__(self, 
                 out_channel,
                 kernel_size=3,
                 num_layers=2,
                 scale_factor=2,
                 activation='relu'):
        super().__init__()

        self.scale_factor = scale_factor
        self.scale_conv = nn.ConvTranspose2d(out_channel*(scale_factor**2), out_channel, kernel_size, stride=scale_factor, padding=kernel_size//2, output_padding=1)
        self.convs = nn.ModuleList([Conv2d(out_channel*2, out_channel, kernel_size, activation=activation)])
        for _ in range(num_layers-2):
            self.convs.append(Conv2d(out_channel, out_channel, kernel_size, activation=activation))
        self.reset_parameters()
    
    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        self.scale_conv.reset_parameters()

    def forward(self, scale_input, unscale_input):
        """
            Parameters:
            -----------
                scale_input: torch.Tensor, shape=(batch_size, out_channel*(scale_factor^2), H//scale_factor, W//scale_factor)
                unscale_input: torch.Tensor, shape=(batch_size, out_channel, H, W)
            Returns:
            --------
                output: torch.Tensor, shape=(batch_size, out_channel, H, W)
        """
        scale_input = self.scale_conv(scale_input)
        output = torch.cat([scale_input, unscale_input], dim=1)
        for conv in self.convs:
            output = conv(output)
        return output

class UNet2d(nn.Module):
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
                hidden_size=16,
                num_layers=4,
                kernel_size=3,
                depth=3,
                activation="relu"):
        super().__init__()
        self.depth = depth
        self.down_blocks = nn.ModuleList([])
        self.up_blocks   = nn.ModuleList([])
        self.res_blocks  = nn.ModuleList([])
        self.input_transform  = Conv2d(in_channel, hidden_size, kernel_size, activation=activation)
        self.output_transform = Conv2d(hidden_size, out_channel, kernel_size, activation=None)
        
        for d in range(depth-1):
            self.down_blocks.append(DownBlock2d(hidden_size * 4 ** d, kernel_size, num_layers=2, scale_factor=2, activation=activation))            
            self.up_blocks.append(UpBlock2d(hidden_size * 4 **d,  kernel_size, num_layers=2, scale_factor=2, activation=activation))
        for d in range(depth):
            self.res_blocks.append(nn.Sequential(*[
                ResBlock2d(hidden_size * 4**d, hidden_size * 4**d, num_layers=num_layers, kernel_size=kernel_size, activation=activation)
            ]))
        bottom_channel = hidden_size * 4**(depth - 1)
        self.bottom_down_transform = Conv2d(bottom_channel, bottom_channel, kernel_size, activation=activation)
        self.bottom_up_transform   = Conv2d(bottom_channel, bottom_channel, kernel_size, activation=activation)
        self.reset_parameters()

    def reset_parameters(self):
        self.input_transform.reset_parameters()
        self.output_transform.reset_parameters()
        self.bottom_up_transform.reset_parameters()
        self.bottom_down_transform.reset_parameters()
        for down_block, up_block, res_block in zip(self.down_blocks, self.up_blocks, self.res_blocks):
            down_block.reset_parameters()
            up_block.reset_parameters()
            for res in res_block:
                res.reset_parameters()

    def forward(self, x):
        """
            Parameters:
            -----------
                x: torch.Tensor, shape=(batch_size, in_channel, H, W) 
            Returns:
            --------
                output: torch.Tensor, shape=(batch_size, out_channel, H, W)
        """
        down_feature = self.input_transform(x)
             
        shortcuts = []
        for d in range(self.depth-1):
            next_feature, down_feature = self.down_blocks[d](down_feature)
            res_feature = self.res_blocks[d](next_feature)
            shortcuts.append(res_feature)

        bottom_feature = self.bottom_down_transform(down_feature)
        bottom_feature = self.res_blocks[self.depth-1](bottom_feature)
        up_feature = self.bottom_up_transform(bottom_feature)
        
        for d in range(self.depth-2, -1, -1):
            up_feature = self.up_blocks[d](up_feature, shortcuts[d])
        
        x = self.output_transform(up_feature)
        return x
        

class CNO2d(UNet2d):
    pass
