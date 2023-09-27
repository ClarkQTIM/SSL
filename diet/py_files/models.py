# Imports

from typing import Sequence
from monai.networks.nets import UNet, AttentionUnet, ResNet, resnet18, resnet50, resnet101
import torch
from torch import nn

# DIET_UNet
class DIET_UNet(nn.Module):
    def __init__(self, spatial_dims, in_channels, out_channels, channels, strides, dropout, num_classes, norm):
        super(DIET_UNet, self).__init__()
        self.spatial_dims: int = spatial_dims
        self.in_channels: int = in_channels
        self.out_channels: int = out_channels
        self.channels: Sequence[int] = channels
        self.strides: Sequence[int] = strides
        self.dropout: float = dropout
        self.num_classes: int = num_classes
        self.normalization: str = norm

        # f_theta
        self.loc_model = UNet(spatial_dims = spatial_dims,
                    in_channels = in_channels,
                    out_channels = out_channels,
                    channels = channels,
                    strides = strides,
                    norm = norm,
                    dropout = dropout
                )

    # Forward Pass
    def forward(self, x):
        loc_output = self.loc_model(x) # Getting the spatial localization
        self.regressor_model = nn.Linear(loc_output.shape[1]*loc_output.shape[2]*loc_output.shape[3], self.num_classes).to(loc_output.device)
        logits = self.regressor_model(loc_output.view(len(loc_output), -1))
        return logits
    
# DIET_AttentionUNet
class DIET_AttentionUNet(nn.Module):
    def __init__(self, spatial_dims, in_channels, out_channels, channels, strides, dropout, num_classes, norm):
        super(DIET_AttentionUNet, self).__init__()
        self.spatial_dims: int = spatial_dims
        self.in_channels: int = in_channels
        self.out_channels: int = out_channels
        self.channels: Sequence[int] = channels
        self.strides: Sequence[int] = strides
        self.dropout: float = dropout
        self.num_classes: int = num_classes
        self.normalization: str = norm

        # f_theta
        self.loc_model = AttentionUnet(spatial_dims = spatial_dims,
                in_channels = in_channels,
                out_channels = out_channels,
                channels = channels,
                strides = strides,
                dropout = dropout
            )

    # Forward Pass
    def forward(self, x):
        loc_output = self.loc_model(x) # Getting the spatial localization
        self.regressor_model = nn.Linear(loc_output.shape[1]*loc_output.shape[2]*loc_output.shape[3], self.num_classes).to(loc_output.device)
        logits = self.regressor_model(loc_output.view(len(loc_output), -1))
        return logits
    
# ResNet models built from: https://docs.monai.io/en/stable/_modules/monai/networks/nets/resnet.html#ResNet

# DIET Resnet18
class DIET_ResNet18(nn.Module):
    def __init__(self, spatial_dims, n_input_channels, num_classes):
        super(DIET_ResNet18, self).__init__()
        self.spatial_dims = 2,
        self.n_input_channels = 3,
        self.num_classes: int = num_classes

        # f_theta
        self.loc_model = ResNet(spatial_dims = spatial_dims,
                n_input_channels = n_input_channels,
                block = "basic",
                layers = [1, 1, 1, 1],
                block_inplanes = [64, 128, 256, 512], 
                feed_forward=False       
            )

    # Forward Pass
    def forward(self, x):
        loc_output = self.loc_model(x) # Getting the spatial localization
        self.regressor_model = nn.Linear(loc_output.shape[1], self.num_classes).to(loc_output.device)
        logits = self.regressor_model(loc_output.view(len(loc_output), -1))
        return logits

# DIET Resnet50
class DIET_ResNet50(nn.Module):
    def __init__(self, spatial_dims, n_input_channels, num_classes):
        super(DIET_ResNet50, self).__init__()
        self.spatial_dims = 2,
        self.n_input_channels = 3,
        self.num_classes: int = num_classes

        # f_theta
        self.loc_model = ResNet(spatial_dims = spatial_dims,
                n_input_channels = n_input_channels,
                block = "basic",
                layers = [3, 4, 6, 3],
                block_inplanes = [64, 128, 256, 512], 
                feed_forward=False       
            )

    # Forward Pass
    def forward(self, x):
        loc_output = self.loc_model(x) # Getting the spatial localization
        self.regressor_model = nn.Linear(loc_output.shape[1], self.num_classes).to(loc_output.device)
        logits = self.regressor_model(loc_output.view(len(loc_output), -1))
        return logits

# DIET Resnet101
class DIET_ResNet101(nn.Module):
    def __init__(self, spatial_dims, n_input_channels, num_classes):
        super(DIET_ResNet101, self).__init__()
        self.spatial_dims = 2,
        self.n_input_channels = 3,
        self.num_classes: int = num_classes

        # f_theta
        self.loc_model = ResNet(spatial_dims = spatial_dims,
                n_input_channels = n_input_channels,
                block = "basic",
                layers = [3, 4, 23, 3],
                block_inplanes = [64, 128, 256, 512], 
                feed_forward=False       
            )

    # Forward Pass
    def forward(self, x):
        loc_output = self.loc_model(x) # Getting the spatial localization
        self.regressor_model = nn.Linear(loc_output.shape[1], self.num_classes).to(loc_output.device)
        logits = self.regressor_model(loc_output.view(len(loc_output), -1))
        return logits

__supported_networks__ = {
    'unet': DIET_UNet,
    'attentionunet': DIET_AttentionUNet,
    'res18': resnet18(spatial_dims=2, n_input_channels=3, num_classes=100),
    'res50': resnet50(spatial_dims=2, n_input_channels=3, num_classes=100),
    'res101': resnet101(spatial_dims=2, n_input_channels=3, num_classes=100),
}

# __supported_schedulers__ = {
#     'consineannealing': torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = 32)
# }