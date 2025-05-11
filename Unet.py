import torch
import torch.nn as nn
from loss import Pairwise_Feature_Similarity
from pytorch_patched_decoder import PatchedTimeSeriesDecoder
# import timesfm_local as timesfm
import logging

# Define the Projector class
class Projector(nn.Module):
    def __init__(self, input_nc, nc=256, norm=nn.BatchNorm1d, activation=nn.ReLU(inplace=True), device=None):
        super(Projector, self).__init__()

        self.device = device  # 保存设备信息

        self.mlp = nn.Sequential(
            nn.Linear(input_nc, nc, bias=False),
            norm(nc),
            activation,
            nn.Linear(nc, nc, bias=False),
            norm(nc),
            activation,
            nn.Linear(nc, nc, bias=False),
            norm(nc, affine=False)  # batch norm without learnable parameters
        ).to(self.device)  # 确保 MLP 层被移动到指定设备上

    def forward(self, x):
        return self.mlp(x)

class DoubleConv3D(nn.Module):
    def __init__(self, in_channels, out_channels, use_residual=False):
        super(DoubleConv3D, self).__init__()
        self.use_residual = use_residual
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.conv_block = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1 if out_channels <= 32 else 0.2 if out_channels <= 128 else 0.3),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )

        # 如果输入输出通道不一样，残差连接需要额外的映射
        if self.use_residual and in_channels != out_channels:
            self.residual_conv = nn.Conv3d(in_channels, out_channels, kernel_size=1)
        else:
            self.residual_conv = None

    def forward(self, x):
        out = self.conv_block(x)
        if self.use_residual:
            residual = self.residual_conv(x) if self.residual_conv else x
            out = out + residual
        return out
      
class UNet3D(nn.Module):
    def __init__(self, opt, in_channels, out_channels, base_channel = 16, device=None):
        super().__init__()
    
        #### networks
        # Encoder
        self.encoder1 = DoubleConv3D(in_channels=in_channels, out_channels=base_channel)
        self.pool1 = nn.MaxPool3d(kernel_size=2)
    
        self.encoder2 = DoubleConv3D(in_channels=base_channel, out_channels=base_channel * 2)
        self.pool2 = nn.MaxPool3d(kernel_size=2)
    
        self.encoder3 = DoubleConv3D(in_channels=base_channel * 2, out_channels=base_channel * 4)
        self.pool3 = nn.MaxPool3d(kernel_size=2)
    
        self.encoder4 = DoubleConv3D(in_channels=base_channel * 4, out_channels=base_channel * 8)
        self.pool4 = nn.MaxPool3d(kernel_size=2)

        # BottleNeck
        self.bottleneck = DoubleConv3D(in_channels=base_channel * 8, out_channels=base_channel * 16)
        
        # Decoder
        self.upconv4 = nn.ConvTranspose3d(in_channels=base_channel * 16, out_channels=base_channel * 8, kernel_size=2, stride=2)
        self.decoder4 = DoubleConv3D(in_channels=base_channel * 16, out_channels=base_channel * 8)
    
        self.upconv3 = nn.ConvTranspose3d(in_channels=base_channel * 8, out_channels=base_channel * 4, kernel_size=2, stride=2)
        self.decoder3 = DoubleConv3D(in_channels=base_channel * 8, out_channels=base_channel * 4)
    
        self.upconv2 = nn.ConvTranspose3d(in_channels=base_channel * 4, out_channels=base_channel * 2, kernel_size=2, stride=2)
        self.decoder2 = DoubleConv3D(in_channels=base_channel * 4, out_channels=base_channel * 2)
    
        self.upconv1 = nn.ConvTranspose3d(in_channels=base_channel * 2, out_channels=base_channel, kernel_size=2, stride=2)
        self.decoder1 = DoubleConv3D(in_channels=base_channel * 2, out_channels=base_channel)
    
        self.out_conv = nn.Conv3d(in_channels=base_channel, out_channels=out_channels, kernel_size=1)

        self.mod = opt.mod

    def forward(self, x):
        
        #################
        #### Encoder ####
        #################

        e1 = self.encoder1(x)
        p1 = self.pool1(e1)
    
        e2 = self.encoder2(p1)
        p2 = self.pool2(e2)
    
        e3 = self.encoder3(p2)
        p3 = self.pool3(e3)
    
        e4 = self.encoder4(p3)
        p4 = self.pool4(e4)
        
        bt = self.bottleneck(p4)

        #################
        #### Decoder ####
        #################

        d4 = self.upconv4(bt)  # upscale
        d4 = torch.cat([d4, e4], dim=1)  # skip connections along channel dim
        d4 = self.decoder4(d4)
    
        d3 = self.upconv3(d4)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.decoder3(d3)
    
        d2 = self.upconv2(d3)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.decoder2(d2)
    
        d1 = self.upconv1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.decoder1(d1)
    

        outputs = self.out_conv(d1)
        # if self.mod == 'long':
        #     ch_len = outputs.shape[1] // 2
        #     output1 = outputs[:,:ch_len,::]
        #     output2 = outputs[:,ch_len:,::]
        #     return [output1, output2]
        # else:

        return outputs

    