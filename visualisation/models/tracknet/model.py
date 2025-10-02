# ABOUTME: TrackNetV2 model implementation for badminton ball tracking
# ABOUTME: Implements U-Net-like architecture with CBAM attention mechanisms for ball detection

import torch
import torch.nn as nn
import torch.nn.functional as F

class ChannelAttentionModule(nn.Module):
    def __init__(self, channel, ratio=16):
        super(ChannelAttentionModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.shared_MLP = nn.Sequential(
            nn.Conv2d(channel, channel // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(channel // ratio, channel, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = self.shared_MLP(self.avg_pool(x))
        maxout = self.shared_MLP(self.max_pool(x))
        return self.sigmoid(avgout + maxout)

class SpatialAttentionModule(nn.Module):
    def __init__(self):
        super(SpatialAttentionModule, self).__init__()
        self.conv2d = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=7, stride=1, padding=3)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = torch.mean(x, dim=1, keepdim=True)
        maxout, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avgout, maxout], dim=1)
        out = self.sigmoid(self.conv2d(out))
        return out

class CBAM(nn.Module):
    def __init__(self, channel):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttentionModule(channel)
        self.spatial_attention = SpatialAttentionModule()

    def forward(self, x):
        out = self.channel_attention(x) * x
        out = self.spatial_attention(out) * out
        return out

class Conv2DBlock(nn.Module):
    """Conv + BN + ReLU"""
    def __init__(self, in_dim, out_dim, kernel_size, padding='same', bias=True):
        super(Conv2DBlock, self).__init__()
        self.conv = nn.Conv2d(in_dim, out_dim, kernel_size=kernel_size, padding=padding, bias=bias)
        self.bn = nn.BatchNorm2d(out_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class Double2DConv(nn.Module):
    """Conv2DBlock x 2"""
    def __init__(self, in_dim, out_dim):
        super(Double2DConv, self).__init__()
        self.conv_1 = Conv2DBlock(in_dim, out_dim, (3, 3))
        self.conv_2 = Conv2DBlock(out_dim, out_dim, (3, 3))

    def forward(self, x):
        x = self.conv_1(x)
        x = self.conv_2(x)
        return x

class Double2DConv2(nn.Module):
    """Multi-scale Conv2DBlock with 1x1, 3x3, and 5x5 kernels"""
    def __init__(self, in_dim, out_dim):
        super(Double2DConv2, self).__init__()
        self.conv_1 = Conv2DBlock(in_dim, out_dim, (1, 1))
        self.conv_2 = Conv2DBlock(out_dim, out_dim, (3, 3))

        self.conv_3 = Conv2DBlock(in_dim, out_dim, (3, 3))
        self.conv_4 = Conv2DBlock(out_dim, out_dim, (3, 3))

        self.conv_5 = Conv2DBlock(in_dim, out_dim, (5, 5))
        self.conv_6 = Conv2DBlock(out_dim, out_dim, (3, 3))

        self.conv_7 = Conv2DBlock(out_dim*3, out_dim, (3, 3))

    def forward(self, x):
        x1 = self.conv_1(x)
        x1 = self.conv_2(x1)

        x2 = self.conv_3(x)
        x2 = self.conv_4(x2)

        x3 = self.conv_5(x)
        x3 = self.conv_6(x3)

        x = torch.cat([x1, x2, x3], dim=1)

        x = self.conv_7(x)
        x = x + x2

        return x

class Triple2DConv(nn.Module):
    """Conv2DBlock x 3"""
    def __init__(self, in_dim, out_dim):
        super(Triple2DConv, self).__init__()
        self.conv_1 = Conv2DBlock(in_dim, out_dim, (3, 3))
        self.conv_2 = Conv2DBlock(out_dim, out_dim, (3, 3))
        self.conv_3 = Conv2DBlock(out_dim, out_dim, (3, 3))

    def forward(self, x):
        x = self.conv_1(x)
        x = self.conv_2(x)
        x = self.conv_3(x)
        return x

class TrackNetV2(nn.Module):
    def __init__(self, input_channels=9, out_channels=3):
        super(TrackNetV2, self).__init__()

        # Down blocks
        self.down_block_1 = Double2DConv2(in_dim=input_channels, out_dim=64)
        self.down_block_2 = Double2DConv2(in_dim=64, out_dim=128)
        self.down_block_3 = Double2DConv2(in_dim=128, out_dim=256)

        # Bottleneck
        self.bottleneck = Triple2DConv(in_dim=256, out_dim=512)

        # Up blocks
        self.up_block_1 = Double2DConv(in_dim=768, out_dim=256)
        self.up_block_2 = Double2DConv(in_dim=384, out_dim=128)
        self.up_block_3 = Double2DConv(in_dim=192, out_dim=64)

        # Output predictor
        self.predictor = nn.Conv2d(64, out_channels, (1, 1))
        self.sigmoid = nn.Sigmoid()

        # CBAM modules
        self.cbam1 = CBAM(channel=256)
        self.cbam2 = CBAM(channel=128)
        self.cbam3 = CBAM(channel=64)
        self.cbam0_2 = CBAM(channel=256)
        self.cbam1_2 = CBAM(channel=128)
        self.cbam2_2 = CBAM(channel=64)

    def forward(self, x):
        # Downsampling path
        x1 = self.down_block_1(x)  # (batch, 64, H, W)
        x = nn.MaxPool2d((2, 2), stride=(2, 2))(x1)  # (batch, 64, H/2, W/2)

        x2 = self.down_block_2(x)  # (batch, 128, H/2, W/2)
        x = nn.MaxPool2d((2, 2), stride=(2, 2))(x2)  # (batch, 128, H/4, W/4)

        x3 = self.down_block_3(x)  # (batch, 256, H/4, W/4)
        x = nn.MaxPool2d((2, 2), stride=(2, 2))(x3)  # (batch, 256, H/8, W/8)

        # Bottleneck
        x = self.bottleneck(x)  # (batch, 512, H/8, W/8)

        # Upsampling path with skip connections and CBAM attention
        x3 = self.cbam0_2(x3)
        x = torch.cat([nn.Upsample(scale_factor=2)(x), x3], dim=1)  # (batch, 768, H/4, W/4)
        x = self.up_block_1(x)  # (batch, 256, H/4, W/4)
        x = self.cbam1(x)

        x2 = self.cbam1_2(x2)
        x = torch.cat([nn.Upsample(scale_factor=2)(x), x2], dim=1)  # (batch, 384, H/2, W/2)
        x = self.up_block_2(x)  # (batch, 128, H/2, W/2)
        x = self.cbam2(x)

        x1 = self.cbam2_2(x1)
        x = torch.cat([nn.Upsample(scale_factor=2)(x), x1], dim=1)  # (batch, 192, H, W)
        x = self.up_block_3(x)  # (batch, 64, H, W)
        x = self.cbam3(x)

        # Final prediction
        x = self.predictor(x)  # (batch, out_channels, H, W)
        x = self.sigmoid(x)

        return x

def create_model(input_channels=9, out_channels=3):
    return TrackNetV2(input_channels=input_channels, out_channels=out_channels)

if __name__ == "__main__":
    # Test the model
    model = create_model()
    print(f"Model created with input channels: 9, output channels: 3")

    # Test with dummy input
    x = torch.randn(1, 9, 288, 512)
    output = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")

    # Print total parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")