"""
CeLDA+ V7 - Medical Landmark Detection Network
Soft-Argmax Coordinate Regression with Learnable Temperature

Key Features:
1. UNet-style Encoder-Decoder backbone
2. Learnable prototype with Xavier initialization
3. Learnable temperature for similarity scaling
4. Simple feature refinement
5. Support for Soft-Argmax differentiable coordinate decoding
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConvnnUNet(nn.Module):
    """nnUNet-style double-convolution block (conv -> IN -> LeakyReLU) * 2"""
    
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=True),
            nn.InstanceNorm2d(mid_channels, eps=1e-5, affine=True),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=True),
            nn.InstanceNorm2d(out_channels, eps=1e-5, affine=True),
            nn.LeakyReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class DownnnUNet(nn.Module):
    """nnUNet-style downsampling block"""
    
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConvnnUNet(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class UpnnUNet(nn.Module):
    """nnUNet-style upsampling block"""
    
    def __init__(self, in_channels, skip_channels, out_channels, bilinear=True):
        super().__init__()
        
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            concat_channels = in_channels + skip_channels
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            concat_channels = in_channels // 2 + skip_channels
        
        self.conv = DoubleConvnnUNet(concat_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class Encoder(nn.Module):
    def __init__(self, n_channels=1, bilinear=False, features=[32, 64, 128, 256, 512]):
        super().__init__()
        self.n_channels = n_channels
        self.bilinear = bilinear
        
        self.inc = DoubleConvnnUNet(n_channels, features[0])
        self.down1 = DownnnUNet(features[0], features[1])
        self.down2 = DownnnUNet(features[1], features[2])
        self.down3 = DownnnUNet(features[2], features[3])
        self.down4 = DownnnUNet(features[3], features[4])

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        return [x1, x2, x3, x4, x5]


class Decoder(nn.Module):
    def __init__(self, n_channels=1, bilinear=False, features=[32, 64, 128, 256, 512]):
        super().__init__()
        self.n_channels = n_channels
        self.bilinear = bilinear

        factor = 2 if bilinear else 1
       
        self.up1 = UpnnUNet(features[4], features[3], features[3] // factor, bilinear)  
        self.up2 = UpnnUNet(features[3], features[2], features[2] // factor, bilinear)  
        self.up3 = UpnnUNet(features[2], features[1], features[1] // factor, bilinear)  
        self.up4 = UpnnUNet(features[1], features[0], features[0], bilinear)           

    def forward(self, features):
        x1, x2, x3, x4, x5 = features
        d4 = self.up1(x5, x4)
        d3 = self.up2(d4, x3)
        d2 = self.up3(d3, x2)
        d1 = self.up4(d2, x1)
        
        return [d1, d2, d3]


class SimpleFeatureRefinement(nn.Module):
    """Simple feature refinement module"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.InstanceNorm2d(in_channels, affine=True)
        self.act1 = nn.ReLU(inplace=True)
        
        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.InstanceNorm2d(out_channels, affine=True)
        self.act2 = nn.ReLU(inplace=True)
    
    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act1(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.act2(out)
        
        return out


class CeLDAPlus(nn.Module):
    """
    CeLDA+ V7 - Medical Landmark Detection Network
    
    Args:
        in_channels: Number of input channels (1 for grayscale, 3 for RGB)
        bilinear: Use bilinear upsampling in decoder
        features: Feature dimensions for each encoder level
        landmark_num: Number of landmarks to detect
    """
    def __init__(self, in_channels=1, bilinear=False, features=[32, 64, 128, 256, 512], landmark_num=46):
        super().__init__()
        self.bilinear = bilinear
        self.landmark_num = landmark_num
        
        # Feature dimension from multi-scale concatenation
        feature_dim = features[0] + features[1] + features[2]  # 32+64+128=224
        
        # Learnable prototype with Xavier initialization
        self.prototype = nn.Parameter(torch.randn(landmark_num, feature_dim))
        nn.init.xavier_uniform_(self.prototype)
        
        # Learnable temperature for similarity scaling
        self.temperature = nn.Parameter(torch.ones(1))
        
        # Encoder-Decoder backbone
        self.encoder = Encoder(in_channels, bilinear, features)
        self.decoder = Decoder(in_channels, bilinear, features)
        
        # Simple feature refinement
        self.refine_head = SimpleFeatureRefinement(feature_dim, feature_dim)
        
        # Prototype MLP (for compatibility)
        self.prototype_mlp = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feature_dim, feature_dim),
        )
        
    def forward(self, x, return_similarity=False):
        """
        Forward pass
        
        Args:
            x: Input image tensor [B, C, H, W]
            return_similarity: If True, return similarity maps for Soft-Argmax decoding
        
        Returns:
            If return_similarity=True: similarity maps [B, K, H, W]
            Else: decoder features list
        """
        # Encoder
        enc_features = self.encoder(x)
        
        # Decoder
        dec_features = self.decoder(enc_features)
        
        if return_similarity:
            # Interpolate all to full resolution
            target_size = x.shape[-2:]
            features_upsampled = [
                F.interpolate(feat, size=target_size, mode='bilinear', align_corners=True)
                for feat in dec_features
            ]
            
            # Concatenate multi-scale features
            features_cat = torch.cat(features_upsampled, dim=1)  # [B, 224, H, W]
            
            # Simple refinement
            features_refined = self.refine_head(features_cat)  # [B, 224, H, W]
            
            # Dot product similarity
            similarity = torch.einsum('kd,bdhw->bkhw', self.prototype, features_refined.float())
            
            # Apply learnable temperature
            similarity = similarity * self.temperature
            
            return similarity
        else:
            return dec_features
