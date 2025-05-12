import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DataParallel
import os
import glob
import re
import random
import time


class BinaryImageDataset(Dataset):
    def __init__(self, numpy_arrays):
        # Convert numpy arrays to tensors and ensure proper shape
        self.data = torch.FloatTensor(numpy_arrays)
        if len(self.data.shape) == 3:  # If input is (n_samples, height, width)
            self.data = self.data.unsqueeze(1)  # Add channel dimension
            
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]
    

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
        self.relu = nn.LeakyReLU()
        
    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        out = self.relu(out)
        return out

class ThirdAutoencoder(nn.Module):
    def __init__(self, base_channels=32,dropout_rate=0.05):
        super(ThirdAutoencoder, self).__init__()
        self.dropout_rate=dropout_rate
        
        # Encoder Layers (Works with 4x reduced)
        self.encoder_conv0 = nn.Conv2d(1, 4, kernel_size=3, stride=1, padding=1) # 256x256 -> 256x256
        self.encoder_conv1 = nn.Conv2d(4, 8, kernel_size=3, stride=2, padding=1) # 256x256 -> 128x128
        self.encoder_resid0 = ResidualBlock(8)
        self.encoder_conv2 = nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=1) # 128x128 -> 64x64
        self.encoder_conv3 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1) # 64x64 -> 32x32
        self.encoder_resid1 = ResidualBlock(32)
        self.encoder_conv4 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1) # 32x32 -> 16x16
        self.encoder_conv5 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1) # 16x16 -> 8x8
        self.encoder_conv6 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1) # 8x8 -> 4x4
        self.encoder_resid2 = ResidualBlock(256)

        self.flatten = nn.Flatten()  # Flatten 4x4x1024/4 -> 16384/4
        
        # Add Nonlinear Dense Layers before the bottleneck
        self.encoder_fc1 = nn.Linear(16*256, 16*64)  # Fully connected: 16384/4 -> 512/2
        self.encoder_fc2 = nn.Linear(16*64, 16*16)       # Fully connected: 512/2 -> 128/2
        self.encoder_fc3 = nn.Linear(16*16, 16*4)       # Fully connected: 512/2 -> 128/2
        self.encoder_fc4 = nn.Linear(16*4, 16)  # Bottleneck: 128/2 -> 16
        
        # Decoder Layers
        self.decoder_fc1 = nn.Linear(16, 64)         # Expand bottleneck: 16 -> 128/2
        self.decoder_fc2 = nn.Linear(64, 256)       # Fully connected: 128/2 -> 512/2
        self.decoder_fc3 = nn.Linear(256, 1024)  # Fully connected: 512/2 -> 16384/2
        self.decoder_fc4 = nn.Linear(1024,4096)
        
        self.unflatten = nn.Unflatten(dim=1, unflattened_size=(256, 4, 4))
        
        # Transposed convolution layers
        self.decoder_resid1 = ResidualBlock(256)
        self.decoder_deconv1 = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1) # 4x4 -> 8x8
        self.decoder_deconv2 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1)  # 8x8 -> 16x16
        self.decoder_deconv3 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1) # 16x16 -> 32x32
        self.decoder_resid2 = ResidualBlock(32)
        self.decoder_deconv4 = nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1)  # 32x32 -> 64x64
        self.decoder_deconv5 = nn.ConvTranspose2d(16, 8, kernel_size=3, stride=2, padding=1, output_padding=1)   # 64x64 -> 128x128
        self.decoder_resid0 = ResidualBlock(8)
        self.decoder_deconv6 = nn.ConvTranspose2d(8, 4, kernel_size=3, stride=2, padding=1, output_padding=1)   # 128x128 -> 256x256
        self.decoder_deconv0 = nn.ConvTranspose2d(4, 1, kernel_size=3, stride=1, padding=1, output_padding=0)   # 256x256 -> 256x256

        '''
        # Encoder Layers (Works with 1->16 start, testing with 1->8)
        self.encoder_conv0 = nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1) # 256x256 -> 256x256
        self.encoder_conv1 = nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=1) # 256x256 -> 128x128
        self.encoder_conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1) # 128x128 -> 64x64
        self.encoder_conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1) # 64x64 -> 32x32
        self.encoder_conv4 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1) # 32x32 -> 16x16
        self.encoder_conv5 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1) # 16x16 -> 8x8
        self.encoder_conv6 = nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1) # 8x8 -> 4x4

        self.flatten = nn.Flatten()  # Flatten 4x4x1024/2 -> 16384/2
        
        # Add Nonlinear Dense Layers before the bottleneck
        self.encoder_fc1 = nn.Linear(4*4*512, 256)  # Fully connected: 16384/2 -> 512/2
        self.encoder_fc2 = nn.Linear(256, 64)       # Fully connected: 512/2 -> 128/2
        self.encoder_fc_bottleneck = nn.Linear(64, 16)  # Bottleneck: 128/2 -> 16
        
        # Decoder Layers
        self.decoder_fc1 = nn.Linear(16, 64)         # Expand bottleneck: 16 -> 128/2
        self.decoder_fc2 = nn.Linear(64, 256)       # Fully connected: 128/2 -> 512/2
        self.decoder_fc3 = nn.Linear(256, 4*4*512)  # Fully connected: 512/2 -> 16384/2
        
        self.unflatten = nn.Unflatten(dim=1, unflattened_size=(512, 4, 4))
        
        # Transposed convolution layers
        self.decoder_deconv1 = nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1) # 4x4 -> 8x8
        self.decoder_deconv2 = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1)  # 8x8 -> 16x16
        self.decoder_deconv3 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1) # 16x16 -> 32x32
        self.decoder_deconv4 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1)  # 32x32 -> 64x64
        self.decoder_deconv5 = nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1)   # 64x64 -> 128x128
        self.decoder_deconv6 = nn.ConvTranspose2d(16, 8, kernel_size=3, stride=2, padding=1, output_padding=1)   # 128x128 -> 256x256
        self.decoder_deconv0 = nn.ConvTranspose2d(8, 1, kernel_size=3, stride=1, padding=1, output_padding=0)   # 256x256 -> 256x256
        
        # Encoder Layers
        self.encoder_conv0 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1) # 256x256 -> 256x256
        self.encoder_conv1 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1) # 256x256 -> 128x128
        self.encoder_conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1) # 128x128 -> 64x64
        self.encoder_conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1) # 64x64 -> 32x32
        self.encoder_conv4 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1) # 32x32 -> 16x16
        self.encoder_conv5 = nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1) # 16x16 -> 8x8
        self.encoder_conv6 = nn.Conv2d(512, 1024, kernel_size=3, stride=2, padding=1) # 8x8 -> 4x4

        self.flatten = nn.Flatten()  # Flatten 4x4x1024 -> 16384
        
        # Add Nonlinear Dense Layers before the bottleneck
        self.encoder_fc1 = nn.Linear(4*4*1024, 512)  # Fully connected: 16384 -> 512
        self.encoder_fc2 = nn.Linear(512, 128)       # Fully connected: 512 -> 128
        self.encoder_fc_bottleneck = nn.Linear(128, 16)  # Bottleneck: 128 -> 16
        
        # Decoder Layers
        self.decoder_fc1 = nn.Linear(16, 128)         # Expand bottleneck: 16 -> 128
        self.decoder_fc2 = nn.Linear(128, 512)       # Fully connected: 128 -> 512
        self.decoder_fc3 = nn.Linear(512, 4*4*1024)  # Fully connected: 512 -> 16384
        
        self.unflatten = nn.Unflatten(dim=1, unflattened_size=(1024, 4, 4))
        
        # Transposed convolution layers
        self.decoder_deconv1 = nn.ConvTranspose2d(1024, 512, kernel_size=3, stride=2, padding=1, output_padding=1) # 4x4 -> 8x8
        self.decoder_deconv2 = nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1)  # 8x8 -> 16x16
        self.decoder_deconv3 = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1) # 16x16 -> 32x32
        self.decoder_deconv4 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1)  # 32x32 -> 64x64
        self.decoder_deconv5 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1)   # 64x64 -> 128x128
        self.decoder_deconv6 = nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1)   # 128x128 -> 256x256
        self.decoder_deconv0 = nn.ConvTranspose2d(16, 1, kernel_size=3, stride=1, padding=1, output_padding=0)   # 256x256 -> 256x256
        '''
    def encode(self, x):
        # Forward pass through the encoder
        x = F.leaky_relu(self.encoder_conv0(x))
        x = F.dropout2d(x, p=self.dropout_rate, training=self.training)
        x = F.leaky_relu(self.encoder_conv1(x))
        x = F.leaky_relu(self.encoder_resid0(x))
        x = F.leaky_relu(self.encoder_conv2(x))
        x = F.leaky_relu(self.encoder_conv3(x))
        x = F.leaky_relu(self.encoder_resid1(x))
        x = F.leaky_relu(self.encoder_conv4(x))
        x = F.leaky_relu(self.encoder_conv5(x))
        x = F.leaky_relu(self.encoder_conv6(x))
        x = F.leaky_relu(self.encoder_resid2(x))
        x = self.flatten(x)
        x = F.leaky_relu(self.encoder_fc1(x))  # Nonlinear layer 1
        x = F.dropout(x, p=self.dropout_rate, training=self.training)
        x = F.leaky_relu(self.encoder_fc2(x))  # Nonlinear layer 2
        x = F.leaky_relu(self.encoder_fc3(x))
        x = self.encoder_fc4(x)  # Bottleneck layer (16 values)
        return x.reshape((x.shape[0],4,4))
    
    def decode(self, z):
        # Forward pass through the decoder
        z = z.reshape((z.shape[0],16))
        z = F.leaky_relu(self.decoder_fc1(z))
        z = F.leaky_relu(self.decoder_fc2(z))  # Nonlinear layer 1
        z = F.leaky_relu(self.decoder_fc3(z))  # Nonlinear layer 2
        z = F.dropout(z, p=self.dropout_rate, training=self.training)
        z = F.leaky_relu(self.decoder_fc4(z))  # Expand to match convolution input
        z = self.unflatten(z)
        z = F.leaky_relu(self.decoder_resid1(z))
        z = F.leaky_relu(self.decoder_deconv1(z))
        z = F.leaky_relu(self.decoder_deconv2(z))
        z = F.leaky_relu(self.decoder_deconv3(z))
        z = F.leaky_relu(self.decoder_resid2(z))
        z = F.leaky_relu(self.decoder_deconv4(z))
        z = F.leaky_relu(self.decoder_deconv5(z))
        z = F.leaky_relu(self.decoder_resid0(z))
        z = F.leaky_relu(self.decoder_deconv6(z))
        z = F.dropout2d(z, p=self.dropout_rate, training=self.training)
        z = torch.sigmoid(self.decoder_deconv0(z))  # Sigmoid to constrain output to [0, 1]
        return z
    
    def forward(self, x):
        # Combined encode-decode forward pass
        z = self.encode(x)  # Latent representation
        x_reconstructed = self.decode(z)  # Reconstructed image
        return x_reconstructed

class NewAutoencoder(nn.Module):
    def __init__(self, base_channels=32):
        super(NewAutoencoder, self).__init__()

        kernel_size = 3
        stride = 2
        padding = 1
        output_padding = 1
        
        self.enc1 = nn.Sequential(
            nn.Conv2d(1, base_channels, kernel_size=kernel_size, stride=stride, padding=padding),  # 256x256 -> 128x128
            nn.BatchNorm2d(base_channels),
            nn.ReLU(),
            ResidualBlock(base_channels),
        )

        self.enc2 = nn.Sequential(
            nn.Conv2d(base_channels, base_channels * 2, kernel_size=kernel_size, stride=stride, padding=padding),  # 128x128 -> 64x64
            nn.BatchNorm2d(base_channels * 2),
            nn.ReLU(),
            ResidualBlock(base_channels * 2),
        )

        self.enc3 = nn.Sequential(
            nn.Conv2d(base_channels * 2, base_channels * 4, kernel_size=kernel_size, stride=stride, padding=padding),  # 64x64 -> 32x32
            nn.BatchNorm2d(base_channels * 4),
            nn.ReLU(),
            ResidualBlock(base_channels * 4),
        )

        self.enc4 = nn.Sequential(
            nn.Conv2d(base_channels * 4, base_channels * 8, kernel_size=kernel_size, stride=stride, padding=padding),  # 32x32 -> 16x16
            nn.BatchNorm2d(base_channels * 8),
            nn.ReLU(),
            ResidualBlock(base_channels * 8),
        )

        self.enc5 = nn.Sequential(
            nn.Conv2d(base_channels * 8, base_channels * 16, kernel_size=kernel_size, stride=stride, padding=padding),  # 16x16 -> 8x8
            nn.BatchNorm2d(base_channels * 16),
            nn.ReLU(),
            ResidualBlock(base_channels * 16),
        )

        self.enc6 = nn.Sequential(
            nn.Conv2d(base_channels * 16, base_channels * 32, kernel_size=kernel_size, stride=stride, padding=padding),  # 8x8 -> 4x4
            nn.BatchNorm2d(base_channels * 32),
            nn.ReLU(),
            ResidualBlock(base_channels * 32),
        )

        # Final compression to single channel
        self.compress = nn.Conv2d(base_channels * 32, 1, kernel_size=1)

        # Decoder layers
        self.dec6 = nn.Sequential(
            nn.ConvTranspose2d(1, base_channels * 16, kernel_size=kernel_size, stride=stride, padding=padding, output_padding=output_padding),  # 4x4 -> 8x8
            nn.BatchNorm2d(base_channels * 16),
            nn.ReLU(),
            ResidualBlock(base_channels * 16),
        )

        self.dec5 = nn.Sequential(
            nn.ConvTranspose2d(base_channels * 16, base_channels * 8, kernel_size=kernel_size, stride=stride, padding=padding, output_padding=output_padding),  # 8x8 -> 16x16
            nn.BatchNorm2d(base_channels * 8),
            nn.ReLU(),
            ResidualBlock(base_channels * 8),
        )

        self.dec4 = nn.Sequential(
            nn.ConvTranspose2d(base_channels * 8, base_channels * 4, kernel_size=kernel_size, stride=stride, padding=padding, output_padding=output_padding),  # 16x16 -> 32x32
            nn.BatchNorm2d(base_channels * 4),
            nn.ReLU(),
            ResidualBlock(base_channels * 4),
        )

        self.dec3 = nn.Sequential(
            nn.ConvTranspose2d(base_channels * 4, base_channels * 2, kernel_size=kernel_size, stride=stride, padding=padding, output_padding=output_padding),  # 32x32 -> 64x64
            nn.BatchNorm2d(base_channels * 2),
            nn.ReLU(),
            ResidualBlock(base_channels * 2),
        )

        self.dec2 = nn.Sequential(
            nn.ConvTranspose2d(base_channels * 2, base_channels, kernel_size=kernel_size, stride=stride, padding=padding, output_padding=output_padding),  # 64x64 -> 128x128
            nn.BatchNorm2d(base_channels),
            nn.ReLU(),
            ResidualBlock(base_channels),
        )

        self.dec1 = nn.Sequential(
            nn.ConvTranspose2d(base_channels, 1, kernel_size=kernel_size, stride=stride, padding=padding, output_padding=output_padding),  # 128x128 -> 256x256
            nn.BatchNorm2d(1),
            nn.ReLU(),
            ResidualBlock(1),
            nn.Conv2d(1, 1, kernel_size=3, padding=1),
            nn.Sigmoid(),
        )
        
    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        e5 = self.enc5(e4)
        e6 = self.enc6(e5)
        # Compression
        compressed = self.compress(e6)

        # Decoder
        d6 = self.dec6(compressed)
        d5 = self.dec5(d6)
        d4 = self.dec4(d5)
        d3 = self.dec3(d4)
        d2 = self.dec2(d3)
        d1 = self.dec1(d2)

        return d1

    def encode(self, x):
        """Returns the encoded image as a single 4x4 grayscale image"""
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        e5 = self.enc5(e4)
        e6 = self.enc6(e5)
        compressed = self.compress(e6)
        return compressed.squeeze(0)
    
    def decode(self, compressed):
        """Decodes the 4x4 grayscale compressed representation back to a 256x256 image"""
        d6 = self.dec6(compressed.unsqueeze(0))
        d5 = self.dec5(d6)
        d4 = self.dec4(d5)
        d3 = self.dec3(d4)
        d2 = self.dec2(d3)
        d1 = self.dec1(d2)
        return d1

class Autoencoder(nn.Module):
    def __init__(self, base_channels=32):
        super(Autoencoder, self).__init__()
        
        # Encoder layers
        self.enc1 = nn.Sequential(
            nn.Conv2d(1, base_channels, kernel_size=7, stride=4, padding=2),  # 256x256 -> 64x64
            nn.BatchNorm2d(base_channels),
            nn.ReLU(),
        )
        
        self.enc2 = nn.Sequential(
            nn.Conv2d(base_channels, base_channels, kernel_size=7, stride=4, padding=2),  # 64x64 -> 16x16
            nn.BatchNorm2d(base_channels),
            nn.ReLU(),
        )
        
        self.enc3 = nn.Sequential(
            nn.Conv2d(base_channels, base_channels, kernel_size=7, stride=4, padding=2),  # 16x16 -> 4x4
            nn.BatchNorm2d(base_channels),
            nn.ReLU(),
        )
        
        # Final compression to single channel
        self.compress = nn.Conv2d(base_channels, 1, kernel_size=1)
        
        # Decoder layers
        self.dec3 = nn.Sequential(
            nn.ConvTranspose2d(1, base_channels, kernel_size=8, stride=4, padding=2),  # 4x4 -> 16x16
            nn.BatchNorm2d(base_channels),
            nn.ReLU(),
        )
        
        self.dec2 = nn.Sequential(
            nn.ConvTranspose2d(base_channels, base_channels, kernel_size=8, stride=4, padding=2),  # 16x16 -> 64x64
            nn.BatchNorm2d(base_channels),
            nn.ReLU(),
        )
        
        self.dec1 = nn.Sequential(
            nn.ConvTranspose2d(base_channels, base_channels, kernel_size=8, stride=4, padding=2),  # 64x64 -> 256x256
            nn.BatchNorm2d(base_channels),
            nn.ReLU(),
            nn.Conv2d(base_channels, 1, kernel_size=3, padding=1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        
        # Compress to 4x4x1
        compressed = self.compress(e3)
        
        # Decoder
        d3 = self.dec3(compressed)
        d2 = self.dec2(d3)
        d1 = self.dec1(d2)
        
        return d1
    
    def encode(self, x):
        """Returns the encoded image as a single 4x4 grayscale image"""
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        compressed = self.compress(e3)
        return compressed.squeeze(0)
    
    def decode(self, compressed):
        """Decodes the 4x4 grayscale compressed representation back to a 256x256 image"""
        d3 = self.dec3(compressed)
        d2 = self.dec2(d3)
        d1 = self.dec1(d2)
        return d1
    
class old_Autoencoder(nn.Module):
    def __init__(self):
        super(old_Autoencoder, self).__init__()
        
        # Encoder - reduce from 256x256 to 4x4 with single channel
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=4, stride=4, padding=0),  # 64x64
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=4, padding=0),  # 16x16
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=4, padding=0),  # 4x4
            nn.ReLU(),
            # Additional convolution to reduce channels to 1, no activation
            nn.Conv2d(128, 1, kernel_size=1, stride=1, padding=0),  # 4x4x1
        )
        
        self.decoder = nn.Sequential(
            # First expand channels
            nn.Conv2d(1, 128, kernel_size=1, stride=1, padding=0),  # 4x4x128
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=4, padding=0),  # 16x16
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=4, padding=0),  # 64x64
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, kernel_size=4, stride=4, padding=0),  # 256x256
            nn.Sigmoid()  # Keep sigmoid here for final image output (assuming images should be 0-1)
        )
        
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
    
    def encode(self, x):
        return self.encoder(x)

def get_model_for_inference(model):
    """
    Helper function to get the base model from DataParallel if needed
    """
    if isinstance(model, DataParallel):
        return model.module
    return model

def train_model(
    model: ThirdAutoencoder,
    list_images,
    #train_dataloaders: list[DataLoader],
    #val_dataloader: DataLoader,
    num_epochs=100,
    learning_rate=1e-3,
    num_workers=4
    ):

    def custom_lr_schedule(epoch):
        if epoch < 200:
            return 1.0  # Keep the LR the same
        else:
            return 0.75 ** ((epoch - 200) // 100 + 1)  # Every 100 epochs (each dataset)

    def validate_model():
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for val_batch in val_dataloader:
                val_batch = val_batch.to(device)
                val_output = model(val_batch)
                val_loss = criterion(val_output, val_batch)
                total_val_loss += val_loss.item()
        return total_val_loss / len(val_dataloader)

    val_dataset = BinaryImageDataset(list_images[0])
    val_dataloader = DataLoader(
            val_dataset, 
            batch_size=256, 
            #shuffle=True,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=False
        )

    print("Training Started")
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0)
    #scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=10, cooldown=10)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=80, gamma=0.8)
    #scheduler = LambdaLR(optimizer, lr_lambda=custom_lr_schedule)

    # Enable multiprocessing for data loading
    if torch.get_num_threads() != num_workers:
        torch.set_num_threads(num_workers)

    train_losses = []
    val_losses = []
    num_datasets = 9
    best_val_loss = float('inf')



    for epoch in range(num_epochs):
        # Select the appropriate dataset for this epoch
        if (epoch == 0) or (epoch % 80 == 0):
            current_dataset = BinaryImageDataset(list_images[(int(epoch/80) % num_datasets) + 1])
            current_loader = DataLoader(
                current_dataset, 
                batch_size=256, 
                #shuffle=True,
                shuffle=True,
                num_workers=num_workers,
                pin_memory=False
                )
        print(f"\nEpoch {epoch + 1}/{num_epochs} using training dataset {(int(epoch/80) % num_datasets) + 1}. LR: {scheduler.get_last_lr()[0]}")
        
        # Training phase
        model.train()
        total_train_loss = 0
        
        for batch in current_loader:
            batch = batch.to(device)
            
            # Forward pass
            output = model(batch)
            loss = criterion(output, batch)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_train_loss += loss.item()
        
        avg_train_loss = total_train_loss / len(current_loader)
        train_losses.append(avg_train_loss)
        
        # Validation phase
        val_loss = validate_model()
        val_losses.append(val_loss)
        
        print(f"Epoch {epoch + 1} - Training Loss: {avg_train_loss:.6f}, Validation Loss: {val_loss:.6f}")
        
        # Learning rate scheduling based on validation loss
        #scheduler.step(val_loss)
        #scheduler.step(avg_train_loss)
        scheduler.step()

    return train_losses, val_losses

def train_autoencoder(model, train_loader, num_epochs=100, learning_rate=1e-3, num_workers=4):
    print("Training Started")
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=5)
    
    # Enable multiprocessing for data loading
    if torch.get_num_threads() != num_workers:
        torch.set_num_threads(num_workers)
    
    losses = []
    
    for epoch in range(num_epochs):
        total_loss = 0
        for batch in train_loader:
            batch = batch.to(device)
            # Forward pass
            output = model(batch)
            loss = criterion(output, batch)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        avg_loss = total_loss / len(train_loader)
        losses.append(avg_loss)
        scheduler.step(avg_loss)
        
        if (epoch + 1) % 1 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Average Loss: {avg_loss:.6f}, Learning Rate: {scheduler.get_last_lr()[0]}')
    
    return losses

def visualize_results(model, test_images, num_examples=5, save_path=None):
    """
    Visualize original, encoded, and reconstructed images side by side.
    """
    # Get base model for inference
    model = get_model_for_inference(model)
    model.eval()
    
    with torch.no_grad():
        # Convert images to tensor
        test_tensor = torch.FloatTensor(test_images).unsqueeze(1)
        
        # Get reconstructions
        reconstructed = model(test_tensor)
        
        # Get encoded (compressed) representations
        encoded = model.encode(test_tensor)
        
        # Convert to numpy for plotting
        reconstructed = reconstructed.numpy()
        encoded = encoded.numpy()
        #encoded = encoded.reshape((encoded.shape[0],4,4))
        
        # Create figure with enough space for titles
        fig, axes = plt.subplots(3, num_examples, figsize=(15, 10))
        fig.suptitle('Autoencoder Results Comparison', fontsize=16, y=0.95)
        
        # Add row titles
        row_titles = ['Original Images (256x256)', 
                     'Encoded Representations (16x16)', 
                     'Reconstructed Images (256x256)']
        
        for i, title in enumerate(row_titles):
            fig.text(0.02, 0.75 - i*0.3, title, fontsize=12, rotation=90)
        
        for i in range(num_examples):
            # Original
            im0 = axes[0, i].imshow(test_images[i].squeeze(), cmap='binary')
            axes[0, i].axis('off')
            if i == 0:
                fig.colorbar(im0, ax=axes[0, i], label='Binary')
            
            # Encoded (16x16)
            encoded_img = encoded[i]#.mean(axis=0)
            im1 = axes[1, i].imshow(encoded_img, cmap='viridis')
            axes[1, i].axis('off')
            if i == 0:
                fig.colorbar(im1, ax=axes[1, i], label='Activation')
            
            # Reconstructed
            im2 = axes[2, i].imshow(reconstructed[i].squeeze(), cmap='binary')
            axes[2, i].axis('off')
            if i == 0:
                fig.colorbar(im2, ax=axes[2, i], label='Phase Probability')
        
        plt.tight_layout(rect=[0.05, 0, 1, 0.95])
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
            plt.close()
        else:
            plt.show()

def visualize_single_image(model, image, save_path=None):
    """
    Visualize the compression pipeline for a single image with detailed metrics.
    """
    # Get base model for inference
    model = get_model_for_inference(model)
    model.eval()
    
    with torch.no_grad():
        # Prepare image
        image_tensor = torch.FloatTensor(image).unsqueeze(0).unsqueeze(0)
        
        # Get encoded and reconstructed versions
        encoded = model.encode(image_tensor)
        reconstructed = model(image_tensor)
        
        # Convert to numpy
        encoded = encoded.numpy()
        reconstructed = reconstructed.numpy()
        
        # Calculate metrics
        mse = np.mean((image - reconstructed[0, 0])**2)
        binary_accuracy = np.mean((reconstructed[0, 0].round() == image))
        
        # Create figure
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle('Single Image Compression Analysis', fontsize=14)
        
        # Original
        axes[0].imshow(image, cmap='viridis')
        axes[0].set_title('Original (256x256)')
        axes[0].axis('off')
        
        # Encoded
        encoded_img = encoded[0]
        im = axes[1].imshow(encoded_img, cmap='viridis')
        axes[1].set_title('Encoded (16x16)')
        axes[1].axis('off')
        #fig.colorbar(im, ax=axes[1], label='Mean Activation')
        
        # Reconstructed
        axes[2].imshow(reconstructed[0, 0], cmap='viridis')
        axes[2].set_title('Reconstructed (256x256)')
        axes[2].axis('off')
        
        # Add metrics as text
        #plt.figtext(0.02, 0.02, f'MSE: {mse:.4f}', fontsize=10)
        #plt.figtext(0.25, 0.02, f'Binary Accuracy: {binary_accuracy:.2%}', fontsize=10)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
            plt.close()
        else:
            plt.show()

def plot_training_loss(train_losses, val_losses):
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Training Loss Over Time')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    #plt.yscale('log')
    plt.grid(True)
    plt.savefig('new_traininglossimg.png')
    plt.show()

def create_model(list_images, num_channels=2, batch_size=32, num_epochs=100, num_workers=4, learning_rate=1e-3):
    # Set number of threads for PyTorch
     # Set number of threads for PyTorch
    torch.set_num_threads(num_workers)

    legacy = False

    if legacy:
        train_loaders = []
        for i in range(len(list_images)-1):
            dataset = BinaryImageDataset(list_images[i])
            train_loader = DataLoader(
            dataset, 
            batch_size=batch_size, 
            shuffle=True,
            num_workers=num_workers,
            pin_memory=False
            )
            train_loaders.append(train_loader)
        val_dataset = BinaryImageDataset(list_images[len(list_images)-1])
        val_loader = DataLoader(
            val_dataset, 
            batch_size=batch_size, 
            shuffle=True,
            num_workers=num_workers,
            pin_memory=False
        )

    # Initialize model
    base_model = ThirdAutoencoder(base_channels=num_channels)
    
    # Optionally use DataParallel if multiple CPU cores are available
    if num_workers > 1:
        model = DataParallel(base_model)
    else:
        model = base_model

    # Train the model
    train_losses, val_losses = train_model(
        model.to(device),
        list_images,
        #train_loaders,
        #val_loader,
        num_epochs=num_epochs,
        num_workers=num_workers,
        learning_rate=learning_rate
    )
    
    return model, train_losses, val_losses
    
    '''
    # Create dataset and dataloader
    dataset = BinaryImageDataset(images)
    train_loader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=num_workers,
        pin_memory=False
    )
    
    # Initialize model
    base_model = ThirdAutoencoder(base_channels=num_channels)
    
    # Optionally use DataParallel if multiple CPU cores are available
    if num_workers > 1:
        model = DataParallel(base_model)
    else:
        model = base_model

    # Train the model
    losses = train_autoencoder(
        model.to(device), 
        train_loader, 
        num_epochs=num_epochs, 
        num_workers=num_workers,
        learning_rate=learning_rate
    )
    
    return model, losses
    '''

def get_compressed_representation(model, images, batch_size=32, num_workers=4):
    """Get 4x4 compressed representations for new images"""
    # Get base model for inference
    model = get_model_for_inference(model)
    model.eval()

    compressed_list = np.empty((images.shape[0],4,4))

    i=0
    for image in images:
        with torch.no_grad():
            # Prepare image
            image_tensor = torch.FloatTensor(image).unsqueeze(0).unsqueeze(0)
        
            # Get encoded versions
            encoded = model.encode(image_tensor)
            encoded = encoded.numpy()
            compressed_list[i] = encoded[0]
            i+=1

    print(compressed_list)

    return compressed_list

    '''
    with torch.no_grad():
        dataset = BinaryImageDataset(images)
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=False
        )
        
        compressed_list = []
        for batch in loader:
            compressed = model.encode(batch)
            compressed_list.append(compressed.numpy())
    '''
    

def custom_sort_key(filepath):
        # Extract the base name (without the directory or extension)
        filename = filepath.rsplit('.', 1)[0]
        
        # Use regex to separate the alphabetic part and the numeric part at the end
        match = re.match(r"(.+)_([0-9]+)$", filename)
        if match:
            alpha_part = match.group(1)  # Everything before the final underscore
            num_part = int(match.group(2))  # The numeric part after the underscore
            return (alpha_part, num_part)
        return (filename, 0)  # Fallback if no match, unlikely with this format


if __name__ == '__main__':

    print(torch.__file__)
    print(torch.__version__)
    print(torch.cuda.is_available())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_workers = min(8, os.cpu_count())
    #num_workers = 1
    db_folder = 'workdir.5/csv_files/train_npy_files_thr'
    num_images = 8000
    num_subarray = 10
    train_images = int(num_images * 1)
    x_dim = 256
    y_dim = 256
    load = True
    train = True
    encode = False


    if load == True:
        name_list = glob.glob(db_folder + "/*out.*0000.npy")
        names_sorted_npy = sorted(name_list, key=custom_sort_key)
        print(len(names_sorted_npy))
        random.shuffle(names_sorted_npy)

        #all_images= np.empty((num_images, x_dim, y_dim))
        images = np.empty((int(num_images/num_subarray), x_dim, y_dim))
        list_images = []
        for i in range(num_subarray):
            for j in range(int(num_images/num_subarray)):
                images[j] = np.load(names_sorted_npy[int(i*num_images/num_subarray + j)])
                print(f"Loaded Image Number {int(i*num_images/num_subarray + j + 1)} from {names_sorted_npy[int(i*num_images/num_subarray + j)]}")
            list_images.append(images)
        
        #images = all_images[:train_images]
        print(f"{len(list_images)} subarrays with {list_images[0].shape} shape")

    if train == True:
        model, train_losses, val_losses = create_model(list_images, batch_size=1024, num_epochs=18*80, num_workers=num_workers, learning_rate=0.002)
        model = model.to("cpu")
        torch.save(model.state_dict(), 'new_ost_model.pt')
        plot_training_loss(train_losses, val_losses)
        visualize_results(model, list_images[0][0:5])
        #visualize_single_image(model,all_images[4])
        input("Done Training, quit now")
    
    if encode == True:
        device=torch.device('cpu')
        load_model = ThirdAutoencoder()
        if num_workers > 1:
            load_model = DataParallel(load_model)
        load_model.load_state_dict(torch.load("ost_model.pt", map_location=torch.device('cpu'), weights_only=True))
        load_model.eval()

        # To get compressed representations for new images:
        load_model=load_model.to('cpu')
        #all_images=all_images.to('cpu')
        compressed_images = get_compressed_representation(load_model, all_images, num_workers=num_workers)
        print(compressed_images.shape)
        np.save('ost_AE.npy',compressed_images)
        visualize_results(load_model, images[0:5], num_examples=5)
    
    device=torch.device('cpu')
    load_model = ThirdAutoencoder()
    if num_workers > 1:
        load_model = DataParallel(load_model)
    load_model.load_state_dict(torch.load("new_ost_model.pt", map_location=torch.device('cpu'), weights_only=True))
    load_model.eval()
    model = get_model_for_inference(load_model)
    model.eval()
    model.to('cpu')

    
    
    name_list = glob.glob(db_folder + "/*out.*.npy")
    names_sorted_npy = sorted(name_list, key=custom_sort_key)
    n_images = len(names_sorted_npy)                                        #Sim 1 stabilizes at about 50
    #while True:                                                             #check 15, 21, 25, 33, 55
    #n = int(input("Choose Simulation: "))
    n=121 #Good 55, 21?, 15?
    t=100
    #visualize_single_image(model,np.load(names_sorted_npy[505*n+101*0+t])) #[505*30+101*0+2], [505*2+101*0+1], [505*7+101*0+45]
    #visualize_single_image(model,np.load(names_sorted_npy[505*n+101*1+t]))
    #visualize_single_image(model,np.load(names_sorted_npy[505*n+101*2+t]))
    #visualize_single_image(model,np.load(names_sorted_npy[505*n+101*3+t]))
    #visualize_single_image(model,np.load(names_sorted_npy[505*n+101*4+t]))
    #visualize_single_image(model,np.load(names_sorted_npy[505*n+101*0+65]))
    #visualize_single_image(model,np.load(names_sorted_npy[505*n+101*0+75]))
    #visualize_single_image(model,np.load(names_sorted_npy[505*n+101*0+85]))
    #visualize_single_image(model,np.load(names_sorted_npy[505*n+101*0+95]))
    #input('Quit Now')
    '''
    compressed_list = np.empty((n_images,4,4))
    input(compressed_list.shape)

    for i in range(n_images):
        with torch.no_grad():
            # Prepare image
            image = np.load(names_sorted_npy[i])
            image_tensor = torch.FloatTensor(image).unsqueeze(0).unsqueeze(0)
        
            # Get encoded versions
            encoded = model.encode(image_tensor)
            encoded = encoded.numpy()
            compressed_list[i] = encoded[0]
            print(f'Done Converting Image {names_sorted_npy[i]}')
    np.save('ost_AE.npy',compressed_list)
    input("Done Converting...")
    '''
    
    device=torch.device('cpu')
    load_model = ThirdAutoencoder()
    if num_workers > 1:
        load_model = DataParallel(load_model)
    load_model.load_state_dict(torch.load("new_ost_model.pt", map_location=torch.device('cpu'), weights_only=True))
    load_model.eval()
    load_model=load_model.to('cpu')
    model = get_model_for_inference(load_model)
    model.eval()
    
    #Section for figure generation
    timestep = 95
    frame = 300000*timestep
    raw_name = glob.glob(f"target/0007_out.ostwald_3_{frame}.npy")[0]
    original_img = np.load(raw_name)


    encoded_arr = np.load("predicted_images.npy")
    #encoded_arr = np.load("pred_eta4.npy")

    original_tensor = torch.FloatTensor(original_img).unsqueeze(0).unsqueeze(0)
    
    #encoded_tensor = model.encode(original_tensor)
    #reconstruct_tensor = model.decode(encoded_tensor)
    #encoded_img = encoded_tensor.detach().numpy()
    
    encoded_img = encoded_arr[timestep-50]
    encoded_tensor = torch.FloatTensor(encoded_img).unsqueeze(0).unsqueeze(0)


    reconstruct_tensor = model.decode(encoded_tensor)

    reconstruct_img = reconstruct_tensor.detach().numpy()


    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle('Autoencoder Results Comparison', fontsize=16, y=0.95)

    # Original
    im0 = axes[0].imshow(original_img.squeeze(), cmap='viridis')
    axes[0].axis('off')
    fig.colorbar(im0, ax=axes[0], label='Binary')
    
    # Encoded (4x4)
    im1 = axes[1].imshow(encoded_img.squeeze(), cmap='viridis')
    axes[1].axis('off')
    fig.colorbar(im1, ax=axes[1], label='Activation')
    
    # Reconstructed
    im2 = axes[2].imshow(reconstruct_img.squeeze(), cmap='viridis')
    axes[2].axis('off')
    fig.colorbar(im2, ax=axes[2], label='Phase Probability')

    plt.tight_layout()
    plt.show()

    '''
    #Finished

    name_list = glob.glob(db_folder + "/*out.*.npy")
    names_sorted_npy = sorted(name_list, key=custom_sort_key)

    original_arr = np.empty((5, x_dim, y_dim))
    restored_arr = np.empty((5, x_dim, y_dim))
    start = 1*505+2
    end = start + 5
    for i in range(start,end):
        original_arr[i-start] = np.load(names_sorted_npy[i])

    compressed_arr = np.load('ost_AE.npy')[start:end]

    device=torch.device('cpu')
    load_model = ThirdAutoencoder()
    if num_workers > 1:
        load_model = DataParallel(load_model)
    load_model.load_state_dict(torch.load("current_ost_model.pt", map_location=torch.device('cpu'), weights_only=True))
    load_model.eval()
    load_model=load_model.to('cpu')
    model = get_model_for_inference(load_model)
    model.eval()

    print(f"Compressed Shape {compressed_arr.shape}")
    for i in range(5):
        compressed_tensor = torch.FloatTensor(np.expand_dims(compressed_arr[i], axis=0))
        #compressed_tensor = model.encode(torch.FloatTensor(original_arr[i]).unsqueeze(0).unsqueeze(0))
        restored_tensor = model.decode(compressed_tensor)
        restored_arr[i] = restored_tensor.detach().numpy()
    print(f"Restored Shape {restored_arr.shape}")

    fig, axes = plt.subplots(3, 5, figsize=(15, 10))
    fig.suptitle('Autoencoder Results Comparison', fontsize=16, y=0.95)
    
    # Add row titles
    row_titles = ['Original Images (256x256)', 
                    'Encoded Representations (4x4)', 
                    'Reconstructed Images (256x256)']

    for i in range(5):
            # Original
            im0 = axes[0, i].imshow(original_arr[i].squeeze(), cmap='binary')
            axes[0, i].axis('off')
            if i == 0:
                fig.colorbar(im0, ax=axes[0, i], label='Binary')
            
            # Encoded (16x16)
            encoded_img = compressed_arr[i]
            im1 = axes[1, i].imshow(encoded_img, cmap='viridis')
            axes[1, i].axis('off')
            if i == 0:
                fig.colorbar(im1, ax=axes[1, i], label='Activation')
            
            # Reconstructed
            print(restored_arr[i].squeeze())
            im2 = axes[2, i].imshow(restored_arr[i].squeeze(), cmap='binary')
            axes[2, i].axis('off')
            if i == 0:
                fig.colorbar(im2, ax=axes[2, i], label='Probability')

    plt.tight_layout()
    plt.show()
    '''