# -*- coding: utf-8 -*-

from collections import OrderedDict

import torch
import torch.nn as nn


# https://arxiv.org/pdf/1606.06650.pdf
# https://github.com/wolny/pytorch-3dunet/blob/master/pytorch3dunet/unet3d/model.py

class UNet3D(nn.Module):
    # acdc 112x112x112x3
    def __init__(self, in_channels=32, out_channels=1, init_features=30):
        super(UNet3D, self).__init__()
        features = init_features
        self.encoder1 = UNet3D._block(in_channels, features, name="enc1")
        self.pool1 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.encoder2 = UNet3D._block(features, features * 2, name="enc2")
        self.pool2 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.encoder3 = UNet3D._block(features * 2, features * 4, name="enc3")
        self.pool3 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.encoder4 = UNet3D._block(features * 4, features * 8, name="enc4")
        self.pool4 = nn.MaxPool3d(kernel_size=2, stride=2)

        self.bottleneck = UNet3D._block(features * 8, features * 16, name="bottleneck")

        self.upconv4 = nn.ConvTranspose3d(
            features * 16, features * 8, kernel_size=2, stride=2
        )
        self.decoder4 = UNet3D._block((features * 8) * 2, features * 8, name="dec4")
        self.upconv3 = nn.ConvTranspose3d(
            features * 8, features * 4, kernel_size=2, stride=2
        )
        self.decoder3 = UNet3D._block((features * 4) * 2, features * 4, name="dec3")
        self.upconv2 = nn.ConvTranspose3d(
            features * 4, features * 2, kernel_size=2, stride=2
        )
        self.decoder2 = UNet3D._block((features * 2) * 2, features * 2, name="dec2")
        self.upconv1 = nn.ConvTranspose3d(
            features * 2, features, kernel_size=2, stride=2
        )
        self.decoder1 = UNet3D._block(features * 2, features, name="dec1")

        self.conv = nn.Conv3d(
            in_channels=features, out_channels=out_channels, kernel_size=1
        )

    
    def forward(self, x):
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))
        enc4 = self.encoder4(self.pool3(enc3))

        bottleneck = self.bottleneck(self.pool4(enc4))

        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.decoder4(dec4)
        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)
        
        return self.conv(dec1)

    @staticmethod
    def _block(in_channels, features, name):
        return nn.Sequential(
            OrderedDict(
                [
                    (
                        name + "conv1",
                        nn.Conv3d(
                            in_channels=in_channels,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            stride=1,
                            bias=False,
                        ),
                    ),
                    (name + "norm1", nn.InstanceNorm3d(num_features=features)),
                    (name + "relu1", nn.LeakyReLU(negative_slope=0.01, inplace=True)),
                    # (name + "norm1", nn.BatchNorm2d(num_features=features)),
                    # (name + "relu1", nn.ReLU(inplace=True)),
                    (
                        name + "conv2",
                        nn.Conv3d(
                            in_channels=features,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            stride=1,
                            bias=False,
                        ),
                    ),
                    (name + "norm2", nn.InstanceNorm3d(num_features=features)),
                    (name + "relu2", nn.LeakyReLU(negative_slope=0.01, inplace=True)),
                    # (name + "norm2", nn.BatchNorm2d(num_features=features)),
                    # (name + "relu2", nn.ReLU(inplace=True)),
                ]
            )
        )

class UNet3D_ef(nn.Module):
    # acdc 3x32x112x112
    def __init__(self, in_channels=3, out_channels=1, init_features=30, pretrain = None):
        super(UNet3D_ef, self).__init__()
        features = init_features
        
        fin = [in_channels, features, features * 2, features * 4, features * 8]
        fout = [features, features * 2, features * 4, features * 8,features * 16]
        self.features = nn.Sequential(
            # encoder 1
            nn.Conv3d(in_channels=fin[0],out_channels=fout[0],kernel_size=3,padding=1,stride=1,bias=False),
            nn.InstanceNorm3d(num_features=fout[0]),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.Conv3d(in_channels=fout[0],out_channels=fout[1],kernel_size=3,padding=1,stride=1,bias=False),
            nn.InstanceNorm3d(num_features=fout[0]),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.MaxPool3d(kernel_size=2, stride=2, return_indices=True),
            #encoder 2
            nn.Conv3d(in_channels=fin[1],out_channels=fout[1],kernel_size=3,padding=1,stride=1,bias=False),
            nn.InstanceNorm3d(num_features=fout[1]),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.Conv3d(in_channels=fout[1],out_channels=fout[1],kernel_size=3,padding=1,stride=1,bias=False),
            nn.InstanceNorm3d(num_features=fout[1]),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.MaxPool3d(kernel_size=2, stride=2, return_indices=True),
            #encoder 3
            nn.Conv3d(in_channels=fin[2],out_channels=fout[2],kernel_size=3,padding=1,stride=1,bias=False),
            nn.InstanceNorm3d(num_features=fout[2]),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.Conv3d(in_channels=fout[2],out_channels=fout[2],kernel_size=3,padding=1,stride=1,bias=False),
            nn.InstanceNorm3d(num_features=fout[2]),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.MaxPool3d(kernel_size=2, stride=2, return_indices=True),
            #encoder 4
            nn.Conv3d(in_channels=fin[3],out_channels=fout[3],kernel_size=3,padding=1,stride=1,bias=False),
            nn.InstanceNorm3d(num_features=fout[3]),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.Conv3d(in_channels=fout[3],out_channels=fout[3],kernel_size=3,padding=1,stride=1,bias=False),
            nn.InstanceNorm3d(num_features=fout[3]),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.MaxPool3d(kernel_size=2, stride=2, return_indices=True),
            #encoder 5
            nn.Conv3d(in_channels=fin[4],out_channels=fout[4],kernel_size=3,padding=1,stride=1,bias=False),
            nn.InstanceNorm3d(num_features=fout[4]),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.Conv3d(in_channels=fout[4],out_channels=fout[4],kernel_size=3,padding=1,stride=1,bias=False),
            nn.InstanceNorm3d(num_features=fout[4]),
            nn.LeakyReLU(negative_slope=0.01, inplace=True)
            )
        
        self.fc = nn.Sequential(nn.Linear(480*2*7*7, 60*2*7*7),
                                 nn.ReLU(),
                                 nn.Linear(60*2*7*7, 1)
                                ) 
        # index of conv
        self.conv_layer_indices = [0, 3, 7, 10, 14, 17, 21, 24, 28, 31]
        # feature maps
        self.feature_maps = OrderedDict()
        # switch
        self.pool_locs = OrderedDict()
        self.ini_weights()
        # for idx, layer in enumerate(self.features):
        #     if isinstance(layer, nn.Conv3d):
        #         print(idx,layer)
        
    def forward(self, x):
        for idx, layer in enumerate(self.features):
            if isinstance(layer, nn.MaxPool3d):
                x, location = layer(x)
                # print(location)
                # self.pool_locs[idx] = location
            else:
                x = layer(x)
        x = x.view(x.size(0),-1)
        output = self.fc(x)
        return output
    
    def ini_weights(self):
        checkpoint = torch.load("/home/zdadadaz/Desktop/course/medical/code/echodyn/output/video/echonet_ef/best.pt")
        # model.load_state_dict(checkpoint['state_dict'])
        # Print model's state_dict
        # print("Model's state_dict:")
        count = 0
        for idx, layer in enumerate(checkpoint['state_dict']):
            # print(idx, layer, checkpoint['state_dict'][layer].size())
            if idx<10:
                self.features[self.conv_layer_indices[count]].weight.data = checkpoint['state_dict'][layer]
                count += 1
            elif idx == 10:
                self.fc[0].weight.data = checkpoint['state_dict'][layer]
            elif idx == 11:
                self.fc[0].bias.data = checkpoint['state_dict'][layer]
            elif idx == 12:
                self.fc[2].weight.data = checkpoint['state_dict'][layer]
            elif idx == 13:
                self.fc[2].bias.data = checkpoint['state_dict'][layer]
        
# model = UNet3D_ef(3, 1)
# print(model)
