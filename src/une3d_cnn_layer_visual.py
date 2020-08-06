# -*- coding: utf-8 -*-

import os
import numpy as np
import cv2
import torch
import torch.nn as nn
from torch.optim import Adam
from torchvision import models
from torch.autograd import Variable
from misc_functions import preprocess_image, recreate_image, save_image,loadvideo
from unet3d import UNet3D_ef

class CNNLayerVisualization():
    """
        Produces an image that minimizes the loss of a convolution
        operation for a specific layer and filter
    """
    def __init__(self, model, selected_layer, selected_filter, iter_num,frame_num,device,video=None):
        self.model = model
        self.model.eval()
        self.selected_layer = selected_layer
        self.selected_filter = selected_filter
        self.conv_output = 0
        self.iter_num = iter_num
        self.frame_num=frame_num
        self.device=device
        self.video=video
        # Create the folder to export images if not exists
        if not os.path.exists('../generated'):
            os.makedirs('../generated')

    def hook_layer(self):
        def hook_function(module, grad_in, grad_out):
            # Gets the conv output of the selected filter (from selected layer)
            self.conv_output = grad_out[0, self.selected_filter]
        # Hook the selected layer
        self.model[self.selected_layer].register_forward_hook(hook_function)

    def visualise_layer_with_hooks(self):
        # Hook the selected layer
        self.hook_layer()
        # Generate a random image
        random_image = np.uint8(np.random.uniform(150, 180, (224, 224, 3)))
        # Process image and return variable
        processed_image = preprocess_image(random_image, False)
        # Define optimizer for the image
        optimizer = Adam([processed_image], lr=0.1, weight_decay=1e-6)
        for i in range(1, 31):
            optimizer.zero_grad()
            # Assign create image to a variable to move forward in the model
            x = processed_image
            for index, layer in enumerate(self.model):
                # Forward pass layer by layer
                # x is not used after this point because it is only needed to trigger
                # the forward hook function
                x = layer(x)
                # Only need to forward until the selected layer is reached
                if index == self.selected_layer:
                    # (forward hook function triggered)
                    break
            # Loss function is the mean of the output of the selected layer/filter
            # We try to minimize the mean of the output of that specific filter
            loss = -torch.mean(self.conv_output)
            print('Iteration:', str(i), 'Loss:', "{0:.2f}".format(loss.data.numpy()))
            # Backward
            loss.backward()
            # Update image
            optimizer.step()
            # Recreate image
            self.created_image = recreate_image(processed_image)
            # Save image
            if i % 5 == 0:
                im_path = '../generated/layer_vis_l' + str(self.selected_layer) + \
                    '_f' + str(self.selected_filter) + '_iter' + str(i) + '.jpg'
                save_image(self.created_image, im_path)

    def visualise_layer_without_hooks(self):
        # Process image and return variable
        # Generate a random image
        tmp = torch.from_numpy(np.random.uniform(150, 180, (1,3,32,112,112))).float()
        for i in range(self.frame_num):
            if self.video is not None:
                random_image = self.video[i,...]
            else:
                random_image = np.uint8(np.random.uniform(150, 180, (112, 112, 3)))
            # Process image and return variable
            processed_image = preprocess_image(random_image, False)
            tmp[0,:,i,...] = processed_image[0,...]
        processed_image=Variable(tmp, requires_grad=True).to(self.device)
        # Define optimizer for the image
        optimizer = Adam([processed_image], lr=0.1, weight_decay=1e-6)
        for i in range(1, self.iter_num+1):
            optimizer.zero_grad()
            # Assign create image to a variable to move forward in the model
            x = processed_image
            for index, layer in enumerate(self.model):
                # Forward pass layer by layer
                x = layer(x)
                if isinstance(layer, nn.MaxPool3d):
                    x=x[0]
                if index == self.selected_layer:
                    # Only need to forward until the selected layer is reached
                    # Now, x is the output of the selected layer
                    break
            # Here, we get the specific filter from the output of the convolution operation
            # x is a tensor of shape 1x512x28x28.(For layer 17)
            # So there are 512 unique filter outputs
            # Following line selects a filter from 512 filters so self.conv_output will become
            # a tensor of shape 28x28
            self.conv_output = x[0, self.selected_filter]
            # Loss function is the mean of the output of the selected layer/filter
            # We try to minimize the mean of the output of that specific filter
            loss = -torch.mean(self.conv_output)
            print('Iteration:', str(i), 'Loss:', "{0:.2f}".format(loss.data.numpy()))
            # Backward
            loss.backward()
            # Update image
            optimizer.step()
            # Recreate image
            # print(processed_image.size())
            if i % self.iter_num == 0:
                for j in range(self.frame_num):
                    out_img = processed_image[:,:,j,...] # self.selected_frame
                    self.created_image = recreate_image(out_img)
                    # Save image
                    im_path = '../generated/layer_vis_l' + str(self.selected_layer) + \
                        '_f' + str(self.selected_filter) + '_frame' + str(j) + '.jpg'
                    save_image(self.created_image, im_path)

if __name__ == '__main__':
    cnn_layer = 23
    filter_pos = 5
    iter_num = 30
    frame_num = 32
    img_path = './data/0X1A0A263B22CCD966.avi'
    video = loadvideo(img_path)
    video = video[:2:-1,...] # period =2
    video = video[15:(32+15),...] # get 32 frames
    device = torch.device("cpu")
    # Fully connected layer is not needed
    pretrained_model = UNet3D_ef().features
    pretrained_model.to(device)
    for cnn_layer in [1,3,7,10,14,17,21,24,28,31]:
        layer_vis = CNNLayerVisualization(pretrained_model, cnn_layer, filter_pos,iter_num,frame_num,device, video=video)
        # # Layer visualization with pytorch hooks
        # layer_vis.visualise_layer_with_hooks()
        layer_vis.visualise_layer_without_hooks()

