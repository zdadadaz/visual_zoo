from PIL import Image
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
from scipy.interpolate import interp1d
from misc_functions import get_vide_example_params, save_class_activation_videos

class CamExtractor():
    """
        Extracts cam features from the model
    """
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer

    def forward_pass_on_convolutions(self, x):
        """
            Does a forward pass on convolutions, hooks the function at given layer
        """
        conv_output = None
        for module_pos, module in self.model.features._modules.items():
            x = module(x)  # Forward
            if isinstance(module, nn.MaxPool3d):
                x=x[0]
            if int(module_pos) == self.target_layer:
                conv_output = x  # Save the convolution output on that layer
        return conv_output, x

    def forward_pass(self, x):
        """
            Does a full forward pass on the model
        """
        # Forward pass on the convolutions
        conv_output, x = self.forward_pass_on_convolutions(x)
#         x = x.view(x.size(0), -1)  # Flatten
        # Forward pass on the classifier
        x = self.model.fc(x)
        return conv_output, x


class ScoreCam():
    """
        Produces class activation map
    """
    def __init__(self, model, target_layer):
        self.model = model
        self.model.eval()
        # Define extractor
        self.extractor = CamExtractor(self.model, target_layer)

    def generate_cam(self, input_video, target_class=None):
        # Full forward pass
        # conv_output is the output of convolutions at specified layer
        # model_output is the final output of the model (1, 1000)
        size_out = input_video.size()[2:]
        conv_output, model_output = self.extractor.forward_pass(input_video)
        if target_class is None:
            target_class = np.argmax(model_output.data.numpy())
        # Get convolution outputs
        target = conv_output[0]
        # Create empty numpy array for cam
        cam = np.ones(target.shape[1:], dtype=np.float32)
#         print('cam',cam.shape) # 2 7 7 
#         # Multiply each weight with its conv output and then, sum
#         for i in range(len(target)):
        for i in range(1):
            # Unsqueeze to 4D
#             for f in range(target.shape[1]):
            w_arr = []
            for f in range(1):
                saliency_map = torch.unsqueeze(torch.unsqueeze(target[i, f, :, :],0),0)
                # Upsampling to input size
                saliency_map = F.interpolate(saliency_map, size=(112, 112), mode='bilinear', align_corners=False)
                saliency_map = torch.unsqueeze(saliency_map,0)
                if saliency_map.max() == saliency_map.min():
                    continue
                # Scale between 0-1
                norm_saliency_map = (saliency_map - saliency_map.min()) / (saliency_map.max() - saliency_map.min())
                # Get the target score
                print('input_video',input_video.shape,'norm_saliency_map',norm_saliency_map.shape)
                w = self.extractor.forward_pass(input_video*norm_saliency_map)[1]
                w = (100-abs(w.view( -1).data.numpy() - target_class))/100
                w_arr.append(w)
#                 w = F.softmax(self.extractor.forward_pass(input_video*norm_saliency_map)[1],dim=1)[0][target_class]
#                 print('w_arr',len(w_arr),'target',target.shape)
                cam[f] += w * target[i, f, :, :].data.numpy()
        
        cam = np.maximum(cam, 0)
        cam = (cam - np.min(cam)) / (np.max(cam) - np.min(cam))  # Normalize between 0-1
#         cam = np.uint8(cam * 255)  # Scale between 0-255 to visualize
#         cam = np.uint8(Image.fromarray(cam).resize((input_image.shape[2],
#                        input_image.shape[3]), Image.ANTIALIAS))/255
        cam = self._extend_time_space_domain(cam, size_out)
#         print('Final cam',cam.shape)
        return cam
    
    def _extend_time_space_domain(self,cam,size_out):
        # input cam 2x7x7
        # output am 32x112x112
        tmp_shape = (size_out[0], cam.shape[1],cam.shape[2])
        out = np.ones(tmp_shape, dtype=np.float32)  # 32x7x7
        out2 = np.ones(size_out, dtype=np.float32)  # 32x112x112
        for i in range(cam.shape[1]):
            for j in range(cam.shape[2]):
                x = np.linspace(0,1, num=cam.shape[0], endpoint=True)
                y = cam[:,i,j]
                f = interp1d(x, y)
                x1 = np.linspace(0, 1, num=size_out[0], endpoint=True)
                out[:,i,j] = f(x1)
        print('out',out.shape)
        out = (out - np.min(out)) / (np.max(out) - np.min(out))  
        for f in range(size_out[0]):
            tmp = out[f,...]
            tmp = np.uint8(tmp * 255) 
            out2[f,...] = cv2.resize(tmp, (size_out[1],size_out[2]))/255.
#             out2[f,...] = np.uint8(Image.fromarray(tmp).resize((size_out[1],size_out[2]), Image.ANTIALIAS))/255
        return out2


if __name__ == '__main__':
    # Get params
    target_example = 0  # 
    target_layer = 31
    device = torch.device("cpu")
    (original_video, prep_video, target_class, file_name_to_export, pretrained_model) =\
        get_vide_example_params(target_example,device)
    for target_layer in [0, 3, 7, 10, 14, 17, 21, 24, 28, 31]:
        # Score cam
        print('get_vide_example_params',get_vide_example_params)
        score_cam = ScoreCam(pretrained_model, target_layer=target_layer)
        # Generate cam mask
        cam = score_cam.generate_cam(prep_video, target_class)

        # Save mask
        save_class_activation_videos(original_video, cam, file_name_to_export, target_layer)
        print('Score cam completed')
