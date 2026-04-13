import cv2
import numpy as np
import torch
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

class BreastCancerGradCAM:
    def __init__(self, model, device="cuda"):
        self.model = model.to(device)
        self.device = device
        
        # For timm's EfficientNet, the final convolutional layer before pooling 
        # is typically named 'conv_head' or is the last block.
        self.target_layers = [self.model.backbone.conv_head]
        self.cam = GradCAM(model=self.model, target_layers=self.target_layers)

    def generate_heatmap(self, input_tensor, original_image):
        """
        Generates a Grad-CAM overlay on the original image.
        
        Args:
            input_tensor (torch.Tensor): Preprocessed image tensor [1, C, H, W]
            original_image (np.ndarray): Original RGB image scaled between 0 and 1
        """
        self.model.eval()
        
        # Generate the raw heatmap
        grayscale_cam = self.cam(input_tensor=input_tensor.to(self.device))[0, :]
        
        # Overlay the heatmap onto the original image
        visualization = show_cam_on_image(original_image, grayscale_cam, use_rgb=True)
        
        return visualization