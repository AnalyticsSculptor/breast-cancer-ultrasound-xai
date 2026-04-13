import sys
import os
import cv2
import torch
import numpy as np
import gradio as gr
from PIL import Image

# Add project root to path so we can import our modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.classifier import BreastCancerClassifier
from src.data.transforms import get_transforms
from src.xai.gradcam import BreastCancerGradCAM

# 1. Configuration & Initialization
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CHECKPOINT_PATH = "outputs/checkpoints/model_best.pth" # Update this to your actual best checkpoint name later!
IMAGE_SIZE = 224

# Load Model
model = BreastCancerClassifier(pretrained=False) # No need to download pretrained weights for inference
# Uncomment the line below once you have a trained checkpoint!
# model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE)) 
model.to(DEVICE)

# Initialize utilities
transforms = get_transforms(phase="val", image_size=IMAGE_SIZE)
grad_cam = BreastCancerGradCAM(model, device=DEVICE)

def predict_with_uncertainty(image_tensor, num_passes=30):
    """Runs Monte Carlo Dropout to get mean prediction and uncertainty (std dev)."""
    # Keep dropout active by calling model.train() instead of model.eval()
    model.train() 
    
    probs = []
    with torch.no_grad():
        for _ in range(num_passes):
            logits = model(image_tensor.to(DEVICE))
            prob = torch.nn.functional.softmax(logits, dim=1)[0, 1].item() # Probability of Malignant
            probs.append(prob)
            
    mean_prob = np.mean(probs)
    std_dev = np.std(probs) # This is our uncertainty metric
    
    return mean_prob, std_dev

def process_image(input_image):
    if input_image is None:
        return None, "Please upload an image.", None
    
    # Preprocess for model
    image_cv = np.array(input_image)
    augmented = transforms(image=image_cv)
    input_tensor = augmented['image'].unsqueeze(0) # Add batch dimension
    
    # 1. Predict with Uncertainty (MC Dropout)
    malignant_prob, uncertainty = predict_with_uncertainty(input_tensor)
    
    prediction_text = "Malignant" if malignant_prob > 0.5 else "Benign"
    confidence_text = f"Probability: {malignant_prob:.1%} (±{uncertainty:.1%} uncertainty)"
    
    # 2. Explainability (Grad-CAM)
    # Prepare original image for Grad-CAM overlay (resize and scale 0-1)
    orig_resized = cv2.resize(image_cv, (IMAGE_SIZE, IMAGE_SIZE))
    orig_float = np.float32(orig_resized) / 255
    heatmap = grad_cam.generate_heatmap(input_tensor, orig_float)
    
    return heatmap, prediction_text, confidence_text

# 3. Gradio Interface
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🩺 Breast Ultrasound Image Classification")
    gr.Markdown("Upload an ultrasound image. The model will predict Benign vs. Malignant, calculate uncertainty using MC Dropout, and explain its focus using Grad-CAM.")
    
    with gr.Row():
        with gr.Column():
            image_input = gr.Image(type="pil", label="Upload Ultrasound Image")
            submit_btn = gr.Button("Analyze Image", variant="primary")
            
        with gr.Column():
            prediction_output = gr.Textbox(label="Prediction", text_align="center")
            confidence_output = gr.Textbox(label="Confidence & Uncertainty", text_align="center")
            heatmap_output = gr.Image(label="Grad-CAM Explainability Heatmap")

    submit_btn.click(
        fn=process_image,
        inputs=image_input,
        outputs=[heatmap_output, prediction_output, confidence_output]
    )

if __name__ == "__main__":
    demo.launch(share=False)