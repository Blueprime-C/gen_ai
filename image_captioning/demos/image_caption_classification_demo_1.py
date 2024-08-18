from typing import Tuple

from transformers import BlipProcessor, BlipForConditionalGeneration
from torchvision import transforms
import gradio as gr
import numpy as np
import torch
import requests


processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
torch_model = torch.hub.load('pytorch/vision:v0.6.0', 'resnet18', pretrained=True).eval()

# get human-readable labels
labels = requests.get('https://git.io/JJkYN').text.split('\n')
def predict_image_category(input_image) -> dict:
    inp = transforms.ToTensor()(input_image).unsqueeze(0)
    with torch.no_grad():
        predictions = torch.nn.functional.softmax(torch_model(inp)[0], dim=0)
        confidences = {labels[i]: float(predictions[i]) for i in range(len(predictions))}
    return confidences

def get_caption(image) -> str:
    inputs = processor(image, return_tensors="pt")
    outputs = model.generate(**inputs)
    caption = processor.decode(outputs[0], skip_special_tokens=True)
    return caption

# def classify_image(image) -> str:

def caption_classify_image(image) -> tuple[str, dict | str]:
    """Takes an image and returns its caption and its classification labels."""

    try:
        caption = get_caption(image)
    except Exception as e:
        caption = f"An error occurred: {str(e)}"

    try:
        labels = predict_image_category(image)
    except ImportError as e:
        labels = f"An error occurred: {str(e)}"

    return caption, labels


demo_ui = gr.Interface(
    fn=caption_classify_image,
    inputs=gr.Image(type="pil"),
    outputs=["text", gr.Label(num_top_classes=3)],
    title="Captioning and classification of image with Blip Tool",
    description="Upload an Image to generate a caption and label.",
    example_labels=['/content/lion.jpg", "/content/cheetah.jpg']
)

demo_ui.launch()