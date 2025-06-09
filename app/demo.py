#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun  8 22:41:59 2025

@author: kunal_patel
"""
import os
import torch
from torchvision import transforms
import requests
from PIL import Image
import gradio as gr

# Load pre-trained model
model = torch.hub.load('pytorch/vision:v0.6.0', 'resnet18', pretrained=True).eval()

# Download human-readable labels for ImageNet
response = requests.get("https://git.io/JJkYN")
labels = response.text.split("\n")

# Define prediction function
def predict(inp):
    inp = transforms.ToTensor()(inp).unsqueeze(0)
    with torch.no_grad():
        prediction = torch.nn.functional.softmax(model(inp)[0], dim=0)
        confidences = {labels[i]: float(prediction[i]) for i in range(1000)}
    return confidences

# Define Gradio interface
gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil"),
    outputs=gr.Label(num_top_classes=3)
).launch(server_name="0.0.0.0", server_port=8080)
