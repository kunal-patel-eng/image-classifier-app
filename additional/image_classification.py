#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun  8 21:33:25 2025

@author: kunal_patel
"""

import torch

#resnet18 is the model
model = torch.hub.load('pytorch/vision:v0.6.0', 'resnet18', pretrained=True).eval()


import requests
from PIL import Image
from torchvision import transforms
# Download human-readable labels for ImageNet. This is just a list
response = requests.get("https://git.io/JJkYN")
labels = response.text.split("\n")

#function to predict the classifications of the image from the labels
def predict(inp):

#inp is the input which is the PIL (Python Imaging Library) image
#image is converted into a PyTorch tensor
 inp = transforms.ToTensor()(inp).unsqueeze(0)
 with torch.no_grad():
#predicitonal probabilities like a dictionary
  prediction = torch.nn.functional.softmax(model(inp)[0], dim=0)
  confidences = {labels[i]: float(prediction[i]) for i in range(1000)}
 return confidences


#setting up the Gradio interface to predict the classification of the image
import gradio as gr
base_path= "/Users/kunal_patel/Library/CloudStorage/OneDrive-Personal/Personal/98_Learning/AI_Projects/Image_Captioning/test_images/"
files = ["animal1.jpeg", "animal2.jpeg", "animal3.jpeg", "animal4.jpeg", "animal5.jpeg", "animal6.jpeg"]
examples = [base_path + f for f in files]

gr.Interface(fn=predict,
       inputs=gr.Image(type="pil"),
       outputs=gr.Label(num_top_classes=3),
       examples=examples).launch(server_name="127.0.0.1", server_port= 7860)