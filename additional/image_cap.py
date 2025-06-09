#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun  8 22:11:08 2025

@author: kunal_patel
"""

import requests
from PIL import Image
from transformers import AutoProcessor, BlipForConditionalGeneration
# Load the pretrained processor and model
processor = AutoProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")



# Load your image
import os
base_path= "/Users/kunal_patel/Library/CloudStorage/OneDrive-Personal/Personal/98_Learning/AI_Projects/Image_Captioning/test_images/"
img_path = os.path.join(base_path, "image2.jpg")
#img_path="/Users/kunal_patel/Downloads/animal4.jpeg"
# convert it into an RGB format 
image = Image.open(img_path).convert('RGB')


# You do not need a question for image captioning
text = "the image of"
inputs = processor(images=image, text=text, return_tensors="pt")


# Generate a caption for the image with max 50 tokens
outputs = model.generate(**inputs, max_length=50, min_length=16)


# Decode the generated tokens to text
caption = processor.decode(outputs[0], skip_special_tokens=True)
# Print the caption
print(caption)