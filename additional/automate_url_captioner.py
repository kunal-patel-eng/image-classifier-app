#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun  8 23:23:35 2025

@author: kunal_patel
"""

import os
import requests
import glob
from PIL import Image
from io import BytesIO
from bs4 import BeautifulSoup
from transformers import AutoProcessor, BlipForConditionalGeneration
# Load the pretrained processor and model
processor = AutoProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")



# Specify the directory where your images are
#------------------------------------------------------------------------------
image_dir = "/path/to/your/images"
image_exts = ["jpg", "jpeg", "png"]  # specify the image file extensions to search for
# Open a file to write the captions
with open("captions.txt", "w") as caption_file:
# Iterate over each image file in the directory
    for image_ext in image_exts:
        for img_path in glob.glob(os.path.join(image_dir, f"*.{image_ext}")):
            # Load your image
            raw_image = Image.open(img_path).convert('RGB')

#------------------------------------------------------------------------------


#URL based method
#------------------------------------------------------------------------------
# URL of the page to scrape
url = "https://en.wikipedia.org/wiki/IBM"
# Download the page
response = requests.get(url)
# Parse the page with BeautifulSoup
soup = BeautifulSoup(response.text, 'html.parser')
# Find all img elements
img_elements = soup.find_all('img')
# Open a file to write the captions
with open("captions.txt", "w") as caption_file:
    # Iterate over each img element
    for img_element in img_elements:
        img_url = img_element.get('src')
        # Skip if the image is an SVG or too small (likely an icon)
        if 'svg' in img_url or '1x1' in img_url:
            continue
        # Correct the URL if it's malformed
        if img_url.startswith('//'):
            img_url = 'https:' + img_url
        elif not img_url.startswith('http://') and not img_url.startswith('https://'):
            continue  # Skip URLs that don't start with http:// or https://
        try:
            # Download the image
            response = requests.get(img_url)
            # Convert the image data to a PIL Image
            raw_image = Image.open(BytesIO(response.content))
            if raw_image.size[0] * raw_image.size[1] < 400:  # Skip very small images
                continue
            
#------------------------------------------------------------------------------            
            
            
            raw_image = raw_image.convert('RGB')
            # Process the image
            inputs = processor(raw_image, return_tensors="pt")
            # Generate a caption for the image
            out = model.generate(**inputs, max_new_tokens=50)
            # Decode the generated tokens to text
            caption = processor.decode(out[0], skip_special_tokens=True)
            # Write the caption to the file, prepended by the image URL
            caption_file.write(f"{img_url}: {caption}\n")
        except Exception as e:
            print(f"Error processing image {img_url}: {e}")
            continue