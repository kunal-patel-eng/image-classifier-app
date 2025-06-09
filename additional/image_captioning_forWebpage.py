#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun  8 23:06:45 2025

@author: kunal_patel
"""

# URL of the page to scrape
url = "https://en.wikipedia.org/wiki/IBM"
# Download the page
response = requests.get(url)
# Parse the page with BeautifulSoup
soup = BeautifulSoup(response.text, 'html.parser')

# Find all img elements
img_elements = soup.find_all('img')
# Iterate over each img elements
for img_element in img_elements:
...