# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import os
import numpy as np
from PIL import Image

IMAGE_SIZE = 128
IMAGE_CHANNELS = 3
IMAGE_DIR = '/Users/mac/Desktop/GAN ART/wikiart/Cubism/'

images_path = IMAGE_DIR 

training_data = []

for filename in os.listdir(images_path):
 path = os.path.join(images_path, filename)
 image = Image.open(path).resize((IMAGE_SIZE, IMAGE_SIZE), Image.ANTIALIAS)
 training_data.append(np.asarray(image))


training_data = np.reshape(
training_data, (-1, IMAGE_SIZE, IMAGE_SIZE, IMAGE_CHANNELS))
training_data = (training_data / 127.5) - 1

np.save('cubism_data.npy', training_data)
