import glob
import os
import sys

import cv2
import numpy as np
import tensorflow as tf

"""
python3 predict.py <MODEL> <INPUT FOLDER> <DST FOLDER>
"""

model = sys.argv[1]
img_dir = os.path.join(sys.argv[2], "*.jpg")

model = tf.keras.models.load_model(model)

paths = [x for x in glob.glob(img_dir)]
paths.sort()

def read_img(img_path):
  img = cv2.imread(img_path)
  img = cv2.resize(img, (720, 576))
  red = img[:, :, 2].copy()
  blue = img[:, :, 0].copy()
  img[:, :, 0] = red
  img[:, :, 2] = blue
  img = np.reshape(img, [1, 576, 720, 3])
  img = img.astype(np.float32) / 255.
  return img

def write_img(mask, num):
  mask *= 255
  mask = mask.astype(np.uint8)
  dst_path = os.path.join(sys.argv[3], str(num) + ".jpg")
  cv2.imwrite(dst_path, mask)
  
for path in paths:
  img_num = int(path[-8:-4])
  image = read_img(path)
  mask = model.predict(image)[0]
  write_img(mask, img_num)
