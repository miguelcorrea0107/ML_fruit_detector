
import tensorflow as tf
import numpy as np
import pathlib
from dataloader_lib import load_images_from_folder, print_tflite_model_details, load_images_by_torch
from run_model_lib import evaluate_model

from keras.utils import to_categorical

batch_size = 1

datapath = '/Users/TomasPacheco/Documents/MA2/MLuC/git_project/ML_fruit_detector/STM32//ML_Model/fruits/fruits-360/Test'
# model_path = '/Users/TomasPacheco/Documents/MA2/MLuC/git_project/ML_fruit_detector/STM32//ML_Model/Models/tflite/int8_quant.tflite'
model_path = '/Users/TomasPacheco/Documents/MA2/MLuC/git_project/ML_fruit_detector/STM32//ML_Model/Models/tflite/qat_int8.tflite'

# model_path = '/Users/TomasPacheco/Documents/MA2/MLuC/git_project/ML_fruit_detector/STM32//ML_Model/Models/tflite/.tflite'
# model_path = '/Users/TomasPacheco/Documents/MA2/MLuC/git_project/ML_fruit_detector/STM32//ML_Model/Models/tflite/.tflite'

#test_images, test_labels = load_images_from_folder(datapath, image_size=(100, 100))
test_images, test_labels, names_labels = load_images_by_torch(datapath, batch_size, image_size=(100, 100), verbose=False)

test_labels_cat = to_categorical(test_labels, num_classes=131)

model_file = pathlib.Path(model_path)
model_type = "Full Post-Quantized INT8"

evaluate_model(model_file, model_type, test_images, test_labels, batch_size)