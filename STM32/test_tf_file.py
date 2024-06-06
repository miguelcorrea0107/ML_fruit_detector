import tensorflow as tf
import numpy as np
from dataloader_lib import load_images_from_folder, print_tflite_model_details, load_images_by_torch


datapath = '/Users/TomasPacheco/Documents/MA2/MLuC/Project/ML_Model/fruits/fruits-360/Test2'

model_path = '/Users/TomasPacheco/Documents/MA2/MLuC/Project/ML_Model/Models/fruit360_cnn'

batch_size = 40

test_images, labels = load_images_by_torch(datapath, batch_size, image_size=(100, 100))

model = tf.keras.models.load_model(model_path)


test_image_indices = range(len(test_images) // batch_size)

predictions = np.zeros((len(test_image_indices), batch_size), dtype=int)
  
for i, test_image_index in enumerate(test_image_indices):
    test_image = test_images[test_image_index * batch_size: (test_image_index + 1)* batch_size]
    # test_label = test_labels[test_image_index * batch_size: (test_image_index + 1)* batch_size]

    if (test_image_index % 100 == 0):
      print("Evaluated on ", test_image_index*40, " images.")

    
    output = model(test_image)
    predictions[i] = output.argmax()


flatten_output =output.flatten()

accuracy = (np.sum(labels== flatten_output) * 100) / len(test_images)

print('Model accuracy is %.4f%% (Number of test samples=%d)' % (
      accuracy, len(test_images)))