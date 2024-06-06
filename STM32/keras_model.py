import tensorflow as tf
import numpy as np


from keras import layers, models, optimizers
from dataloader_lib import load_images_from_folder, print_tflite_model_details, load_images_by_torch

import multiprocessing
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import random

from keras.utils import to_categorical


keras_model = models.Sequential([
    layers.Conv2D(16, kernel_size=(2 ,2), padding='same', input_shape=(100, 100, 3)),
    layers.BatchNormalization(),
    layers.ReLU(),
    layers.MaxPooling2D(pool_size=(2, 2)),  # 16 X 50 X 50

    layers.Conv2D(32, kernel_size=(2 ,2), strides=(1, 1), padding='same'),
    layers.BatchNormalization(),
    layers.ReLU(),
    layers.MaxPooling2D(pool_size=(2, 2)),  # 32 X 25 X 25

    layers.Conv2D(64, kernel_size=(3 ,3), strides=(1, 1), padding='same'),
    layers.BatchNormalization(),
    layers.ReLU(),
    layers.MaxPooling2D(pool_size=(5, 5)),  # 64 X 5 X 5

    layers.Flatten(),
    layers.Dropout(0.3),
    layers.ReLU(),
    layers.Dense(32*10, activation='relu'),
    layers.Dense(131, activation='softmax')
    
])

keras_model.summary()

batch_size = 40

training_datapath = '/Users/TomasPacheco/Documents/MA2/MLuC/Project/ML_Model/fruits/fruits-360/Training'

train_images, train_labels, train_names = load_images_by_torch(training_datapath, batch_size=batch_size, image_size=(100, 100), validation_set = False)
# train_images____, train_labels_string = load_images_from_folder(training_datapath, image_size=(100, 100))
# train_labels = LabelEncoder().fit_transform(train_labels_string)
print("Shape of train_images: ", train_images.shape)

train_labels = to_categorical(train_labels, num_classes=131)

# # Calculate mean and std of the training set
# mean = np.mean(train_images, axis=(0, 1, 2))
# std = np.std(train_images, axis=(0, 1, 2))
# print(f"Mean: {mean}, Std: {std}")


val_datapath = '/Users/TomasPacheco/Documents/MA2/MLuC/Project/ML_Model/fruits/fruits-360/Test2'

# val_images, val_labels, val_names = load_images_by_torch(val_datapath, batch_size=batch_size, image_size=(100, 100), validation_set = False)
# val_images, val_labels_string = load_images_from_folder(val_datapath, image_size=(100, 100))
# val_labels = LabelEncoder().fit_transform(val_labels_string)

# Plot a random image to check data 
plot = False
if plot:

    i = random.randint(0, len(train_images))
    # print("Training image shape: ", train_images[i].shape)
    plt.imshow(train_images[i])
    plt.title(train_names[train_labels[i]])
    plt.show()


# Define the learning rate scheduler
initial_learning_rate = 0.05
lr_schedule = optimizers.schedules.ExponentialDecay(
    initial_learning_rate,
    decay_steps=10000,
    decay_rate=0.96,
    staircase=True
)

# # Compile the model
keras_model.compile(optimizer=optimizers.Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

keras_model.summary()


# Save the model
keras_model.save('/Users/TomasPacheco/Documents/MA2/MLuC/Project/ML_Model/Models/keras/cnn_keras_rand')

# Train the model (this will take a while)
# The early stopping (es) callback will stop the training when the validation loss stops improving
es = tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)
history = keras_model.fit(
    train_images,
    train_labels,
    batch_size=batch_size,
    epochs=30,
    validation_split=0.2,
    shuffle=True,
    #callbacks=[es]
)


# Save the model
keras_model.save('/Users/TomasPacheco/Documents/MA2/MLuC/Project/ML_Model/Models/keras/cnn_keras')

# Evaluate the model on the test set


test_datapath = '/Users/TomasPacheco/Documents/MA2/MLuC/Project/ML_Model/fruits/fruits-360/Test'

test_images, test_labels, test_labels_names= load_images_by_torch(test_datapath, batch_size=batch_size, image_size=(100, 100))
test_labels = to_categorical(test_labels, num_classes=131)

keras_test_loss, keras_test_acc = keras_model.evaluate(test_images,  test_labels, verbose=2)
print('Test accuracy:', keras_test_acc)
print('Test loss:', keras_test_loss)