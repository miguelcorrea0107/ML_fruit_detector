
import tensorflow as tf
import tensorflow_model_optimization as tfmot
from dataloader_lib import load_images_from_folder, print_tflite_model_details, load_images_by_torch, hex_to_c_array

from keras import layers, models, optimizers

from keras.utils import to_categorical
import os


global batch_size
batch_size = 40

no_opt = False
dyn_range_opt = False
quant_opt = False
quant_aware_opt = True

# Load TF model and convert to TFLite
tf_path = '/Users/TomasPacheco/Documents/MA2/MLuC/git_project/ML_fruit_detector/STM32/ML_Model/Models/keras/cnn_keras_good2'
path_test2 = '/Users/TomasPacheco/Documents/MA2/MLuC/git_project/ML_fruit_detector/STM32/ML_Model/fruits/fruits-360/Test2'
path_test = '/Users/TomasPacheco/Documents/MA2/MLuC/git_project/ML_fruit_detector/STM32/ML_Model/fruits/fruits-360/Test'
path_training =  '/Users/TomasPacheco/Documents/MA2/MLuC/git_project/ML_fruit_detector/STM32/ML_Model/fruits/fruits-360/Training'


def print_interpreter(interpreter) :
    # print_tflite_model_details(interpreter)
    input_type = interpreter.get_input_details()[0]['dtype']
    print('input: ', input_type)
    output_type = interpreter.get_output_details()[0]['dtype']
    print('output: ', output_type)

def convert_model_to_c(tflite_model, c_model_name):
    # check if dir 'cfiles' exists, if not create it
    if not os.path.exists('ML_Model/Models/cfiles'):
        os.makedirs('ML_Model/Models/cfiles')
    # Write TFLite model to a C source (or header) file
    with open('ML_Model/Models/cfiles/' + c_model_name + '.h', 'w') as file:
        file.write(hex_to_c_array(tflite_model, c_model_name))
    print("Model converted to C file")

if no_opt :

    converter = tf.lite.TFLiteConverter.from_saved_model(tf_path)
    dyn_range_quant_tflite_model = converter.convert()
    open('/Users/TomasPacheco/Documents/MA2/MLuC/git_project/ML_fruit_detector/STM32/ML_Model/Models/tflite/no_opt.tflite', 'wb').write(dyn_range_quant_tflite_model)

    interpreter = tf.lite.Interpreter(model_content=dyn_range_quant_tflite_model)
    print_interpreter(interpreter)
    convert_model_to_c(dyn_range_quant_tflite_model, 'no_opt')

if dyn_range_opt :

    converter = tf.lite.TFLiteConverter.from_saved_model(tf_path)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    dyn_range_quant_tflite_model = converter.convert()
    open('/Users/TomasPacheco/Documents/MA2/MLuC/git_project/ML_fruit_detector/STM32/ML_Model/Models/tflite/dyn_range_quant.tflite', 'wb').write(dyn_range_quant_tflite_model)

    interpreter = tf.lite.Interpreter(model_content=dyn_range_quant_tflite_model)
    print_interpreter(interpreter)
    convert_model_to_c(dyn_range_quant_tflite_model, 'dyn_range_quant')


if quant_opt :



    converter = tf.lite.TFLiteConverter.from_saved_model(tf_path)
    train_images2, train2_labels, label_names = load_images_by_torch(path_test2, batch_size, image_size=(100, 100))

    def representative_data_gen():
        for input_value in tf.data.Dataset.from_tensor_slices(train_images2).batch(batch_size).take(400):
            yield [input_value]


    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_data_gen
    # Ensure that if any ops can't be quantized, the converter throws an error
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    # Set the input and output tensors to uint8 (APIs added in r2.3)
    converter.inference_input_type = tf.uint8
    converter.inference_output_type = tf.uint8

    tflite_model_quant_int8 = converter.convert()
    open('/Users/TomasPacheco/Documents/MA2/MLuC/git_project/ML_fruit_detector/STM32/ML_Model/Models/tflite/int8_quant.tflite', 'wb').write(tflite_model_quant_int8)

    interpreter = tf.lite.Interpreter(model_content=tflite_model_quant_int8)
    print_interpreter(interpreter)
    convert_model_to_c(tflite_model_quant_int8, 'int8_quant')

if quant_aware_opt:
   # KEras Load do modelo
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
    layers.Dense(16*5, activation='relu'),
    layers.Dense(131, activation='softmax')
    
    ])
   # Retreinar

    # Convert the model to a quantization aware model
    quant_aware_model = tfmot.quantization.keras.quantize_model(keras_model)

    # `quantize_model` requires a recompile.
    quant_aware_model.compile(optimizer=optimizers.Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

    quant_aware_model.summary()
    train_images, train_labels, label_names = load_images_by_torch(path_training, batch_size, image_size=(100, 100))
    train_labels_cat = to_categorical(train_labels, num_classes=131)


    def representative_data_gen():
        for input_value in tf.data.Dataset.from_tensor_slices(train_images).batch(batch_size).take(400):
            yield [input_value]


    history = quant_aware_model.fit(
        train_images,
        train_labels_cat,
        batch_size=batch_size,
        epochs=5,
        validation_split=0.2,
        shuffle=True,
        #callbacks=[es]
    )

    converter = tf.lite.TFLiteConverter.from_keras_model(quant_aware_model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_data_gen
    # Ensure that if any ops can't be quantized, the converter throws an error
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    # Set the input and output tensors to uint8 (APIs added in r2.3)
    converter.inference_input_type = tf.uint8
    converter.inference_output_type = tf.uint8

    print("Converting model to TFLite...")

    tflite_int8_qat = converter.convert()
    
    print("Model converted to TFLite")

    open('/Users/TomasPacheco/Documents/MA2/MLuC/git_project/ML_fruit_detector/STM32/ML_Model/Models/tflite/qat_int8.tflite', 'wb').write(tflite_int8_qat)

    print("Model stored to disk")

    interpreter = tf.lite.Interpreter(model_content=tflite_int8_qat)
    print_interpreter(interpreter)  

    convert_model_to_c(tflite_int8_qat, 'qat_int8')



