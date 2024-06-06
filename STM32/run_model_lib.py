import tensorflow as tf
import numpy as np
from dataloader_lib import print_tflite_model_details


# Helper function to run inference on a TFLite model
def run_tflite_model(tflite_file, test_image_indices, test_images, test_labels, batch_size=40):

  # Initialize the interpreter
  interpreter = tf.lite.Interpreter(model_path=str(tflite_file))
  interpreter.allocate_tensors()


  print_tflite_model_details(interpreter)
  input_details = interpreter.get_input_details()[0]
  output_details = interpreter.get_output_details()[0]

  predictions = np.zeros((len(test_image_indices), batch_size), dtype=int)
  
  for i, test_image_index in enumerate(test_image_indices):
    test_image = test_images[test_image_index * batch_size: (test_image_index + 1)* batch_size]
    test_label = test_labels[test_image_index * batch_size: (test_image_index + 1)* batch_size]

    if (test_image_index % 1000 == 0):
      print("Evaluated on ", test_image_index * batch_size, " images.")


    # Check if the input type is quantized, then rescale input data to uint8
    if input_details['dtype'] == np.uint8:
      input_scale, input_zero_point = input_details["quantization"]
      test_image = test_image / input_scale + input_zero_point
      test_image = test_image.astype(input_details["dtype"])

    interpreter.set_tensor(input_details["index"], test_image)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details["index"])

    predictions[i] = output.argmax()

  return predictions


# Helper function to evaluate a TFLite model on all images
def evaluate_model(tflite_file, model_type, test_images, test_labels, batch_size=40):

  test_image_indices = range(test_images.shape[0] // batch_size) # Number of batches to process
  predictions = run_tflite_model(tflite_file, test_image_indices, test_images, test_labels, batch_size)

  flatten_predictions = predictions.flatten()

  accuracy = (np.sum(test_labels== flatten_predictions) * 100) / len(test_images)

  print('%s model accuracy is %.4f%% (Number of test samples=%d)' % (
      model_type, accuracy, len(test_images)))
  
  