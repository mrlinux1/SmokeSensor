import tensorflow as tf
import numpy as np

# Convert the model
converter = tf.lite.TFLiteConverter.from_saved_model("classification.model") # path to the SavedModel directory
converter.optimizations = [tf.lite.Optimize.DEFAULT]

def representative_dataset_generator(x_test):
  for value in x_test:
    yield [np.array(value, dtype=np.float32, ndmin=2)]
converter.representative_dataset = representative_dataset_generator


tflite_model = converter.convert()

# Save the model.
with open('model.tflite', 'wb') as f:
  f.write(tflite_model)

