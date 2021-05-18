import tensorflow as tf


from tf.keras.applications import vgg16

vgg_conv = vgg16.VGG16(weights='imagenet',
                  include_top=False,
                  input_shape=(224, 224, 3))
