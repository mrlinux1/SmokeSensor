import tensorflow
from tensorflow.keras.applications import vgg16
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras import Sequential, optimizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import matplotlib.pyplot as plt

vgg_conv = vgg16.VGG16(weights='imagenet',
                  include_top=False,
                  input_shape=(224, 224, 3))

# each folder contains three subfolders in accordance with the number of classes
train_dir = './dataset/train'
validation_dir = './dataset/validation'

# the number of images for train and test is divided into 80:20 ratio
nTrain = 802
nVal = 197

# load the normalized images
datagen = ImageDataGenerator(rescale=1./255)

# define the batch size
batch_size = 20


# the defined shape is equal to the network output tensor shape
train_features = np.zeros(shape=(nTrain, 7, 7, 512))
train_labels = np.zeros(shape=(nTrain,2))

# the defined shape is equal to the network output tensor shape
validation_features = np.zeros(shape=(nVal, 7, 7, 512))
validation_labels = np.zeros(shape=(nVal,2))

# generate batches of train images and labels
train_generator = datagen.flow_from_directory(
    train_dir,
    target_size=(224, 224),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=True)

# generate batches of train images and labels
validation_generator = datagen.flow_from_directory(
    validation_dir,
    target_size=(224, 224),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=True)

# iterate through the batches of train images and labels
for i, (inputs_batch, labels_batch) in enumerate(train_generator):
    if i * batch_size >= nTrain:
        break    
    # pass the images through the network
    features_batch = vgg_conv.predict(inputs_batch)
    train_features[i * batch_size : (i + 1) * batch_size] = features_batch
    train_labels[i * batch_size : (i + 1) * batch_size] = labels_batch

# iterate through the batches of validation images and labels
for i, (inputs_batch, labels_batch) in enumerate(validation_generator):
    if i * batch_size >= nVal:
        break    
    # pass the images through the network
    features_batch = vgg_conv.predict(inputs_batch)
    validation_features[i * batch_size : (i + 1) * batch_size] = features_batch
    validation_labels[i * batch_size : (i + 1) * batch_size] = labels_batch

# reshape train_features into vector        
train_features_vec = np.reshape(train_features, (nTrain, 7 * 7 * 512))
print("Train features: {}".format(train_features_vec.shape))

# reshape train_features into vector        
validation_features_vec = np.reshape(validation_features, (nVal, 7 * 7 * 512))
print("Validation features: {}".format(validation_features_vec.shape))

model = Sequential()
model.add(Dense(512, activation='relu', input_dim=7 * 7 * 512))
model.add(Dropout(0.5))
model.add(Dense(2, activation='softmax'))

# configure the model for training
model.compile(optimizer=optimizers.RMSprop(lr=2e-4),
              loss='categorical_crossentropy',
              metrics=['acc'])

# use the train and validation feature vectors 
history = model.fit(train_features_vec,
                    train_labels,
                    epochs=20,
                    batch_size=batch_size,
                    validation_data=(validation_features_vec, 
                    validation_labels)
)

# get the list of all validation file names
fnames = validation_generator.filenames

# get the list of the corresponding classes
ground_truth = validation_generator.classes

# get the dictionary of classes
label2index = validation_generator.class_indices

# obtain the list of classes
idx2label = list(label2index.keys())
print("The list of classes: ", idx2label)

predictions = model.predict_classes(validation_features_vec)
prob = model.predict(validation_features_vec)
model.summary()
errors = np.where(predictions != ground_truth)[0]
print("Number of errors = {}/{}".format(len(errors),nVal))



model.save('classification.model')

# Convert the model
converter = tensorflow.lite.TFLiteConverter.from_saved_model("classification.model") # path to the SavedModel directory
converter.optimizations = [tensorflow.lite.Optimize.DEFAULT]

def representative_dataset_generator():
  for value in train_features:
    yield [np.array(value, dtype=np.float32, ndmin=2)]
converter.representative_dataset = representative_dataset_generator

tflite_model = converter.convert()

# Save the model.
with open('model.tflite', 'wb') as f:
  f.write(tflite_model)

for i in range(len(errors)):
    pred_class = np.argmax(prob[errors[i]])
    pred_label = idx2label[pred_class]
   
    print('Original label:{}, Prediction :{}, confidence : {:.3f}'.format(
        fnames[errors[i]].split('/')[0],
        pred_label,
        prob[errors[i]][pred_class]))
    
    original = tensorflow.keras.preprocessing.image.load_img('{}/{}'.format(validation_dir,fnames[errors[i]]))
    plt.axis('off')
    plt.imshow(original)
    plt.show()