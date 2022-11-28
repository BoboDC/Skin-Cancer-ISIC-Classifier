import tensorflow as tf
import os
import cv2
import imghdr
import numpy as np
from matplotlib import pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout, Rescaling, RandomZoom, RandomFlip, RandomRotation
from keras.models import load_model, save_model
from keras.callbacks import EarlyStopping, ModelCheckpoint


import warnings
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore",message="2022")
plt.close('all')

data_dir = 'Skin cancer ISIC The International Skin Imaging Collaboration\Test'
imgHeight = 180
imgWidth = 180
            
testData = tf.keras.preprocessing.image_dataset_from_directory(data_dir,
                                                           seed=14,
                                                           image_size=(imgWidth,imgHeight),
                                                           )

classes = testData.class_names
numClasses = len(classes)
labels = np.concatenate([y for x, y in testData], axis=0)
realLabels = []

for i in range(len(labels)):
    realLabels.append(classes[labels[i]])

plt.figure(figsize=(10, 10))
plt.title("Normal data")
for images, labels in testData.take(1):
  for i in range(numClasses):
    ax = plt.subplot(3, 3, i + 1)
    plt.imshow(images[i].numpy().astype("uint8"))
    plt.title(classes[labels[i]])
    plt.axis("off")

model = load_model('new_model.h5')

predictions = model.predict(testData)
maxPrediction = []
maxPredictionName = []
  
for i in range(len(predictions)):
        maxPrediction.append(np.max(predictions[i]))
        maxPredictionName.append(classes[np.argmax(predictions[i])])

testImage = tf.keras.preprocessing.image.load_img("nevus.jpg",
                                                  target_size=(imgWidth,imgHeight))
testImageArray = tf.keras.preprocessing.image.img_to_array(testImage)
testImageArray = tf.expand_dims(testImageArray, 0)
predictionImage = model.predict(testImageArray)
print(classes[np.argmax(predictionImage[0])])
print(np.max(predictionImage[0]))

score = model.evaluate(testData)
print("Test loss: ", score[0])
print("Test accuracy: ", score[1])    



