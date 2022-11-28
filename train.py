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

data_dir = 'Skin cancer ISIC The International Skin Imaging Collaboration\Train'
imgHeight = 180
imgWidth = 180
            
trainingData = tf.keras.preprocessing.image_dataset_from_directory(data_dir,
                                                           validation_split=0.2,
                                                           seed=14,
                                                           subset="training",
                                                           image_size=(imgWidth,imgHeight),
                                                           batch_size=32)

validationData =  tf.keras.preprocessing.image_dataset_from_directory(data_dir,
                                                           validation_split=0.2,
                                                           seed=14,
                                                           subset="validation",
                                                           image_size=(imgWidth,imgHeight),
                                                           batch_size=32)
classes = trainingData.class_names
numClasses = len(classes)

dataAugmentation = Sequential()
dataAugmentation.add(RandomFlip("horizontal", input_shape=(imgWidth, imgHeight,3)))
dataAugmentation.add(RandomRotation(0.1))
dataAugmentation.add(RandomZoom(0.1))

plt.figure(figsize=(10, 10))
plt.title("Normal data")
for images, labels in trainingData.take(1):
  for i in range(numClasses):
    ax = plt.subplot(3, 3, i + 1)
    plt.imshow(images[i].numpy().astype("uint8"))
    plt.title(classes[labels[i]])
    plt.axis("off")

plt.figure(figsize=(10, 10))
plt.title("Augmentated data")
for images, labels in trainingData.take(1):
  for i in range(numClasses):
      augImage = dataAugmentation(images)
      bx = plt.subplot(3, 3, i + 1)
      plt.imshow(augImage[i].numpy().astype("uint8"))
      plt.title(classes[labels[i]])
      plt.axis("off")
    
model = Sequential()
model.add(dataAugmentation)
model.add(Rescaling(1./255, input_shape=(imgWidth,imgHeight,3)))
model.add(Conv2D(32, (3,3), 1, activation='relu'))
model.add(Conv2D(32, (3,3), 1, activation='relu'))
model.add(MaxPooling2D())
model.add(Conv2D(64, (3,3), 1, activation='relu'))
model.add(Conv2D(64, (3,3), 1, activation='relu'))
model.add(MaxPooling2D())
model.add(Conv2D(128, (3,3), 1, activation='relu'))
model.add(Conv2D(128, (3,3), 1, activation='relu'))
model.add(Conv2D(128, (3,3), 1, activation='relu'))
model.add(MaxPooling2D())
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dense(512, activation='relu'))
model.add(Dense(numClasses, activation='sigmoid'))

model.compile('adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

es = EarlyStopping(monitor='loss', mode='min', verbose=1, patience=25)
mc = ModelCheckpoint('new_model.h5', monitor='accuracy', mode='max', verbose=1, save_best_only=True)

model.fit(trainingData,
          validation_data=validationData, 
          epochs=200, 
          callbacks=[es,mc]
          )



