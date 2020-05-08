import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import random
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D

#Dataset runterladen: http://laurencemoroney.com/rock-paper-scissors-dataset
DATADIR = "C:/Users/ML/rps" # hier dataset pfad rps anlegen bsp: "C:/Users/ML/rps"
CATEGORIES = ["PAPER", "ROCK", "SCISSORS"]

	
IMG_SIZE = 50

new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
	
training_data = []


def create_training_data():
	for category in CATEGORIES:
		path = os.path.join(DATADIR, category) # paths to paper,rock or scissors dir
		class_num = CATEGORIES.index(category)
			for img in os.listdir(path):
				try:
					img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
					new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
					training_data.append([new_array,class_num])
				except Exception as e:
					pass

create_training_data()

random.shuffle(training_data)
x =[]
y =[]


for features, label in training_data:
	x.append(features)
	y.append(label)
	
	
x = np.array(x).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
y = np.array(y)

	

x = x/255.0



model = Sequential()

model.add(Conv2D(256, (3, 3), input_shape=x.shape[1:]))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(256, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors

model.add(Dense(64))
model.add(Activation('relu'))

model.add(Dense(3))	#3 = Rock, Paper, Scissors
model.add(Activation('sigmoid'))


model.compile(loss='sparse_categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model.fit(x, y, batch_size=32, epochs=3)

model.save('65x3-CNN.model')
	

#END 
