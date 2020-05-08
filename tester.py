import cv2
import os
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

CATEGORIES = ["Paper", "Rock", "Scissors"]

def prepare(filepath):
	IMG_SIZE = 50
	img_array = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
	img_array = img_array/255.0
	new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
	plt.imshow(img_array, cmap="gray")
	plt.show()
	return new_array.reshape(-1, IMG_SIZE, IMG_SIZE, 1)


model = tf.keras.models.load_model("65x3-CNN.model")

p = prepare('C:/Users/ML/rps-test-set/scissors/testscissors03-07.png')	# pfad Bild angeben
prediction = model.predict([p])
class_name=model.predict_classes([p])
print(class_name)						# gibt entweder [0] [1] [2] aus
print(CATEGORIES[int(class_name)])		# gibt Name zu [0] [1] [2] aus
print("Done")


#END