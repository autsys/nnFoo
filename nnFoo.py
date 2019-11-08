import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

data = keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = data.load_data()

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

#normalize data (so data in array is # value between 0 and 1 instead of 0 and 255)
train_images = train_images/255.0
test_images = test_images/255.0

#example to show image in dataset
plt.imshow(train_images[0], cmap=plt.cm.binary)
plt.show()

#hidden layers should be roughly 15-20% amount of imput neurons
#for example in this we have 784 input neurons (28x28) so we
#can add 128 hidden layers

model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)), #flatten (compresses the 2d array)
    keras.layers.Dense(128, activation='relu'), #hidden layer - rectified linear activaton
    keras.layers.Dense(10, activation='softmax') #output layer - softmax 0-1 
])

#look these up (adam, sparse, the metrics etc..)
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=["accuracy"])

#epoch is the amount of times the entire data set is ran through (how many times each image is seen)
model.fit(train_images, train_labels, epochs=5)



