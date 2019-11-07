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