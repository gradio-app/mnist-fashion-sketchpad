# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt

#import the data
fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()


#store the image class names
class_names = ['T-shirt', 'Trouser', 'Pullover', 'Dress', 'Coat', 
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']


#Explore the Data

train_images.shape

len(train_labels)

test_images.shape

len(test_labels)


#Preprocess the data

#need to scale the values to a range 0 to 1
train_images = train_images / 255.0

test_images = test_images / 255.0



#Build the model - LeNet inspired CNN architecture
#Source: https://www.pyimagesearch.com/2016/08/01/lenet-convolutional-neural-network-in-python/

model = keras.models.Sequential([
    keras.layers.Conv2D(20, (5,5), padding='same', activation='relu', input_shape=(28,28,1)),
    keras.layers.MaxPooling2D(pool_size=(2,2), strides=(2,2)),
    keras.layers.Conv2D(50, (5,5), padding='same', activation='relu'),
    keras.layers.MaxPooling2D(pool_size=(2,2), strides=(2,2)),
    keras.layers.Flatten(),
    keras.layers.Dense(500, activation='relu'),
    keras.layers.Dense(10, activation='softmax')
])


#view model details
model.summary()

#reshape the model
train_images=train_images.reshape(train_images.shape[0], 28, 28, 1)
test_images=test_images.reshape(test_images.shape[0], 28, 28 ,1) 
                                            
#output of the model will be 1D vector with size 10
#convert current representation of the labels to "One Hot Representation"
train_labels=keras.utils.to_categorical(train_labels)
test_labels=keras.utils.to_categorical(test_labels)


#view the new shape of train and test
print('train_images shape:', train_images.shape)
print('test_images shape:', test_images.shape)
print('train_labels shape:', train_labels.shape)
print('test_labels shape:', test_labels.shape)

#view one hot representation 
train_labels[0]


#Compile the model
model.compile(optimizer=tf.train.AdamOptimizer(), 
             loss='categorical_crossentropy',
              metrics=['accuracy'])


#Train the model
model.fit(train_images, train_labels, epochs=5)
model.save('mnist-fashion-model.h5')


#Evaluate the accuracy
test_loss, test_acc = model.evaluate(test_images, test_labels)

print('Test accuracy:', test_acc)
