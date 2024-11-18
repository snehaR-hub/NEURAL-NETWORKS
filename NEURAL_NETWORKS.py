'''To demonstrate the application of neural networks using Python, lets walk through a simple example of training a neural network to classify images from the popular MNIST dataset, which consists of handwritten digits (0-9). We'll use TensorFlow and Keras, which are popular libraries for building neural networks.'''

'''Steps for the demonstration:
Install required libraries: We will need tensorflow for building the neural network and matplotlib for visualizing the dataset.
Load the MNIST dataset: This dataset is readily available in TensorFlow.
Preprocess the data: Normalize the images and prepare the labels for training.
Build a neural network model: We'll create a simple neural network with one hidden layer.
Compile the model: Define the loss function, optimizer, and evaluation metrics.
Train the model: Fit the model to the training data.
Evaluate the model: Test the model's performance on unseen data.'''

import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import numpy as np

# Step 1: Load the MNIST dataset
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Step 2: Preprocess the data
# Normalize the images to the range [0, 1] by dividing by 255
x_train, x_test = x_train / 255.0, x_test / 255.0

# Step 3: Build the Neural Network Model
model = models.Sequential([
    # Flatten the 28x28 images into a 1D vector of 784 values
    layers.Flatten(input_shape=(28, 28)),
    
    # Hidden layer with 128 neurons and ReLU activation
    layers.Dense(128, activation='relu'),
    
    # Output layer with 10 neurons (for 10 classes) and softmax activation
    layers.Dense(10, activation='softmax')
])

# Step 4: Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Step 5: Train the model
model.fit(x_train, y_train, epochs=5)

# Step 6: Evaluate the model on the test dataset
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
print(f"Test accuracy: {test_acc}")

# Step 7: Make predictions
predictions = model.predict(x_test)

# Display the first image and its predicted label
plt.imshow(x_test[0], cmap=plt.cm.binary)
plt.title(f"Predicted Label: {np.argmax(predictions[0])}")
plt.show()
