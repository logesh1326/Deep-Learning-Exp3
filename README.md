# Deep-Learning-Exp3

**DL-Convolutional Deep Neural Network for Image Classification**

**AIM**

To develop a convolutional neural network (CNN) classification model for the given dataset.

**THEORY**

The MNIST dataset consists of 70,000 grayscale images of handwritten digits (0-9), each of size 28×28 pixels. The task is to classify these images into their respective digit categories. CNNs are particularly well-suited for image classification tasks as they can automatically learn spatial hierarchies of features through convolutional layers, pooling layers, and fully connected layers.

**Neural Network Model**

Include the neural network model diagram.

**DESIGN STEPS**

STEP 1:

Write your own steps

STEP 2:

STEP 3:

STEP 4:

STEP 5:

STEP 6:

**PROGRAM**
 ```
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import utils
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.preprocessing import image

(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train.shape
X_test.shape

print("Logesh S 2305001014")
single_image = X_train[0]
single_image.shape
plt.imshow(single_image, cmap='gray')

y_train.shape
X_train.min()
X_train.max()

X_train_scaled = X_train / 255.0
X_test_scaled = X_test / 255.0

X_train_scaled.min()
X_train_scaled.max()

y_train[0]

y_train_onehot = utils.to_categorical(y_train, 10)
y_test_onehot = utils.to_categorical(y_test, 10)

type(y_train_onehot)
y_train_onehot.shape

print("Logesh S 2305001014")
single_image = X_train[400]
plt.imshow(single_image, cmap='gray')
y_train_onehot[500]

X_train_scaled = X_train_scaled.reshape(-1, 28, 28, 1)
X_test_scaled = X_test_scaled.reshape(-1, 28, 28, 1)

model = keras.Sequential()
model.add(layers.Input(shape=(28, 28, 1)))
model.add(layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu'))
model.add(layers.MaxPool2D(pool_size=(2, 2)))
model.add(layers.Flatten())
model.add(layers.Dense(32, activation='relu'))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))

model.summary()

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model.fit(X_train_scaled, y_train_onehot, epochs=5,
          batch_size=128,
          validation_data=(X_test_scaled, y_test_onehot))

metrics = pd.DataFrame(model.history.history)
metrics.head()

print("Logesh S 2305001014")
metrics = pd.DataFrame(model.history.history)
metrics.head()

print("Logesh S 2305001014")
metrics[['accuracy', 'val_accuracy']].plot()

print("Logesh S 2305001014")
metrics[['loss', 'val_loss']].plot()

print("Logesh S 2305001014")
x_test_predictions = np.argmax(model.predict(X_test_scaled), axis=1)

print("Logesh S 2305001014")
print(confusion_matrix(y_test, x_test_predictions))

print("Logesh S 2305001014")
print(classification_report(y_test, x_test_predictions))

img = image.load_img('/content/imgs.png')
img_tensor = tf.convert_to_tensor(np.asarray(img))
img_28 = tf.image.resize(img_tensor, (28, 28))
img_28_gray = tf.image.rgb_to_grayscale(img_28)
img_28_gray_scaled = img_28_gray.numpy() / 255.0

x_single_prediction = np.argmax(
    model.predict(img_28_gray_scaled.reshape(1, 28, 28, 1)),
    axis=1
)

print(x_single_prediction)
print("Logesh S 2305001014")

plt.imshow(img_28_gray_scaled.reshape(28, 28), cmap='gray')

img_28_gray_inverted = 255.0 - img_28_gray
img_28_gray_inverted_scaled = img_28_gray_inverted.numpy() / 255.0

x_single_prediction = np.argmax(
    model.predict(img_28_gray_inverted_scaled.reshape(1, 28, 28, 1)),
    axis=1
)

print("Logesh S 2305001014")
print(x_single_prediction)

```
**OUTPUT**
<img width="812" height="615" alt="image" src="https://github.com/user-attachments/assets/a14e4c44-e324-4976-ab7e-52db901d74e5" />


<img width="805" height="611" alt="image" src="https://github.com/user-attachments/assets/04042fc4-866c-4bcb-9ecf-0a05bb4c4de0" />



**Confusion Matrix**

Include confusion matrix here

**Classification Report**

<img width="633" height="410" alt="image" src="https://github.com/user-attachments/assets/61ea836e-3e14-4793-8c2c-957cac728275" />

**New Sample Data Prediction**
<img width="621" height="609" alt="image" src="https://github.com/user-attachments/assets/461879d1-9d85-4696-9bee-a74b211ad2df" />

<img width="533" height="71" alt="image" src="https://github.com/user-attachments/assets/e447c5d4-a51e-477c-8158-61a010beb789" />

**RESULT**

Thus, a convolutional deep neural network for digit classification and to verify the response for scanned handwritten images is written and executed successfully.
