#see https://github.com/keras-team/keras/blob/master/examples/mnist_cnn.py

import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten, BatchNormalization
import numpy as np

# Create model
model = Sequential()
model.add(BatchNormalization(input_shape=(28,28,1)))
model.add(Conv2D(4,kernel_size=(5,5),
                 activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Conv2D(16,(5,5),activation='relu'))
model.add(MaxPooling2D((2,2)))
model.add(Flatten())
model.add(Dense(100,activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(10,activation='softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer=keras.optimizers.Adagrad(lr=0.01),
              metrics=['accuracy'])

# Load data
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
y_train = keras.utils.to_categorical(y_train, num_classes=10)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
y_test = keras.utils.to_categorical(y_test, num_classes=10)
          
# Train
model.fit(x_train, y_train, validation_data=(x_test,y_test), epochs=15, batch_size=100, shuffle=True)

# Validate
score = model.evaluate(x_test, y_test)
print(score)
