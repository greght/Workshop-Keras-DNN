import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras import regularizers
import numpy as np

# Import data
dataIn = np.genfromtxt('iris_training.csv',delimiter=',',skip_header=1)
x_train = dataIn[:,0:-1]
y_train = keras.utils.to_categorical(dataIn[:,-1], num_classes=3)

dataIn = np.genfromtxt('iris_test.csv',delimiter=',',skip_header=1)
x_test = dataIn[:,0:-1]
y_test = keras.utils.to_categorical(dataIn[:,-1], num_classes=3)

# Create model
model = Sequential()
model.add(Dense(units=1000, activation='relu', input_dim=4))
model.add(Dropout(0.5))
model.add(Dense(units=500, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(units=70, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(units=3, activation='softmax'))

# To use l2 regularization, instead use, for example...
#model = Sequential()
#model.add(Dense(units=1000, activation='relu', input_dim=4, kernel_regularizer=regularizers.l2(0.01)))
#model.add(Dense(units=500, activation='relu', kernel_regularizer=regularizers.l2(0.01)))
#model.add(Dense(units=70, activation='relu', kernel_regularizer=regularizers.l2(0.01)))
#model.add(Dense(units=3, activation='softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer=keras.optimizers.Adagrad(lr=0.01),
              metrics=['accuracy'])

# Train
model.fit(x_train, y_train,
          validation_data=(x_test,y_test),
          epochs=2000,
          batch_size=10)

# To use early stopping, instead use, for example...
#earlyStopping = keras.callbacks.EarlyStopping(patience=10)
#model.fit(x_train, y_train, validation_data=(x_test,y_test), epochs=2000, batch_size=10, callbacks=[earlyStopping])

# Create a prediction set
x_predict = np.array([[6.4, 3.2, 4.5, 1.5],
                      [5.8, 3.1, 5.0, 1.7]], dtype=np.float32)
y_predict = model.predict(x_predict)
print(y_predict)
