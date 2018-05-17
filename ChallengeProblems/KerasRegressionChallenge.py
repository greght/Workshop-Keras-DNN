import keras
from keras.models import Sequential
from keras.layers import Dense
import numpy as np

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Create model
model = Sequential()
model.add(Dense(units=20, activation='relu', input_dim=2))
model.add(Dense(units=1))

model.compile(loss='mse',optimizer=keras.optimizers.Adagrad(lr=0.1))

# Import data
dataIn = np.genfromtxt('dataRegression_train.csv',delimiter=',')
x_train = dataIn[:,0:2]
y_train = dataIn[:,2]
dataIn = np.genfromtxt('dataRegression_test.csv',delimiter=',')
x_test = dataIn[:,0:2]
y_test = dataIn[:,2]

# Train
model.fit(x_train, y_train, validation_data=(x_test,y_test), epochs=1000, batch_size=10)

# Validate
loss = model.evaluate(x_train, y_train, batch_size=10)
print(loss)

# Create a prediction set
x_pred = np.mgrid[0:1:25j, 0:1:25j].reshape(2,-1).T
y_pred = model.predict(x_pred)

# Plot the actual and predicted values
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

x1 = x_pred[:,0].reshape(25,-1)
x2 = x_pred[:,1].reshape(25,-1)
y = y_pred.reshape(25,-1)

ax.scatter(x_train[:,0], x_train[:,1], y_train, c='r', marker='o')
ax.scatter(x_test[:,0], x_test[:,1], y_test, c='b', marker='o')
ax.plot_surface(x1,x2,y)

ax.update({'xlabel':'x1', 'ylabel':'x2', 'zlabel':'f(x1,x2)'})

plt.show()
