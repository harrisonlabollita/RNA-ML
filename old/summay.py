import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Reshape
from keras.layers import Conv2D, MaxPooling2D
from keras import optimizers


model = Sequential()
model.add(Conv2D(16, (3, 3), activation = 'relu', input_shape = (30, 30 ,1)))
model.add(MaxPooling2D(pool_size = (3, 3)))
model.add(Conv2D(16, (3, 3), activation = 'relu'))
model.add(MaxPooling2D(pool_size = (3, 3)))
#dim = 16*24*24
model.add(Dense(30, input_shape = (30, ), activation = 'relu'))
model.add(Dense(30, activation = 'relu'))
model.add(Dense(3, activation = 'relu'))
model.summary()
