import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Conv2D, BatchNormalization, MaxPooling2D, Flatten, Dense, Dropout
from keras.models import Sequential 
from keras.layers import Dense, Activation 


mnist = keras.datasets.mnist

(X_train, Y_train), (X_test, Y_test) = mnist.load_data()

#Normalization
X_train = X_train/255

#Reshaped for Conv2D, it needs 4 dimensions
X_train = X_train.reshape(60000, 28, 28, 1)


model = Sequential()
model.add(Conv2D(96, (3,3), activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(Conv2D(96, (3,3), activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(MaxPooling2D(2,2))
model.add(Conv2D(192, (3,3), activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(Conv2D(256, (3,3), activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(MaxPooling2D(2,2))

model.add(Conv2D(256, (3,3), activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(Conv2D(256, (3,3), activation='relu', padding='same'))

model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.25))
model.add(Dense(10, activation='softmax'))

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['acc'])

#The very same model achieves 99.635% test accuracy on Kaggle.
model.fit(X_train,Y_train,epochs=10)

model.save('MNIST_model.h5')

