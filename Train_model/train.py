

from tensorflow.keras.datasets import cifar10
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten ,Conv2D, MaxPooling2D
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.optimizers import Nadam


import pickle


opt=Nadam(lr=0.0005, beta_1=0.9, beta_2=0.999, epsilon=None, schedule_decay=0.004)

pickle_in = open("X.pickle","rb")
X = pickle.load(pickle_in)

pickle_in = open("y.pickle","rb")
y = pickle.load(pickle_in)






X = X/255.0


# a simple keras model with three layers tends to overfit on later epochs so it saves on best validation loss and uses Dropout layers to reduce overfitting


save=tf.keras.callbacks.ModelCheckpoint('mdl.h5', monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=False, mode='min', period=1)

layer_1_neurons=128
layer_2_neurons=32
output_layer=1

model = Sequential()

model.add(Conv2D(layer_1_neurons, (3, 3), input_shape=X.shape[1:]))
model.add(Activation('elu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(rate=0.85, noise_shape=None, seed=None))


model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors

model.add(Dense(layer_2_neurons))
model.add(Activation('elu'))
model.add(Dropout(rate=0.7, noise_shape=None, seed=None))

model.add(Dense(output_layer))
model.add(Activation('sigmoid'))



model.compile(loss='binary_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])
model.fit(X, y, batch_size=12, epochs=100, validation_split=0.1,callbacks=[save])

