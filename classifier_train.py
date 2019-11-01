import numpy as np
import matplotlib.pyplot as plt
#plt.rcParams.update({'font.size': 14})
from keras.models import Sequential #,model_from_json

#from keras import optimizers
from keras.layers import Dense,Dropout,Flatten
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Convolution2D,MaxPooling2D#,ZeroPadding2D
from keras.layers.core import Activation
from keras.callbacks import ModelCheckpoint
from keras import backend as K
K.set_image_data_format('channels_last')

seed =7
np.random.seed(seed)
num_classes = 4


npzfile = 'C:/Users/salehpc/Desktop/ws/Lec 6/Labels/labels.npz'

dataset =  np.load(npzfile)
x_train = dataset['X_train']
y_train = dataset['Y_train']

x = x_train
#x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
'''
def CNN_model():
    # create model
    model = Sequential()
    model.add(Convolution2D(16, (2,2), input_shape=(96,96,3), activation='relu',padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    #model.add(Dropout(0.20))
    model.add(Convolution2D(32, (3,3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(3, 3)))
    #model.add(Dropout(0.20))
    model.add(Convolution2D(64, (5,5), activation='relu',padding='same'))
    model.add(MaxPooling2D(pool_size=(5, 5), strides=(5, 5)))
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.25))
    model.add(Dense(num_classes, activation='softmax'))
    #sgd = optimizers.SGD(lr=1e-2)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model
'''

def CNN_model():
    model = Sequential()
    model.add(Convolution2D(32, kernel_size=(3,3),input_shape=(128,128,3),padding='same' ))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=-1))
    #model.add(ZeroPadding2D((1,1),input_shape=(28,28,3)))
    model.add(MaxPooling2D(pool_size=(3,3)))
    #model.add(Dropout(0.20))
    #model.add(ZeroPadding2D((1,1),input_shape=(28,28,3)))
    
    model.add(Convolution2D(64, kernel_size = (3, 3),padding='same'))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=-1))
    
    model.add(Convolution2D(64, kernel_size = (3, 3),padding='same'))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=-1))
    
    model.add(MaxPooling2D(pool_size=(2, 2)))
    #model.add(Dropout(0.25))
    '''
    model.add(Convolution2D(128,kernel_size = (3, 3), padding="same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=-1))
    #model.add(Convolution2D(128, kernel_size =(3, 3), padding="same"))
    #model.add(Activation("relu"))
    #model.add(BatchNormalization(axis=-1))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    
    model.add(MaxPooling2D(pool_size=(2, 2)))
    #model.add(Dropout(0.20))
    #model.add(Convolution2D(32, kernel_size = (5, 5), activation = 'relu',padding='same'))
    #model.add(ZeroPadding2D((1,1),input_shape=(28,28,3)))
    #model.add(MaxPooling2D(pool_size = (2, 2)))
    '''
    model.add(Flatten())
    model.add(Dense(128,activation='relu'))
    #model.add(BatchNormalization())
    #model.add(Dropout(0.5))
    #model.add(Dropout(0.20))
    model.add(Dense(num_classes, activation = 'softmax'))
    model.compile(loss= 'categorical_crossentropy' , optimizer= 'adam' , metrics=[ 'accuracy' ])
    return model

model = CNN_model()

model.summary()

check  = ModelCheckpoint('best.hdf5', monitor = 'val_categorical_accuracy')
checkpoints = [check]

history=model.fit(x, y_train,validation_split=0.20,nb_epoch=10, batch_size=16,verbose=1, callbacks = checkpoints)

model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("best.hdf5")
print("Saved model to disk")

print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper right')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper right')
plt.show()
