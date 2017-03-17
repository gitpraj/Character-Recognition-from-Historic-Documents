import numpy as np
np.random.seed(1337)  # for reproducibility
#import tensorflow as tf
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras import backend as K

#setting batchsize and epoch
batch_size = 64
nb_classes = 98
nb_epoch = 20
nb_data=50282
img_rows, img_cols = 13, 33

#Load training data from train.csv
train_file = open("train.csv")
train_data =  np.loadtxt(fname = train_file, delimiter = ',')
x=list(train_data)
conv_traindata=np.array(x).astype('int')

#Checking the imbalance in data
def zerolistmaker(n):
    listofzeros = [0] * n
    return listofzeros

count = zerolistmaker(nb_classes)
for class_ind in range(nb_classes):
    for img_index in range(nb_data):
        if conv_traindata[img_index,1] == class_ind:
            count[class_ind] = count[class_ind] + 1

# perfect_classes.csv has best instances for each class based on 
# visual representation of the instance
h = open("perfect_classes.csv")
perfect_class =  np.loadtxt(fname = h, delimiter = ',')
p_class = perfect_class.astype("int")


for class_ind in range(nb_classes):
    for img_index in range(nb_data):
        if train_data[img_index,1] == class_ind:
            if (train_data[img_index, 0] == p_class[class_ind]) and (p_class[class_ind] != -1):
                while ((count[class_ind] < 1000) and (count[class_ind] != 0)):
                    x_new = train_data[img_index, 0:438].astype("int")
                    x.append(x_new)
                    count[class_ind] = count[class_ind] + 1
                break    
new_traindata=np.array(x).astype("int")
y_train = new_traindata[:, 1].astype("int")
X_train = new_traindata[:, 9:438].astype("int")


#Load test data from test.csv
test_file = open("test.csv")
test_data =  np.loadtxt(fname = test_file, delimiter = ',')
X_test = test_data[:, 9:438].astype("int")
y_test = test_data[:, 1].astype("int")

#Reshaping train and test data
X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 1)
X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)
Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)

# Sequential wrapper model
model = Sequential()

#Input Layer
model.add(Convolution2D(32, 6, 6, border_mode='same', input_shape=(img_rows, img_cols, 1), activation='relu'))

#First hidden convolutional layer
model.add(Convolution2D(32, 3,3, activation='relu', border_mode='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))

#Second hidden convolutional layer
model.add(Convolution2D(64, 3, 3, activation='relu', border_mode='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))

#Third hidden convolutional layer
model.add(Convolution2D(128, 3,3, activation='relu', border_mode='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))


# Flattens the input
model.add(Flatten())

#First fully connected NN layer 
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.25))

#Second fully connected NN layer
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.25))

#Last fully connected N layer 
model.add(Dense(nb_classes, activation='softmax'))

#Configure the model for training
model.compile(loss='categorical_crossentropy', optimizer="adadelta", metrics=['accuracy'])

#Trains the model for a fixed number of epochs
model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch, verbose=1)

#Predict the classes
classes = model.predict_classes(X_test)
id = test_data[:, 0].astype("int")

#Save the predicted classes into file
np.savetxt("submission.csv", np.column_stack((id, classes)), delimiter=",", fmt='%s', header="Id,Character")

