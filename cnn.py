# Classify Images Using Convolutional Neural Networks & Python
# link to article https://medium.com/@randerson112358/classify-images-using-convolutional-neural-networks-python-a89cecc8c679
# link to video https://www.youtube.com/watch?time_continue=4&v=iGWbqhdjf2s&feature=emb_title

# =============================================================================
# The purpose of this file is to give you a working example of a Convolutional 
# Neural Netwrk for image recognition. Your task, should you chose to accept it
# is to identify the componets of each code block and document it aong with the 
# location of a reference that shows you ahve to modify the inputs along with 
# the anticpated results of the change(s). This is a method teaching yourself 
# how to learn something completely new. P.O.C. abenson.ctr@kr.af.mil for any
# questions. Have fun! Really, have fun.

import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout
from tensorflow.keras import layers
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')
print("libraries loaded")

current_path = os.getcwd()
print("\n= + = + = + = + = + = + = + = + = + = + = + = + = + = + = + = + = + \n"
    "You are curently in {}. \n"
    "You can add a test file or directory of test files here, or change\n"
    "the working directory to refence where the code can locate your \n"
    "test material.\n"
    "\n".format(current_path))

# I modified my working directoy to point to the on where I am keeping this 
# module. I find it makes it easier to share with others. 
project_path = "".join([current_path,"/Toolbox/machine_learning_nerual_networks/CNNs"])
os.chdir(project_path)
print("you are now working in {}".format(project_path))


#Load the data
from tensorflow.keras.datasets import cifar10
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
print("data set loaded")

#Print the data type of x_train
print(type(x_train))
#Print the data type of y_train
print(type(y_train))
#Print the data type of x_test
print(type(x_test))
#Print the data type of y_test
print(type(y_test))
print("data set type identified")

#Get the shape of x_train
print("x_train shape: {}".format(x_train.shape))
#Get the shape of y_train
print("y_train shape: {}".format(y_train.shape))
#Get the shape of x_train
print("x_test shape: {}".format(x_test.shape))
#Get the shape of y_train
print("y_test shape: {}".format(y_test.shape))
print("shape of data listed")


index = 0
print("{}".format(x_train[index]))
print("first element shown")


img = plt.imshow(x_train[index])
plt.show()
print("image shown") # close the image dialog to continue the code


print('The image label is: ', y_train[index])
print("image identified")


classification = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
print('The image class is: ', classification[y_train[index][0]])
print("string for image given")


y_train_one_hot = to_categorical(y_train)
y_test_one_hot = to_categorical(y_test)
print(y_train_one_hot)


x_train = x_train / 255
x_test = x_test / 255


model = Sequential()


model.add(Conv2D(32, (5, 5), activation='relu', input_shape=(32,32,3)))
# Conv2D

model.add(MaxPooling2D(pool_size=(2, 2)))


model.add(Conv2D(64, (5, 5), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))


model.add(Flatten())


model.add(Dense(1000, activation='relu'))


model.add(Dropout(0.5))


model.add(Dense(500, activation='relu'))


model.add(Dropout(0.5))


model.add(Dense(250, activation='relu'))


model.add(Dense(10, activation='softmax'))


model = Sequential()
model.add(Conv2D(32, (5, 5), activation='relu', input_shape=(32,32,3)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, (5, 5), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(1000, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(500, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(250, activation='relu'))
model.add(Dense(10, activation='softmax'))


model.compile(loss='categorical_crossentropy', 
              optimizer='adam',
              metrics=['accuracy'])
# model.compile(): ...


hist = model.fit(x_train, y_train_one_hot, 
           batch_size=256, epochs=10, validation_split=0.2 )
# NOTE TO SELF: should pickle hist here to speed things along, saving ~5 minutes

model.evaluate(x_test, y_test_one_hot)[1]


#Visualize the models accuracy
# img = plt.plot(hist.history['accuracy'])
img = plt.plot(hist.history['acc'])
plt.plot(hist.history['val_acc'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc='upper left')
plt.show()

#Visualize the models loss
plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc='upper right')
plt.show()

# =============================================================================
## !!!! GOT THIS FAR DURING VALIDATION !!!!
# =============================================================================


# You will need to get an image from something other than the training data.
# You will find a selection to chose from at this GitHub repo 
# https://github.com/cunhamaicon/catsxdogs I downloaded the repo. For the
# first runthrough, it's probably a good idea to select a specific file then 
# load the data. Later, for fun, you might use a random file selector to keep
# it honest.

new_image = plt.imread("".join([project_path, "/catsxdogs/single_prediction/floyd3.jpg"]))
img = plt.imshow(new_image)


from skimage.transform import resize
resized_image = resize(new_image, (32,32,3))
img = plt.imshow(resized_image)



predictions = model.predict(np.array( [resized_image] ))


print("{}".format(predictions))


list_index = [0,1,2,3,4,5,6,7,8,9]
x = predictions
for i in range(10):
  for j in range(10):
    if x[0][list_index[i]] > x[0][list_index[j]]:
      temp = list_index[i]
      list_index[i] = list_index[j]
      list_index[j] = temp
#Show the sorted labels in order from highest probability to lowest
print(list_index)


i=0
for i in range(5):
  print(classification[list_index[i]], ':', round(predictions[0][list_index[i]] * 100, 2), '%')


model.save('my_model.h5')


#To load this model
model = load_model('my_model.h5')




print("that's all folks")