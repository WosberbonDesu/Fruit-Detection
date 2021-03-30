# -*- coding: utf-8 -*-

import numpy as np
import cv2
import os
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout, BatchNormalization 
from keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator
import pickle

# STAGE 1 ---------------------------------------------------------------------
path = 'myData'
# Opening out data path
mylist = os.listdir(path)

noOfClasses = len(mylist)

print('Label(class) num: ',noOfClasses)

images = []
classNo = []
# STAGE 2 ---------------------------------------------------------------------
for i in range(noOfClasses):
    
    myImageList = os.listdir(path+'\\'+str(i))
    
    for j in myImageList:
        # Accesing every image in our data and appending them to our list
        # cause we can split them together 
        img = cv2.imread(path + "\\" + str(i) + "\\" + j)
        img = cv2.resize(img,(32,32))
        images.append(img)
        classNo.append(i)

# let see lenght it must be 6000-6000 neither there is a problem
print(len(images))
print(len(classNo))
# STAGE 3 ---------------------------------------------------------------------

# lets make them to np array the reason why are we doing this 
# we are making our data to we do it as the machine will understand

images = np.array(images)

classNo = np.array(classNo)



print(images.shape)
print(classNo.shape)

# STAGE 4 ---------------------------------------------------------------------
# Split the Data
# I think %60 percent for train is enough we have 600 images per each class 
# so its enough 
x_train, x_test, y_train, y_test = train_test_split(images,classNo,test_size = 0.4,
                                                    random_state = 42)
# We need validation data so:
x_train, x_validation, y_train, y_validation = train_test_split(x_train,
                                                                y_train,
                                                                test_size = 0.2,
                                                                random_state = 42)


print(images.shape)
print(x_train.shape)
print(x_test.shape)
print(x_validation.shape)
# STAGE 5 ---------------------------------------------------------------------
# print(y_train.shape)
# print(y_test.shape)
# print(y_validation.shape)


fig, axes = plt.subplots(3,1,figsize = (7,7))

fig.subplots_adjust(hspace = 0.5)
sns.countplot(y_train, ax = axes[0])
axes[0].set_title("y_train")

sns.countplot(y_test, ax = axes[1])
axes[1].set_title("y_test")

sns.countplot(y_validation, ax = axes[2])
axes[2].set_title("y_validation")



#fig, axes = plt.subplots(3,1,figsize = (7,7))

#fig.subplots_adjust(hspace = 0.5)
#sns.countplot(x_train, ax = axes[0])
#axes[0].set_title("x_train")

#sns.countplot(x_test, ax = axes[1])
#axes[1].set_title("x_test")

#sns.countplot(x_validation, ax = axes[2])
#axes[2].set_title("x_validation")
# STAGE 6 ---------------------------------------------------------------------

# Preprocess

def preProcess(img):
    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.equalizeHist(img)
    img = img / 255
    
    return img


idx = 1923

img = preProcess(x_train[idx])

img = cv2.resize(img,(300,300))

cv2.imshow("Preprocess",img)

# STAGE 7 ---------------------------------------------------------------------

# We are preparing our data to be ready for training

x_train = np.array(list(map(preProcess,x_train)))
x_test = np.array(list(map(preProcess,x_test)))
x_validation = np.array(list(map(preProcess,x_validation)))


x_train = x_train.reshape(-1,32,32,1)
print(x_train.shape)

x_test = x_test.reshape(-1,32,32,1)
print(x_test.shape)

x_validation = x_validation.reshape(-1,32,32,1)
print(x_validation.shape)

# STAGE 8 ---------------------------------------------------------------------

# Data Generate

# you can increase zoom_range i thought this enough but if you increase
# model will be much better
dataGen = ImageDataGenerator(width_shift_range = 0.1,
                             height_shift_range = 0.1,
                             zoom_range = 0.3,
                             rotation_range = 10)


# lets fit and convert to categorical then we can easily see our model 
# will be 0 or 1
dataGen.fit(x_train)

y_train = to_categorical(y_train,noOfClasses)
y_test = to_categorical(y_test,noOfClasses)
y_validation = to_categorical(y_validation,noOfClasses)

print(y_train)
print(y_test)
print(y_validation)


# STAGE 9 ---------------------------------------------------------------------

# Building Model

# Fundemental of model
model = Sequential()

model.add(Conv2D(input_shape = (32,32,1),filters = 8,kernel_size = (5,5),
                  activation = 'relu',padding = 'same')

model.add(MaxPooling2D(pool_size= (2,2)))


model.add(Conv2D(filters = 16,kernel_size = (3,3),
                  activation = 'relu',padding = 'same')

model.add(MaxPooling2D(pool_size= (2,2)))


model.add(Dropout(0.2))
model.add(Flatten())

model.add(Dense(units = 256,activation = 'relu'))
model.add(Dropout(0.2))
model.add(Dense(units = 256,activation = 'relu'))
model.add(Dropout(0.2))
#model.add(Dense(units = 256,activation = 'relu'))
#model.add(Dropout(0.2))
#model.add(Dense(units = 256,activation = 'relu'))
#model.add(Dropout(0.2))
model.add(Dense(units = noOfClasses,activation = 'softmax'))


model.compile(loss = 'categorical_crossentropy',optimizer = ('Adam'),
              metrics = ['accuracy']) 


batch_size = 250


# STAGE 10 --------------------------------------------------------------------


hist = model.fit_generator(dataGen.flow(x_train,y_train,batch_size=batch_size),
                                        validation_data = (x_validation,
                                                           y_validation),
                                        epochs = 17,
                                        steps_per_epoch = x_train.shape[0]//batch_size,
                                        shuffle = 1)


pickle_out = open('model_trained_new.p','wb')

pickle.dump(model,pickle_out)

pickle_out.close()


# Evaluation
hist.history.keys()

plt.figure()

plt.plot(hist.history['loss'], label = 'Training Loss')
plt.plot(hist.history['val_loss'], label = 'Validation Loss')
plt.legend()
plt.show()


plt.figure()

plt.plot(hist.history['accuracy'], label = 'accuracy Loss')
plt.plot(hist.history['val_accuracy'], label = 'val_accuracy Loss')
plt.legend()
plt.show()


plt.figure()

plt.plot(hist.history['loss'], label = 'Training Loss')
plt.plot(hist.history['val_loss'], label = 'Validation Loss')
plt.plot(hist.history['accuracy'], label = 'accuracy Loss')
plt.plot(hist.history['val_accuracy'], label = 'val_accuracy Loss')
plt.legend()
plt.show()

score = model.evaluate(x_test,y_test,verbose = 1)

print('Test Loss: ',score[0])
print('Accuracy Loss: ',score[1])


from sklearn.metrics import confusion_matrix

y_predic = model.predict(x_validation)


y_predic_class = np.argmax(y_predic,axis = 1)

y_true = np.argmax(y_validation,axis = 1)


cm = confusion_matrix(y_true,y_predic_class)


f, ax = plt.subplots(figsize = (8,8))
sns.heatmap(cm,annot=True,linewidths=0.01,cmap = 'Greens',
            linecolor = "gray",fmt = '.1f',ax = ax)
plt.xlabel("Predicted")
plt.ylabel("true")
plt.title("Confusion Matrix")
plt.show()
