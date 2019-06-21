# -*- coding: utf-8 -*-
"""
Created on Thu Jun 20 23:44:27 2019

@author: vinayver
"""

# Importing the packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D,MaxPooling2D,Flatten,Dense

# Import the datasets
train_dataset = pd.read_csv('train.csv')
test_dataset = pd.read_csv('test.csv')


# Lets check that if dataset contains any null values and balancing of labels
train_dataset.isnull().any().describe()
test_dataset.isnull().any().describe()
sns.countplot(train_dataset['label'])

# Creating feature set and target variable from train and test
X_train = train_dataset.iloc[:,1:]
Y_train =  train_dataset.iloc[:,0]

X_test = test_dataset.copy()

# Let's normalize the X_train and Y_train since values ranges from 0 to 255
X_train/=255
X_test/=255

# reshape
X_train = X_train.values.reshape(-1,28,28,1)
X_test = X_test.values.reshape(-1,28,28,1)


# Convert the Y_test to labels
Y_train = to_categorical(Y_train)


# creating the model
model = Sequential()
model.add(Conv2D(filters=32,kernel_size=(5,5),input_shape=(28,28,1),activation='relu',padding='same'))
model.add(MaxPooling2D(2,2))
model.add(Conv2D(filters=64,kernel_size=(5,5),input_shape=(28,28,1),activation='relu',padding='same'))
model.add(MaxPooling2D(2,2))
model.add(Conv2D(filters=64,kernel_size=(5,5),input_shape=(28,28,1),activation='relu',padding='same'))
model.add(Flatten())
model.add(Dense(units=64,activation='relu'))
model.add(Dense(units=10,activation='softmax'))

# compiling the model
model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])

# summarizing the model
model.summary()

# Fitting the model
model.fit(X_train,Y_train,epochs=10,batch_size=128,validation_split=0.3)

# Testing the model
y_pred = model.predict(X_test)

Labels =  np.argmax(y_pred,axis=1)
ImageId = list(range(1,28001))

submission_df = pd.DataFrame({'ImageId':ImageId,'Labels':Labels})
submission_df.to_csv('MySubmission.csv',index=False,columns=['ImageId','Labels'])

