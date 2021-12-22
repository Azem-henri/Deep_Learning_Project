# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

# Data processing

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importation of the dataset

dataset = pd.read_csv("/home/ther/LAB/Deep_Learning_from_A_to_Z/ANN_Temporal_Lobe/Churn_Modelling.csv")

# Creation of the matrix of the independant variables

X = dataset.iloc[:, 3:-1].values
y = dataset.iloc[:, -1].values

# Let us transform our cathegorical data into numerical values
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
X[:, 2] = le.fit_transform(X[:, 2])

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [1])], remainder='passthrough')
X = np.array(ct.fit_transform(X))


# Let us split our dataset nouw into training and testint set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

# Scalling our variables
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)

# Let's create our model now
import keras
from keras.models import Sequential
from keras.layers import Dense
import tensorflow as tf

# Creation and initialisation of our ANN
my_model = Sequential()  # my_model = tf.keras.models.Sequential()

#Adding the input layer and the first hidden layer
my_model.add(tf.keras.layers.Dense(units=6, activation='relu'))

#Adding the second hidden layer
my_model.add(tf.keras.layers.Dense(units=6, activation='relu'))

#Adding the output layer
my_model.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))

# Compilation of our ANN 
my_model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

# Trainning of our ANN
my_model.fit(X_train, y_train, batch_size=32, epochs=100)

# Let us try new prediction for new customer with these observations :
# [France, 600, Male, 40 ans, 3 ans, 60 000 euro, 2, oui, oui, 50 000 euro]
print("He will stay ?\n")
print(my_model.predict(sc.transform([[1,0,0,600,0,40,3,60000,2,1,1,50000]])) > 0.5)

#new_prediction = np.array([[1,0,0,600,0,40,3,60000,2,1,1,50000]])

# Let's buil the confusion matrix to see the performance of our model
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

# prediction 
y_pred = my_model.predict(X_test)
y_pred = (y_pred > 0.5)
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)

accuracy_score(y_test, y_pred)

clf = SVC(random_state=0)
clf.fit(X_train, y_train)

disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=clf.classes_)
disp.plot()
plt.show()








































