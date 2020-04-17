#1-Import Libraries
from keras.models import Sequential
from keras.layers import Dense,Dropout
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical
import numpy as np
import pandas as pd

#2-set up Hyper-parameters
classes = 2 #the number of possible classifications (Benign or Malignant)
batch_size = 2 #the number of samples that will be used to train the network at one time
input_shape = (9,) # the number of inputs 

#3-Data Extraction and Processing
df = pd.read_csv('breast_cancer.csv')
df = df.replace({'?': np.nan}).dropna()#Remove faulty data 
#Split the data into training and testing data
test, train = train_test_split(df, test_size = 0.8)
x_train, y_train = train.ix[:,0:9].as_matrix(), train.ix[:,9].as_matrix()
x_test, y_test = test.ix[:,0:9].as_matrix(), test.ix[:,9].as_matrix()
y_train = to_categorical(y_train, classes)
y_test = to_categorical(y_test, classes)

#4-Model Creation
model = Sequential()
model.add(Dense(128, activation = 'relu', input_shape = input_shape))
model.add(Dropout(.2))
model.add(Dense(classes, activation = 'softmax'))
#compile the model
model.compile(optimizer = 'sgd', loss = 'binary_crossentropy', metrics = ['accuracy'])

#training the model
model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=100,
              verbose=1,
              validation_data=(x_test, y_test))

#Testing our model
inputs = np.array([[3,10,7,8,5,8,7,4,1]])
prediction = model.predict(inputs)
print ("Probability of Benign: " + str(prediction[0][0]))
print ("Probability of Malignancy: " + str(prediction[0][1]))


