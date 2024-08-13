# Predict whether the person is suffering from diabeties or not according to given dataset

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score

#Data Collection and Analysis
diabetes_dataset=pd.read_csv(r'C:\Users\priya\OneDrive\Desktop\ML_projects\ML_project\Diabetes_Dataset.csv')
print(diabetes_dataset.head()) # Getting 5 rows data of dataset
print(diabetes_dataset.shape) #Getting the Rows and columns of dataset
print(diabetes_dataset.describe) #Getting the statistical data
print(diabetes_dataset['Outcome'].value_counts()) #count the frequency of values of outcome column
print(diabetes_dataset.groupby('Outcome').mean()) #Find the mean of each column according to the group of outcome
x= diabetes_dataset.drop(columns='Outcome',axis=1) # Dataframe of dataset which doesn't comtain outcome column
y= diabetes_dataset['Outcome'] # dataframe that contain only outcome column
print(x,y) 

scaler =StandardScaler()
'''The StandardScaler is a tool that standardizes features by removing the mean and scaling to unit variance.
This means that it transforms the data so that it has a mean of 0 and a standard deviation of 1'''
scaler.fit(x) #find mean and varience for each features
standard_data=scaler.transform(x) #Transform the data into a standard form
print(standard_data)
x= standard_data
y=diabetes_dataset['Outcome']

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)
print("X_train shape is:", x_train.shape)
print("X_test shape is:", x_test.shape)
print("y_train shape is:", y_train.shape)
print("y_test shape is:", y_test.shape)

#Training the model
classifier=svm.SVC(kernel='linear')
classifier.fit(x_train,y_train)
#check the accuracy score on the training dataset
x_train_prediction=classifier.predict(x_train)
train_data_accuracy=accuracy_score(x_train_prediction,y_train)
#print("Accuracy of the trained dataset: ",train_data_accuracy*100,"%")

#check the accuracy score on the training dataset
x_test_prediction=classifier.predict(x_test)
test_data_accuracy=accuracy_score(x_test_prediction,y_test)
#print("Accuracy of the testing dataset: ",train_data_accuracy*100,"%")

#Making a predictive system using randomly selecting a patient
random_patient = diabetes_dataset.sample(n=1)
print("Random Patient Data:")
print(random_patient)
# Extracting the input features for the selected patient
input_data = random_patient.drop(columns='Outcome', axis=1).values
input_data_scaled = scaler.transform(input_data)
# Making a prediction
prediction = classifier.predict(input_data_scaled)
print("Model Prediction: ", "Diabetic" if prediction[0] == 1 else "Not Diabetic")
# Actual status
actual_status = "Diabetic" if random_patient['Outcome'].values[0] == 1 else "Not Diabetic"
print("Actual Status: ", actual_status)
