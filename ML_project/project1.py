import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import GridSearchCV, cross_val_score

# Load the data
churn_data = pd.read_csv(r'C:\Users\priya\OneDrive\Desktop\python\Telecomcutomer_churn_data.csv')
print(churn_data.head())

# Display the first 5 values of specific columns
customer_5 = churn_data.iloc[:, 4]
print(customer_5.head())
customer_15 = churn_data.iloc[:, 14]
print(customer_15.head())

# Filter specific groups of customers
senior_male_electronic = churn_data[(churn_data['gender'] == 'Male') & (churn_data['seniorCitizen'] == 1) & (churn_data['PaymentMethod'] == 'Electronic check')]
print(senior_male_electronic.head())

customer_tenure = churn_data[(churn_data['MonthlyCharges'] > 100) | (churn_data['tenure'] > 62)]
print(customer_tenure.head())

two_mail_yes = churn_data[(churn_data['contract'] == 'Two year') & (churn_data['PaymentMethod'] == 'Mailed check') & (churn_data['churn'] == 'Yes')]
print(two_mail_yes.head())

customer_333 = churn_data.sample(n=333)
print(customer_333.head())

# Show the distribution of the 'churn' variable
print(churn_data['churn'].value_counts())

# B. DATA VISUALIZATION

# a. Bar plot of internet service
x = churn_data['InternetService'].value_counts().keys().tolist()
y = churn_data['InternetService'].value_counts().tolist()
print(x)
print(y)
plt.bar(x, y, color="purple")
plt.xlabel('Categories of Internet Service')
plt.ylabel('Count Categories')
plt.title('Distribution of Internet Service')
plt.show()

# b. Histogram plot for tenure
tenure_values = churn_data['tenure'].value_counts().tolist()
print(tenure_values)
plt.hist(tenure_values, bins=30, color="maroon")
plt.title('Distribution of Tenure')
plt.show()

# c. Scatter plot between Monthly Charges and Tenure
tenure_values = churn_data['tenure']
monthlyCharge_values = churn_data['MonthlyCharges']
print(tenure_values.head())
print(monthlyCharge_values.head())
plt.scatter(tenure_values, monthlyCharge_values, color="brown")
plt.xlabel('Tenure')
plt.ylabel('Monthly Charges')
plt.title('Scatter Plot of Tenure vs Monthly Charges')
plt.show()

#d.box plot between contract and tenure
tenure_values_list=churn_data['tenure'].value_counts().tolist()
contract_values_list=churn_data['contract'].value_counts().tolist()
data=list([tenure_values_list,contract_values_list])
print(tenure_values_list)
print(contract_values_list)
plt.boxplot(data,showmeans=True)
plt.xlabel('tenure and contract')
plt.show()

# C. LINEAR REGRESSION
# Predict Monthly Charges based on Tenure
y = churn_data[['MonthlyCharges']].fillna(churn_data['MonthlyCharges'].median())
x = churn_data[['tenure']].fillna(churn_data['tenure'].median())
print(y.head())
print(x.head())

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)
print("X_train shape is:", x_train.shape)
print("X_test shape is:", x_test.shape)
print("y_train shape is:", y_train.shape)
print("y_test shape is:", y_test.shape)

# Train the Linear Regression model
regressor = LinearRegression()
regressor.fit(x_train, y_train)

# Predict on the test set
y_pred = regressor.predict(x_test)
print("y predicted values are:", y_pred[:5])
print("y_tested values are:", y_test[:5])

# Calculate mean squared error
mean_squarederror = np.sqrt(mean_squared_error(y_test, y_pred))
print("Mean squared error is", mean_squarederror)

# Plot actual vs predicted values
plt.scatter(x_test, y_test, color="orange", label="Actual values")
plt.plot(x_test, y_pred, color="green", label="Predicted values")
plt.title('Actual Values vs Predicted Values')
plt.xlabel('Tenure')
plt.ylabel('Monthly Charges')
plt.legend()
plt.show()

# D. LOGISTIC REGRESSION
# a. Univariate Logistic Model
x = churn_data[['MonthlyCharges']].fillna(churn_data['MonthlyCharges'].median())
y = churn_data['churn'].apply(lambda x: 1 if x == 'Yes' else 0)
print(x.head())
print(y.head())

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.35, random_state=0)

# Train the Logistic Regression model
log_model = LogisticRegression()
log_model.fit(x_train, y_train)

# Predict on the test set
y_pred = log_model.predict(x_test)
print("y predicted values are:", y_pred[:5])
print("y_tested values are:", y_test[:5])

# Evaluate the model
confusionmatrix = confusion_matrix(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)
print(confusionmatrix)
print("The accuracy percentage of logistic model is:", 100 * accuracy, "%")

# b. Multivariate Logistic Model
x = churn_data[['MonthlyCharges', 'tenure']].fillna(churn_data['MonthlyCharges'].median())
y = churn_data['churn'].apply(lambda x: 1 if x == 'Yes' else 0)
print(x.head())
print(y.head())

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.35, random_state=0)

# Train the Logistic Regression model
log_model = LogisticRegression()
log_model.fit(x_train, y_train)

# Predict on the test set
y_pred = log_model.predict(x_test)
print("y predicted values are:", y_pred[:5])
print("y_tested values are:", y_test[:5])

# Evaluate the model
confusionmatrix = confusion_matrix(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)
print(confusionmatrix)
print("The accuracy percentage of logistic model is:", 100 * accuracy, "%")

# E. DECISION TREE
x = churn_data[['tenure']].fillna(churn_data['tenure'].median())
y = churn_data['churn'].apply(lambda x: 1 if x == 'Yes' else 0)
print(x.head())
print(y.head())

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.5, random_state=1)

# Train the Decision Tree model
my_tree = DecisionTreeClassifier()
my_tree.fit(x_train, y_train)

# Predict on the test set
y_pred = my_tree.predict(x_test)
print("Y_predicted values are: ", y_pred[:5])
print("y_Actual values are:", y_test[:5])

# Evaluate the model
confusionMatrix = confusion_matrix(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)
print(confusionMatrix)
print("The percentage of accuracy of decision tree is:", 100 * accuracy, "%")

# F. RANDOM FOREST CLASSIFIER
x = churn_data[['tenure']].fillna(churn_data['tenure'].median())
y = churn_data['churn'].apply(lambda x: 1 if x == 'Yes' else 0)
print(x.head())
print(y.head())

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.5, random_state=1)

# Train the Random Forest model
my_forest = RandomForestClassifier()
my_forest.fit(x_train, y_train)

# Predict on the test set
y_pred = my_forest.predict(x_test)
print("Y_predicted values are: ", y_pred[:5])
print("y_Actual values are:", y_test[:5])

# Evaluate the model
confusionMatrix = confusion_matrix(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)
print(confusionMatrix)
print("The percentage of accuracy of Random forest is:", 100 * accuracy, "%")


