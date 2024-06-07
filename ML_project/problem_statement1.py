'''you are the data scientist at a telecom company "Neo"whose customers are churning out
 to its competitors.You have to analyse the data of your company and find insights and stop 
 your customers from churning out to another telecom company '''

#A./////////////Data Manipulation
'''a. Extarct the 5th column and store it in 'customer_5
b.Extract the 15th column andb store it in customer_15
c.Extract all the male senior citizen whose payment method is electronic check and store 
the result in "senior_male_electronic"
d.Extract all the customers whose tenure is greater than 70 months or their monthly charges 
is more than 100$ and store the result in" customer total tenure"
e.Extract all the customers whose contract is of two years ,payment method mailed check and 
the value of churn is "yes" and store the result in "two_mail_yes"
f.Extract 333 random records from the customer churn dataframe and store the result in "customer_333"
g.Get the count of different levels from the "churn" column '''
#B.//////////Data visualization
#a.Build a bar plot for the 'internetsService' :
'''1. Set x-axis label to 'Categories' of internet Service
 2.Set y-axis label to 'count categories'
 3.Set the title of plot to be 'Distribution of internet service '
 4. Set the color of bar to be 'orange' '''
#b. Build a histogram for the 'tenure column'
'''1.Set the numbers of bins to be 30
2. Set the color of the to be 'green'
3. Assing the title 'Distribution of tenure' '''
#c. Bulld a scatter plot between 'monthlyCharges' and 'tenure' .map 'MonthlyCharges to the 'y-axis' and tenure to the 'x-axis' . 
'''1.Assign the points a color of 'brown' 
2.Set the x-axis label to'tenure
3.set the y-axis label to 'MonthlyCharges of customer'
4.Set the title to 'tenure vs MonthlyCharges' '''
#d. Build a box plot between 'tenure'and 'contract'.Map tenure on the y-axis and contract onthe x-axis .
#C.//////////Linear regression
#Build a simple linear model where dependent variable is  'MonthlyCharges' and  independent variable is 'tenure'.
'''1.Divide the dataset into train and test sets in 70:30 ratio.
2.Build the model on train set and predict the values on test set.
3.After predicting the values ,find root mean square error. 
4. Find out the error in prediction and store the result in 'error' 
5.Find the root mean square error. 
6.plot the graph of actual and predicted values of Monthly charges'''
#D.//////////Logistic Regression
#a.Build a simple logistic regression model where dependent variable is 'churn'and independent variable is'MonthlyCharges'
'''1.Divide the dataset in 65:35 ratio.
2.Build the model on train set and predict the values on test set.
3.Build the confusion matrix and get the accuracy score.
'''
#.Build a multiple logistic regression where dependent variable is 'churn' and independent variable is 'tenure'
'''1.Divide the dataset in 80:20 ratio
2. Build the model on train set and predict the values on train set 
3. Build the cofusion matrix and calculate the accuracy'''
#E.Decision Tree
#Build a decision tree model where dependent variable is 'churn' and independent varaible is 'tenure'
'''1.devide the dataset in 80:20 ratio
2.Build a model on train set andd predict the values on test set
3.Build the confusion matrix and calculate the accuracy'''
#F.