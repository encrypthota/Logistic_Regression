import the necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Sklearn processing
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

# Sklearn linear regression model
from sklearn.linear_model import LinearRegression

# Sklearn regression model evaluation functions
from sklearn.metrics import r2_score

# Perform feature selection using a variance threshold
from sklearn.feature_selection import VarianceThreshold

# Feature selection using Recursive Feature Elimimation
from sklearn.feature_selection import RFE

#load the data and inspect the first 5 rows
!wget https://github.com/DeepConnectAI/challenge-week-6/raw/master/data/bike.csv
data = pd.read_csv('bike.csv')

# print the data types of each feature name
data.dtypes

data.head()
# check for null values in each column

# print out the unique values of the features ['season', 'year', 'weather', 'promotion_type']
print(set(data['season']))
print(set(data['year']))
print(set(data['weather']))
print(set(data['promotion_type']))

# print out the value counts (frequency of occurence) of the unique values in these features ['season', 'year', 'weather', 'promotion_type']
print(data['season'].value_counts())
print(data['year'].value_counts())
print(data['weather'].value_counts())
print(data['promotion_type'].value_counts())

# print the shape of data
data.shape

# drop the feature 'id' as it has no information to deliver.

#data = data.drop('year', axis = 1)
data = data.drop('id', axis = 1)

# print the shape of data
data.shape

# one hot encode the categorical columns.
t= pd.get_dummies(data.weather, prefix='weather')
data = pd.concat([t, data], axis=1)
data=data.drop(['weather'],axis=1)
t= pd.get_dummies(data.season, prefix='season')
data = pd.concat([t, data], axis=1)
data=data.drop(['season'],axis=1)

# print the shape of data 
# notice the increase in the no. of features
data.shape
data.head()

"""Notice that our target feature "cnt" is the sum of the features 
---
"registered" + "casual"<br>
To avoid data leakage remove the feature "casual" for the training purpose. <br>
To understand more about data leakage refer the article mentioned in the uselful links.
"""

# Split the dataset into X and y
# While loading data into X drop the columns "cnt" and "casual". 
X = data.iloc[:,:-1]
X=X.drop(['casual'],axis=1)

# notice the target variable is 'cnt'
y = data['cnt']

# store the names of the training features / name of the columns used for training. [Very important step for visualization later.]

train_columns = list(X.columns)
print(train_columns)

# Apply scaling if our data is spread across wide differences of range values.
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, MinMaxScaler
x = X.values #returns a numpy array
scaler = MinMaxScaler()
X[train_columns] = scaler.fit_transform(X[train_columns])

X.head()

# print the type of X
type(X)

"""Note : <br>
Type of X should be pandas dataframe.
If not then convert X into pandas DataFrame object before proceeding further.
"""

# convert X into pandas Dataframe
# in the parameters specify columns = train_columns.

X = pd.DataFrame(X, columns = train_columns)
X.head()

# split the dataset into X_train, X_test, y_train, y_test
# play around with test sizes.

test_size = 0.2
X_train, X_test, y_train, y_test = train_test_split(X,y,train_size=0.80,test_size=0.20,random_state=42)

# print the shapes
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

# build the Linear Regression model.
model = LinearRegression()

# fit the model on the training data
model.fit(X_train,y_train)

# print the score on training set
y_pred_train = model.predict(X_train)
print("On Training set : ", r2_score(y_train, y_pred_train))

# print the score on the test set
y_pred_test = model.predict(X_test)
print("On testing set : ", r2_score(y_test,y_pred_test))

"""Do not edit the code given below. Observe the distribution of weights. 
Which feature has the maximum coefficient ? <br>
Keep this figure as a base reference for visualizing the effects of l1-norm and l2-norm later in this notebook.
"""

# custom summary function to plot the coefficients / weightage of the features.
def custom_summary(model, column_names, title):
    '''Show a summary of the trained linear regression model'''

    # Plot the coeffients as bars
    fig = plt.figure(figsize=(8,len(column_names)/3))
    fig.suptitle(title, fontsize=16)
    rects = plt.barh(column_names, model.coef_,color="lightblue")

    # Annotate the bars with the coefficient values
    for rect in rects:
        width = round(rect.get_width(),4)
        plt.gca().annotate('  {}  '.format(width),
                    xy=(0, rect.get_y()),
                    xytext=(0,2),  
                    textcoords="offset points",  
                    ha='left' if width<0 else 'right', va='bottom')        
    plt.show()

# coefficients plot
# let's call the above custom function.
custom_summary(model, train_columns, "Linear Regression coefficients.")

X_test.head()

# evaluate the model with k = 10 Fold Cross validation

folds = KFold(n_splits = 10, shuffle = True, random_state = 100)
results = cross_val_score(model, X, y, scoring = 'r2', cv = folds)

print(type(model)._name_)
print("kFoldCV:")
print("Fold R2 scores:", results)
print("Mean R2 score:", results.mean())
print("Std R2 score:", results.std())
print("Generalizability on training set : ", results.mean(), " +/- ", results.std())

"""Feature Selection using Variance Thresholding"""

print("Original shape of X_train : ", X_train.shape)
