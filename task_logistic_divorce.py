import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Download the dataset from the source
!wget URL

# Read the data from local cloud directory
data = pd.read_csv("divorce.csv",delimiter=";")
data
# Set delimiter to semicolon(;) in case of unexpected results

# Add column which has all 1s
# The idea is that weight corresponding to this column is equal to intercept
# This way it is efficient and easier to handle the bias/intercept term
data=pd.DataFrame(data)
data.insert(loc=0, column="One", value=1, allow_duplicates=True)

# Print the dataframe rows just to see some samples
data

# Define X (input features) and y (output feature) 
X = np.array(data.iloc[0:,0:-1])
y = np.array(data.iloc[:,-1])

X_shape = X.shape
X_type  = type(X)
y_shape = y.shape
y_type  = type(y)
print(f'X: Type-{X_type}, Shape-{X_shape}')
print(f'y: Type-{y_type}, Shape-{y_shape}')

"""<strong>Expected output: </strong><br><br>
X: Type-<class 'numpy.ndarray'>, Shape-(170, 55)<br>
y: Type-<class 'numpy.ndarray'>, Shape-(170,)
"""

# Check and fill any missing values if any

# Perform standarization (if required)

# Split the dataset into training and testing here
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)

# Print the shape of features and target of training and testing: X_train, X_test, y_train, y_test
X_train_shape = X_train.shape
y_train_shape = y_train.shape
X_test_shape  = X_test.shape
y_test_shape  = y_test.shape

print(f"X_train: {X_train_shape} , y_train: {y_train_shape}")
print(f"X_test: {X_test_shape} , y_test: {y_test_shape}")
assert (X_train.shape[0]==y_train.shape[0] and X_test.shape[0]==y_test.shape[0]), "Check your splitting carefully"

"""##### Let us start implementing logistic regression from scratch. Just follow code cells, see hints if required.
##### We will build a LogisticRegression class
"""

# DO NOT EDIT ANY VARIABLE OR FUNCTION NAME(S) IN THIS CELL
# Let's try more object oriented approach this time :)
class MyLogisticRegression:
    #Write your own code

# Now initialize logitic regression implemented by you
model = MyLogisticRegression()

# And now fit on training data
model.fit(X_train,y_train)

"""##### Phew!! That's a lot of code. But you did it, congrats !!"""

# Train log-likelihood
train_log_likelihood = model.log_likelihood(y_train, model.predict_proba(X_train))
print("Log-likelihood on training data:", train_log_likelihood)

# Test log-likelihood
test_log_likelihood = model.log_likelihood(y_test, model.predict_proba(X_test))
print("Log-likelihood on testing data:", test_log_likelihood)

# Plot the loss curve
plt.plot([i+1 for i in range(len(model.likelihoods))], model.likelihoods)
plt.title("Log-Likelihood curve")
plt.xlabel("Iteration num")
plt.ylabel("Log-likelihood")
plt.show()

"""##### Let's calculate accuracy as well. Accuracy is defined simply as the rate of correct classifications."""

#Make predictions on test data
y_pred = model.predict(X_test)

def accuracy(y_true,y_pred):
    '''Compute accuracy.
    Accuracy = (Correct prediction / number of samples)
    Args:
        y_true : Truth binary values (num_examples, )
        y_pred : Predicted binary values (num_examples, )
    Returns:
        accuracy: scalar value
    '''
    
    ### START CODE HERE
    
    accuracy = np.equal(y_true,y_pred).mean()
    ### END CODE HERE
    return accuracy

# Print accuracy on train data
y_pred = model.predict(X_train)
print(accuracy(y_train,y_pred))

# Print accuracy on test data
y_pred = model.predict(X_test)
print(accuracy(y_test,y_pred))

"""## Part 1.2: Use Logistic Regression from sklearn on the same dataset
#### Tasks
- Define X and y again for sklearn Linear Regression model
- Train Logistic Regression Model on the training set (sklearn.linear_model.LogisticRegression class)
- Run the model on testing set
- Print 'accuracy' obtained on the testing dataset (sklearn.metrics.accuracy_score function)
#### Further fun (will not be evaluated)
- Compare accuracies of your model and sklearn's logistic regression model
#### Helpful links
- Classification metrics in sklearn: https://scikit-learn.org/stable/modules/classes.html#module-sklearn.metrics
"""

from sklearn.metrics import accuracy_score

# Define X and y
X = np.array(data.iloc[0:,0:-1])
y = np.array(data.iloc[:,-1])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)

# Initialize the model from sklearn
from sklearn.linear_model import LogisticRegression
model=LogisticRegression()

# Fit the model
model.fit(X_train, y_train)

# Predict on testing set X_test
y_pred = model.predict(X_test)

# Print Accuracy on testing set
test_accuracy_sklearn = accuracy_score(y_test, y_pred)

print(f"\nAccuracy on testing set: {test_accuracy_sklearn}")
