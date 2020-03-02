

# # Programming Assignment 4 - Linear versus Ridge Regression 

# Your friend Bob just moved to Boston. He is a real estate agent who is trying to evaluate the prices of houses in the Boston area. He has been using a linear regression model but he wonders if he can improve his accuracy on predicting the prices for new houses. He comes to you for help as he knows that you're an expert in machine learning. 
# 
# As a pro, you suggest doing a *polynomial transformation*  to create a more flexible model, and performing ridge regression since having so many features compared to data points increases the variance. 
# 
# Bob, however, being a skeptic isn't convinced. He wants you to write a program that illustrates the difference in training and test costs for both linear and ridge regression on the same dataset. Being a good friend, you oblige and hence this assignment :) 

# In this notebook, you are to explore the effects of ridge regression.  We will use a dataset that is part of the sklearn.dataset package.  Learn more at https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_boston.html

# ## Step 1:  Getting, understanding, and preprocessing the dataset
# 
# We first import the standard libaries and some libraries that will help us scale the data and perform some "feature engineering" by transforming the data into $\Phi_2({\bf x})$




import numpy as np
import sklearn
from sklearn.datasets import load_boston
from sklearn.preprocessing import PolynomialFeatures
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import sklearn.linear_model
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# ###  Importing the dataset




# Import the boston dataset from sklearn
boston_data = load_boston()





#  Create X and Y variables - X holding the .data and Y holding .target 
X = boston_data.data
y = boston_data.target

#  Reshape Y to be a rank 2 matrix 
y = y.reshape(X.shape[0], 1)

# Proprocesing by adding a column of 1's to the front of X
one_col = np.ones((X.shape[0],1))
X = np.hstack((one_col, X))

# Observe the number of features and the number of labels
print('The number of features is: ', X.shape[1])
# Printing out the features
print('The features: ', boston_data.feature_names)
# The number of examples
print('The number of exampels in our dataset: ', X.shape[0])
#Observing the first 2 rows of the data
print(X[0:2])


# We will also create polynomial features for the dataset to test linear and ridge regression on data with degree = 1 and data with degree = 2. Feel free to increase the # of degress and see what effect it has on the training and test error. 



# Create a PolynomialFeatures object with degree = 2. 
# Transform X and save it into X_2. Simply copy Y into Y_2 
# Note: PolynomialFeatures creates a column of ones as the first feature
poly = PolynomialFeatures(degree=2)
X_2 = poly.fit_transform(X)
y_2 = y


# In[5]:


# the shape of X_2 and Y_2 - should be (506, 105) and (506, 1) respectively
print(X_2.shape)
print(y_2.shape)




# TODO - Define the get_coeff_ridge_normaleq function. Use the normal equation method.
# TODO - Return w values



def get_coeff_ridge_normaleq(X_train, y_train, alpha):
    if alpha == 0:
        w = np.dot(np.dot(np.linalg.pinv(np.dot(X_train.T, X_train)), X_train.T), y_train)
    else:  # ridge     
        N = X_train.shape[1] 
        I = np.eye(N, dtype=int)
        w = np.dot(np.dot(np.linalg.pinv(np.dot(X_train.T, X_train) + N*alpha*I), X_train.T), y_train)
    return w



# TODO - Define the evaluate_err_ridge function.
# TODO - Return the train_error and test_error values



def evaluate_err_ridge(X_train, X_test, y_train, y_test, w, alpha): 
    N = X_train.shape[0] 

    if alpha == 0:
        train_error = (1/N)*np.dot((y_train - np.dot(X_train, w)).T,(y_train - np.dot(X_train, w)))
        test_error = (1/N)*np.dot((y_test - np.dot(X_test, w)).T,(y_test - np.dot(X_test, w)))
    else :  # ridge 
        train_error = (1/N)*np.dot((y_train - np.dot(X_train, w)).T,(y_train - np.dot(X_train, w))) + alpha * np.dot(w, w.T)
        test_error = (1/N)*np.dot((y_test - np.dot(X_test, w)).T,(y_test - np.dot(X_test, w))) + alpha * np.dot(w, w.T)
    return train_error, test_error




# TODO - Finish writting the k_fold_cross_validation function. 
# TODO - Returns the average training error and average test error from the k-fold cross validation
# use Sklearns K-Folds cross-validator: https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.KFold.html

def k_fold_cross_validation(k, X, y, alpha):
    kf = KFold(n_splits=k, random_state=21, shuffle=True)
    total_E_val_test = 0
    total_E_val_train = 0
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        
        # scaling the data matrix (except for for the first column of ones)
        scaler = preprocessing.StandardScaler().fit(X_train[:,1:(X_train.shape[1]+1)])
        X_train[:,1:(X_train.shape[1]+1)] = scaler.transform(X_train[:,1:(X_train.shape[1]+1)])
        X_test[:,1:(X_train.shape[1]+1)] = scaler.transform(X_test[:,1:(X_train.shape[1]+1)])
    
        
        # determine the training error and the test error
        w = get_coeff_ridge_normaleq (X_train, y_train, alpha)
        val_train_err, val_test_err = evaluate_err_ridge(X_train, X_test, y_train, y_test, w, alpha)
        
        total_E_val_train += (val_train_err/X_train.shape[0])
        total_E_val_test += (val_test_err/X_test.shape[0])
    
    E_val_test= total_E_val_test/k
    E_val_train= total_E_val_train/k
    
    return  E_val_test, E_val_train


# ### Part 1.a



train_err, test_err = k_fold_cross_validation(10, X, y, alpha = 0)



print("Average MSE for train set is {} and for test set is {}".format(train_err[0][0],test_err[0][0]))


# ### Part 1.b




alpha_values = np.logspace(0.01, 1, num=13)

train_err_ridge = []
test_err_ridge = []
for alpha in alpha_values:
    train_err, test_err = k_fold_cross_validation(10, X, y, alpha)
    train_err_ridge.append(train_err[0][0])
    test_err_ridge.append(test_err[0][0])
    print(" For alpha = {} the train error is {} and test error is {}".format(alpha, train_err[0][0], test_err [0][0]))





plt.plot(alpha_values, test_err_ridge)
plt.ylabel('Error')
plt.xlabel('Alpha value')
plt.show()


# ### Part 1.c



alpha_values = np.logspace(-5, 1, num=10)

poly = PolynomialFeatures(degree=3)
X_3 = poly.fit_transform(X)
y_3 = y

train_err_ridge = []
test_err_ridge = []
for alpha in alpha_values:
    train_err, test_err = k_fold_cross_validation(10, X_3, y_3, alpha)
    train_err_ridge.append(train_err[0][0])
    test_err_ridge.append(test_err[0][0])
    print(" For alpha = {} the train error is {} and test error is {}".format(alpha, train_err[0][0], test_err [0][0]))


plt.plot(alpha_values, test_err_ridge)
plt.ylabel('Error')
plt.xlabel('Alpha value')
plt.show()


# ### Part 2



train_err, test_err = k_fold_cross_validation(10, X_2, y_2, alpha = 0)
print("Average MSE for train set is {} and for test set is {}".format(train_err[0][0],test_err[0][0]))


alpha_values = np.logspace(0.01, 1, num=13)

train_err_ridge = []
test_err_ridge = []
for alpha in alpha_values:
    train_err, test_err = k_fold_cross_validation(10, X_2, y_2, alpha)
    train_err_ridge.append(train_err[0][0])
    test_err_ridge.append(test_err[0][0])
    print(" For alpha = {} the train error is {} and test error is {}".format(alpha, train_err[0][0], test_err [0][0]))





plt.plot(alpha_values, test_err_ridge)
plt.ylabel('Error')
plt.xlabel('Alpha value')
plt.show()




alpha_values = np.logspace(-5, 1, num=10)

train_err_ridge = []
test_err_ridge = []
for alpha in alpha_values:
    train_err, test_err = k_fold_cross_validation(10, X_2, y_2, alpha)
    train_err_ridge.append(train_err[0][0])
    test_err_ridge.append(test_err[0][0])
    print(" For alpha = {} the train error is {} and test error is {}".format(alpha, train_err[0][0], test_err [0][0]))




plt.plot(alpha_values, test_err_ridge)
plt.ylabel('Error')
plt.xlabel('Alpha value')
plt.show()





X_test = np.array([5, 0.5, 2, 0, 4, 8, 4, 6, 2, 2, 2, 4, 5.5]).reshape(1,-1)
one_col = np.ones((X_test.shape[0],1))
X_test = np.hstack((one_col, X_test))





X_test_2 = poly.transform(X_test)
scaler = preprocessing.StandardScaler().fit(X_2[:,1:(X_2.shape[1]+1)])
X_test_2[:,1:(X_test_2.shape[1]+1)] = scaler.transform(X_test_2[:,1:(X_test_2.shape[1]+1)])





w = get_coeff_ridge_normaleq (X_2, y_2, alpha = 1e-05)




predicted_price = np.dot(X_test_2, w)




print("Predicted price of the house is {}".format(predicted_price[0]))





w.T



