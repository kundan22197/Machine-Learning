
# # Salmon classification with the bivariate Gaussian
# 
# In this assigment, you will predict if a fish is an 'Alaskan' salmon or a 'Canadian' salmon.
# 
# The algorithm you will use a generative algorithm.  Where you model each class as a **bivariate Gaussian**.

# ## Step 0. Import statements
# 
# The Python programming language, as most programming languages, is augmented by **modules**.  These modules contain functions and classes for specialized tasks needed in machine learning.
# 
# Below, we will `import` three modules:
# * **pandas**
# * **numpy**
# * **matplotlib.pyplot**
# 
# Note that we imported these modules using **aliases**



# Standard libraries
get_ipython().run_line_magic('matplotlib', 'inline')
import pandas as pd
import numpy as np # for better array operations
import matplotlib.pyplot as plt # plotting utilities 

# Module computing the Gaussian density
from scipy.stats import norm, multivariate_normal 


# ## Step 1. Data preparation: loading, understanding and cleaning the dataset

# ### Importing the dataset
# Make sure the file `Salmon_dataset.csv` is in the same directory as this notebook.
# 
# The dataset contains 100  examples, each example has 3 features (*gender, Freshwater, marine*) and a label (*Alaskan, Canadian*).




# Loading the data set using Panda's in a dataframe 

df = pd.read_csv('Salmon_dataset.csv', delimiter=',') 

#Lets check that everything worked before continuing on
df.head()


# ### Data preprocesssing
# We will change the labels 'Alaskan' and 'Canadian' to $0$ and $1$ respectively.  In our code it is easier to work with numerical values instead of strings.
# 
# Often we will do more dataprepocessing, such as looking for missing values and scaling the data though that is NOT required for this assignment yet. 




# It is easier to work with the data if the labels are integers
# Changing the 'Origin' column values, map 'Alaskan':0 and 'Canadian':1
df['Origin']=df.Origin.map({'Alaskan':0, 'Canadian':1})

#Lets check that everything worked before continuing on
df.head()




# We will store the dataframe as a Numpy array
data = df.to_numpy() 

# Split the examples into a training set (trainx, trainy) and test set (testx, testy) 

n =  data.shape[0] # the number of rows
train_n = int(.9*n) # this test set is a bit small to really evaluate our hypothesis - what could we do to get a better estimate and still keep most of the data to estimate our parameters?
np.random.seed(0) # Our code randomly chooses which examples will be the training data, but for grading purposes we want the random numbers used to seperate the data are the same for everyone
perm = np.random.permutation(n)
trainx = data[perm[0:train_n],1:3] #selecting the two of the features `Freshwater' and 'Marine'
trainy = data[perm[0:train_n],3]
testx = data[perm[train_n:n], 1:3] # We won't look at the testx data until it is time to evauate our hypothesis.  This numpy array contains the set of test data for the assignment
testy = data[perm[train_n:n],3]



# ### Plotting the dataset
# Visualization can be helpful when exploring and getting to know a dataset.



# plotting the Alaskan salmon as a green dot
plt.plot(trainx[trainy==0,0], trainx[trainy==0,1], marker='o', ls='None', c='g')
# plotting the Canadian salmon as a red dot
plt.plot(trainx[trainy==1,0], trainx[trainy==1,1], marker='o', ls='None', c='r')


# ## Step 2. Model training: implementing Gaussian Discriminant Analysis
# 
# 
# 

# ###  Sufficient statistics
# 
# Just as if we were doing these calculations by hand, we break the process down into managable pieces
# 
# Our first helper function will find the mean and covariance of the Gaussian for a set of examples




# Input: a design matrix
# Output: a numpy array containing the means for each feature, and a 2-dimensional numpy array containng the covariance matrix sigma
def fit_gaussian(x):
    fresh = x[0: ,0] # storing freshwater values in an array
    marine = x[0: ,1] # storing marine values in an array
    
    mu1 = np.mean(fresh) # finding mean of the "fresh" array
    mu2 = np.mean(marine) # finding mean of the "marine" array
    
    mu = np.array([mu1, mu2]) # numpy array containing the two means
    
    sigma = np.cov(x[0: ,0], x[0: ,1]) # 2D array containing covariance
    
    return mu, sigma




alaskan = trainx[trainy == 0] # All the alaskan species(with origin = 0) are stored in this array
canadian = trainx[trainy == 1] # All the canadian species(with origin = 1) are stored in this array





a_mu, a_sigma = fit_gaussian(alaskan) # The model parameters(mu and sigma) for alaskan species are estimated



c_mu, c_sigma = fit_gaussian(canadian) # The model parameters(mu and sigma) for canadian species are estimated




def predict(x): # function for predicting class of new fishes
    prediction = []
    for i in range(10):
        a = multivariate_normal.pdf(x[i], a_mu, a_sigma) # probability of fish to belong to alaskan species
        b = multivariate_normal.pdf(x[i], c_mu, c_sigma) # probability of fish to belong to canadian species
        if a > b:
            prediction.append(0) # if probability of alaskan species is higher, append '0' and the fish is alaskan
        else:
            prediction.append(1) # if probability of canadian species is higher, append '1' and the fish is canadian
    return prediction




out = predict(testx) # pass the test values for predicting the class of fishes




with open('output.txt', 'w') as f:     # Writing the outputs to an external text file
    f.write("mu0: " + str(a_mu) + "\n")
    f.write("sigma0: " + str(a_sigma) + "\n")
    f.write("mu1: " + str(c_mu) + "\n")
    f.write("sigma1: " + str(c_sigma) + "\n")
    f.write("Predicted classes of the fishes(0 - Alaska, 1 - Canada): " + str(out) + "\n")

