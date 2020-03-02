

# # Using Naive Bayes algorithm for spam detection
# 
# In this assigment, you will predict if a sms message is 'spam' or 'ham' (i.e. not 'spam') using the Bernoulli Naive Bayes *classifier*.
# 
# The training data is from the UCI Machine Learning Repository: https://archive.ics.uci.edu/ml/datasets/SMS+Spam+Collection.  Please download the zip file before running the code below. 
# 

# ## Step 1:  Getting, understanding, and cleaning the dataset
# 

# ###  Importing the dataset


# Import the usual libraries
import numpy as np 
import pandas as pd  # To read in the dataset we will use the Panda's library
df = pd.read_table('SMSSpamCollection', sep = '\t', header=None, names=['label', 'sms_message'])

# Next we observe the first 5 rows of the data to ensure everything was read correctly
df.head() 


# ### Data preprocesssing
# It would be more convenient if the labels were binary instead of 'ham' and 'spam'.  This way our code can always work with numerical values instead of strings.



df['label']=df.label.map({'spam':1, 'ham':0})
df.head() # Again, lets observe the first 5 rows to make sure everything worked before we continue


# ### Splitting the dcoument into training set and test set




# This time we will use sklearn's method for seperating the data
from sklearn.model_selection import train_test_split

df_train_msgs, df_test_msgs, df_ytrain, df_ytest = train_test_split(df['sms_message'],df['label'], random_state=0)

#Looking at the train/test split
print("The number of training examples: ", df_train_msgs.shape[0])
print("The number of test exampels: ", df_test_msgs.shape)

print("The first four labels")
print(df_ytrain[0:4])

print("The first four sms messages")
print(df_train_msgs[0:4])


# ###  Creating the feature vector from the text (feature extraction)
# 
# Each message will have its own feature vector.  For each message we will create its feature vector as we discussed in class; we will have a feature for every word in our vocabulary.  The $j$th feature is set to one ($x_j=1$) if the $j$th word from our vocabulary occurs in the message, and set the $j$ feature is set to $0$ otherwise ($x_j=0$).
# 
# We will use the sklearn method CountVectorize to create the feature vectors for every messge.
# 
# We could have written the code to do this directly by performing the following steps:
# * remove capitalization
# * remove punctuation 
# * tokenize (i.e. split the document into individual words)
# * count frequencies of each token 
# * remove 'stop words' (these are words that will not help us predict since they occur in most documents, e.g. 'a', 'and', 'the', 'him', 'is' ...


# importing the library
from sklearn.feature_extraction.text import CountVectorizer
# creating an instance of CountVectorizer
# Note there are issues with the way CountVectorizer removes stop words.  To learn more: https://scikit-learn.org/stable/modules/feature_extraction.html#stop-words
vectorizer = CountVectorizer(binary = True, stop_words='english')

# If we wanted to perform Multnomial Naive Bayes, we would include the word counts and use the following code
#vectorizer = CountVectorizer(stop_words='english')

# To see the 'count_vector' object
print(vectorizer)



# Create the vocabulary for our feature transformation
vectorizer.fit(df_train_msgs)

# Next we create the feature vectors for both the training data and the test data
X_train = vectorizer.transform(df_train_msgs).toarray() # code to turn the training emails into a feature vector
X_test = vectorizer.transform(df_test_msgs).toarray() # code to turn the test email into a feature vector

# Changing the target vectors data type  
y_train=df_ytrain.to_numpy() # Convereting from a Panda series to a numpy array
y_test = df_ytest.to_numpy()

# To observe what the data looks like 
print("The label of the first training example: ", y_train[0])
print("The first training example: ", X_train[0].tolist())# I needed to covernt the datatype to list so all the feature values would be printed



# You should NOT use the features of sklearn library in your code.
# total training examples
total_train = y_train.size

train_y0 = np.count_nonzero(y_train == 0) # examples with class 0
train_y1 = np.count_nonzero(y_train == 1) # examples with class 1

p_y0 = train_y0/total_train # probability of class 0
p_y1 = train_y1/total_train # probability of class 1

print(p_y0)
print(p_y1)




# messages with class 0 with check of occurence of the word 'admirer'
y0_admirers = df_train_msgs[df_ytrain == 0].map(lambda x: 'admirer' in x) 

# messages with class 1 with check of occurence of the word 'admirer'
y1_admirers = df_train_msgs[df_ytrain == 1].map(lambda x: 'admirer' in x)



# messages with class 1 having the word 'admirer'
y0_admirers_count = np.count_nonzero(y0_admirers == True) 

# messages with class 0 having the word 'admirer'
y1_admirers_count = np.count_nonzero(y1_admirers == True)



p_admirers_y0 = y0_admirers_count/train_y0 # probability of class 0 having the word 'admirer'
p_admirers_y1 = y1_admirers_count/train_y1 # probability of class 1 having the word 'admirer'

print(p_admirers_y0)
print(p_admirers_y1)




# messages with class 0 with check of occurence of the word 'secret'
y0_secret = df_train_msgs[df_ytrain == 0].map(lambda x: 'secret' in x)

# messages with class 1 with check of occurence of the word 'admirer'
y1_secret = df_train_msgs[df_ytrain == 1].map(lambda x: 'secret' in x)


# messages with class 0 having the word 'secret'
y0_secret_count = np.count_nonzero(y0_secret == True)

# messages with class 1 having the word 'secret'
y1_secret_count = np.count_nonzero(y1_secret == True)




p_secret_y0 = y0_secret_count/train_y0 # probability of class 0 having the word 'secret'
p_secret_y1 = y1_secret_count/train_y1 # probability of class 1 having the word 'secret'

print(p_secret_y0)
print(p_secret_y1)



class BernoulliNB:
    def __init__(self, m=0): # if not mentioned explicitly, we do not perform m smoothing, therefore m = 0
        
        # initilaizing variables('m' and estimated probabilities)
        self.m = m
        self.estimate0_0 = [] # estimated probabilities for feature = 0 and class = 0
        self.estimate0_1 = [] # estimated probabilities for feature = 0 and class = 1
        self.estimate1_0 = [] # estimated probabilities for feature = 1 and class = 0
        self.estimate1_1 = [] # estimated probabilities for feature = 1 and class = 1
    
    def training(self, x_train, y_train): # perform training and estimate probabilites for labels(classes)
        total_train = y_train.size

        train_y0 = np.count_nonzero(y_train == 0)
        train_y1 = np.count_nonzero(y_train == 1)

        # calculating the prior probabilities(p(y = 0) and p(y = 1))
        self.p_y0 = train_y0/total_train
        self.p_y1 = train_y1/total_train
        
        for i in range(x_train.shape[1]):
            features = x_train[:,i] # selecting the first feature of the examples
            
            feature1_y0 = sum(features[y_train == 0]) # total features in class 0 with value = 1 
            feature1_y1 = sum(features[y_train == 1]) # total features in class 1 with value = 1 
            feature0_y0 = len(features[y_train == 0]) - feature1_y0 # total features in class 0 with value = 0 
            feature0_y1 = len(features[y_train == 1]) - feature1_y1 # total features in class 1 with value = 0 
            
            # appending all the estimated probabilities for each feature
            self.estimate0_0.append(float(feature0_y0 + self.m)/(train_y0 + 2*self.m)) 
            self.estimate0_1.append(float(feature0_y1 + self.m)/(train_y1 + 2*self.m))
            self.estimate1_0.append(float(feature1_y0 + self.m)/(train_y0 + 2*self.m))
            self.estimate1_1.append(float(feature1_y1 + self.m)/(train_y1 + 2*self.m))
        
        
        
    def estimate(self): # return the estimated probabilties for each feature in each class
        a = self.estimate0_0
        b = self.estimate1_0
        c = self.estimate0_1
        d = self.estimate1_1
        
        str = '''
        Estimates for each attribute in class 0:
        Estimates for feature = 0 (not existing in the email) and belonging to class 0(ham): {}
        Estimates for feature = 1 (not existing in the email) and belonging to class 0(ham): {}
        Estimates for each attribute in class 1:
        Estimates for feature = 0 (not existing in the email) and belonging to class 1(spam): {}
        Estimates for feature = 1 (not existing in the email) and belonging to class 1(spam): {}
        '''.format(a, b, c, d)
        
        return str
            
    def prediction(self, x_test): # predict the class for the test examples
        predict = []
        for example in x_test: # for every example in the x_test
            log_prob_y0 = 0
            log_prob_y1 = 0
            
            for i, feature in enumerate(example): # for every feature in the example
                if feature == 0:
                    log_prob_y0 += np.log(self.estimate0_0[i])
                    log_prob_y1 += np.log(self.estimate0_1[i])
                elif feature == 1:
                    log_prob_y0 += np.log(self.estimate1_0[i])
                    log_prob_y1 += np.log(self.estimate1_1[i])
                    
            total_prob_y0 = log_prob_y0 + np.log(self.p_y0) # total log of the probabilities
            total_prob_y1 = log_prob_y1 + np.log(self.p_y1)
            
            if total_prob_y0 > total_prob_y1:
                predict.append(0) # if probability of class 0 is greater, we predict class 0, else class 1
            else:
                predict.append(1)
        return np.array(predict)
        



b = BernoulliNB() # object of class BernoulliNB




b.training(X_train, y_train) # training the train examples





b.prediction(X_test[:5,:]) # prediction of first five test examples





b.prediction(X_test[-5:,:]) # prediction of last five test examples





predict_test = b.prediction(X_test) # prediction of test examples





accuracy = (predict_test == y_test).mean()
error = 1 - accuracy
percentage_error = error*100
percentage_error




bm = BernoulliNB(m = 0.2)  # m = 0.2 for smoothing
bm.training(X_train, y_train)




predict_test_m = bm.prediction(X_test)
accuracy_m = (predict_test_m == y_test).mean()
error_m = 1 - accuracy_m
percent_error_m = error_m * 100
percent_error_m



bm = BernoulliNB(m = 0.4)  # m = 0.4 for smoothing
bm.training(X_train, y_train)





predict_test_m = bm.prediction(X_test)
accuracy_m = (predict_test_m == y_test).mean()
error_m = 1 - accuracy_m
percent_error_m = error_m * 100
percent_error_m



bm = BernoulliNB(m = 0.5)  # m = 0.2 for smoothing
bm.training(X_train, y_train)




predict_test_m = bm.prediction(X_test)
accuracy_m = (predict_test_m == y_test).mean()
error_m = 1 - accuracy_m
percent_error_m = error_m * 100
percent_error_m




pred_test = np.array([0]*len(y_test))
accuracy = (pred_test == y_test).mean() # accuracy using zeroR

print("Accuracy using zeroR is {}".format(accuracy)) 




with open('output.txt', 'w') as f: # writing the values to output file
    f.write("P(y = 0):{}".format(p_y0) + '\n')
    f.write("P(y = 1):{}".format(p_y1) + '\n\n')
    f.write("{}".format(b.estimate()) + '\n')
    f.write("predicted class for first 50 examples: " + str(b.prediction(X_test[:50,:])) + '\n')
    f.write("Total number of test examples classified correctly: {}".format(np.count_nonzero(predict_test == y_test)) + '\n')
    f.write("Total number of test examples classified incorrectly: {}".format(np.count_nonzero(predict_test != y_test)) + '\n')
    f.write("The percentage error on the test examples: {}".format(percentage_error))

