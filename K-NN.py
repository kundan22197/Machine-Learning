

# # Using the K-NN algorithm for classification of iris
# 
# In this assigment, you will classify if an Iris is 'Iris Setosa' or 'Iris Versicolour' or 'Iris Virginica' using the k nearest neighbor algorithm.
# 
# The training data is from the UCI Machine Learning Repository: https://archive.ics.uci.edu/ml/datasets/iris.  Please download the dataset before running the code below. 

# ## Step 1:  Getting, understanding, and cleaning the dataset
# 

# ###  Importing the dataset
# 


# Import the usual libraries
import matplotlib.pyplot as plt # plotting utilities 
get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np 
import pandas as pd  # To read in the dataset we will use the Panda's library
df = pd.read_csv('iris.csv', header=None, names = ["sepal length[cm]","sepal width[cm]","petal length[cm]", "petal width", "label"])

# Next we observe the first 5 rows of the data to ensure everything was read correctly
df.head()


# ### Data preprocesssing
# It would be more convenient if the labels were integers instead of 'Iris-setosa', 'Iris-versicolor' and 'Iris-virginica'.  This way our code can always work with numerical values instead of strings.


df['label'] = df.label.map({'Iris-setosa': 0,
              'Iris-versicolor': 1,
              'Iris-virginica': 2})
df.head()# Again, lets observe the first 5 rows to make sure everything worked before we continue




# This time we will use sklearn's method for seperating the data
from sklearn.model_selection import train_test_split
names = ["sepal length[cm]","petal width"]
#After completing the assignment, try your code with all the features
#names = ["sepal length[cm]","sepal width[cm]","petal length[cm]", "petal width"]
df_X_train, df_X_test, df_y_train, df_y_test = train_test_split(df[names],df['label'], random_state=0)

X_train=df_X_train.to_numpy()
X_test=df_X_test.to_numpy()
y_train=df_y_train.to_numpy()
y_test=df_y_test.to_numpy()

#Looking at the train/test split
print("The number of training examples: ", X_train.shape[0])
print("The number of test exampels: ", X_test.shape[0])

print("The first four training labels")
print(y_train[0:4])

print("The first four iris' measurements")
print(X_test[0:4])





X_train[y_train == 1, 0]


X_train


# ## visualizing the data set
# 
# Using a scatter plot to visualize the dataset




iris_names=['Iris-setosa','Iris-versicolor','Iris-virginica']
for i in range(0,3):
    plt.scatter(X_train[y_train == i, 0],
                X_train[y_train == i, 1],
            marker='o',
            label='class '+ str(i)+ ' '+ iris_names[i])

plt.xlabel('sepal width[cm]')
plt.ylabel('petal length[cm]')
plt.legend(loc='lower right')

plt.show()

def euclidean_distance(x1, x2):
    distance = np.sum((x1 -x2)**2) 
    return np.sqrt(distance) 




def get_neighbors( X_train, y_train, x_test, k, distance):
    neigh_distances=list()
    for index in range(len(X_train)):
        calc_distance = distance(X_train[index], x_test)
        neigh_distances.append((calc_distance, y_train[index]))
    neigh_distances.sort(key=lambda x :x[0])
    
    neighbors=list()
    for index in range(k):
        neighbors.append(neigh_distances[index][1])

    return neighbors


def predict(X_train, y_train, X_test, k, distance= euclidean_distance):
    y_pred=list()
    
    for row in X_test:
        neighbors = get_neighbors(X_train, y_train, row, k, distance)
        y_pred.append(max(set(neighbors), key = neighbors.count)) 
            
    return y_pred


# # Part 1 and 2


# For k =1

k=1

y_pred = predict(X_train, y_train, X_test, k, euclidean_distance)

misclassified = np.where(y_pred!=y_test)
print("Indices of elements that where misclassfied in test set are :", misclassified[0], sep='\n')

accuracy = (y_pred==y_test).mean()
print('Accuracy for k = {} is {}'.format(k,accuracy))


# # Part 3 and 4



# For k =3

k=3

y_pred = predict(X_train, y_train, X_test, k, euclidean_distance)

misclassified = np.where(y_pred!=y_test)
print("Indices of elements that where misclassfied in test set are :", misclassified[0], sep='\n')

accuracy = (y_pred==y_test).mean()
print('Accuracy for k = {} is {}'.format(k,accuracy))


# # Part 5 and 6




# For k =5

k=5

y_pred = predict(X_train, y_train, X_test, k, euclidean_distance)

misclassified = np.where(y_pred!=y_test)
print("Indices of elements that where misclassfied in test set are :", misclassified[0], sep='\n')

accuracy = (y_pred==y_test).mean()
print('Accuracy for k = {} is {}'.format(k,accuracy))


# # Part 7




#PART 7 - Zero- R classifier

test_preds = np.array([max(set(y_train), key = list(y_train).count)]*len(y_test))

accuracy = (test_preds==y_test).mean()
print('Accuracy for zeroR is {}'.format(accuracy))


# # Part 8




def manhattan_distance(x1, x2):
    return np.sum(np.absolute(x1 - x2))





# For k =1

k=1

y_pred = predict(X_train, y_train, X_test, k, manhattan_distance)

misclassified = np.where(y_pred!=y_test)
print("Indices of elements that where misclassfied in test set are :", misclassified[0], sep='\n')

accuracy = (y_pred==y_test).mean()
print('Accuracy for k = {} is {}'.format(k,accuracy))



# For k =3

k=3

y_pred = predict(X_train, y_train, X_test, k, manhattan_distance)

misclassified = np.where(y_pred!=y_test)
print("Indices of elements that where misclassfied in test set are :", misclassified[0], sep='\n')

accuracy = (y_pred==y_test).mean()
print('Accuracy for k = {} is {}'.format(k,accuracy))


# For k =5

k=5

y_pred = predict(X_train, y_train, X_test, k, manhattan_distance)

misclassified = np.where(y_pred!=y_test)
print("Indices of elements that where misclassfied in test set are :", misclassified[0], sep='\n')

accuracy = (y_pred==y_test).mean()
print('Accuracy for k = {} is {}'.format(k,accuracy))


# # Part 9

# For k =3

k=3

y_pred = predict(X_train, y_train, X_test, k, manhattan_distance)

misclassified = np.where(y_pred!=y_test)
print("Indices of elements that where misclassfied in test set are :", misclassified[0], sep='\n')

accuracy = (y_pred==y_test).mean()
print('Accuracy for k = {} is {}'.format(k,accuracy))




# For k =5

k=5

y_pred = predict(X_train, y_train, X_test, k, manhattan_distance)

misclassified = np.where(y_pred!=y_test)
print("Indices of elements that where misclassfied in test set are :", misclassified[0], sep='\n')

accuracy = (y_pred==y_test).mean()
print('Accuracy for k = {} is {}'.format(k,accuracy))



# For k =7

k=7

y_pred = predict(X_train, y_train, X_test, k, manhattan_distance)

misclassified = np.where(y_pred!=y_test)
print("Indices of elements that where misclassfied in test set are :", misclassified[0], sep='\n')

accuracy = (y_pred==y_test).mean()
print('Accuracy for k = {} is {}'.format(k,accuracy))





len(X_train)




X_new= np.append(X_train, X_test, axis=0)




y_new = np.append(y_train, y_test, axis=0)




y_new




x =np.array([[1, 28540],[1,40133],[1,39900],[1,42050],[1,43220],[1,39565],[1,40400],[1,54506],[1,0],[1,0]]).astype(float)





y =np.array([137,135,127,118,118,117,117,114,122,120])




w = np.dot(np.dot(np.linalg.inv(np.dot(x.T, x)), x.T), y)




regression_line = np.arange(1,60000,1000)
regression_line_y = w[0] + w[1]*regression_line




plt.scatter(x[:,1], y, alpha=0.5)
plt.grid(True)
plt.title('Mid-Career Salary vs Yearly Tuition (With outliers)')
plt.xlabel('Yearly Tuition')
plt.ylabel('Mid-Career Salary')
plt.plot(regression_line, regression_line_y, color='red')
plt.show()

