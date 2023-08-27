# Introduction to Deep Learning in Python
# setting working directory
import os
os.chdir('E:/DataCampProjects/Deep Learning')




# CHAPTER 1. Basics of deep learning and neural networks


# Lesson 1.1 Introduction to deep learning

# imagine to build a model predicting how many transactions each customer will make next year
# have features like age, bank balance, whether they are retired and so on
# seen it as by linear regression
# y = a1*x1 + a2*x2 + a3*x3 + ... + c
# linear regression lacks of interaction

# neural networks account for interactions very well
# deep learning is a powerful neural network
# it can capture complex interactions
# it can do amazing things with text, images, videos, audios, source code and so on

# on left side, we have nodes called input
# on right side, we have node called output
# we have hidden layer in the middle to represent interactions


# Lesson 1.2 Forward propagation (spread)

# imagine we predict number of transactions
# inputs are number of children and number of existing accounts
# there are two nodes in hidden layers
# each forward propagation has a weight value
# we do multiply - add process for one data point at a time

# EXAMPLE:
# 1) coding the forward propagation algorithm
import numpy as np
input_data = np.array([3, 5])
weights = {'node_0': np.array([2, 4]),
           'node_1': np.array([4, -5]),
           'output': np.array([2, 7])}

# calculate node 0 value:
node_0_value = (input_data * weights['node_0']).sum()

# calculate node 1 value:
node_1_value = (input_data * weights['node_1']).sum()

# put node values into hidden layer array:
hidden_layer_outputs = np.array([node_0_value, node_1_value])

# calculate output:
output = (hidden_layer_outputs * weights['output']).sum()
print(output) # -39


# Lesson 1.3 Activation functions

# we need activation functions to fully optimize neural network prediction power
# activation functions allow to catch non-linearities
# it is called ReLU (Rectified Linear Activation)
# it is achieved in Python by np.tanh()

# EXERCISE:
# 1) the rectified linear activation function


def relu(input):
    """Define your relu activation function here"""
    # calculate the value for the output of the relu function:
    output = max(0, input)
    return(output)


# calculate the node 0 value:
node_0_input = (input_data * weights['node_0']).sum()
node_0_output = relu(node_0_input)

# calculate the node 1 value:
node_1_input = (input_data * weights['node_1']).sum()
node_1_output = relu(node_1_input)

# put node values into hidden layer array:
hidden_layer_outputs = np.array([node_0_output, node_1_output])

# calculate model output (do not apply relu)
model_output = (hidden_layer_outputs * weights['output']).sum()
print(model_output) # 52

# 2) applying the network to many observations/ rows of data
# define predict_with_network()


def predict_with_network(input_data_row, weights):

    # calculate node 0 value
    node_0_input = (input_data_row * weights['node_0']).sum()
    node_0_output = relu(node_0_input)

    # calculate node 1 value
    node_1_input = (input_data_row * weights['node_1']).sum()
    node_1_output = relu(node_1_input)

    # put node values into array:
    hidden_layer_outputs = np.array([node_0_output, node_1_output])

    # calculate model output
    input_to_final_layer = (hidden_layer_outputs * weights['output']).sum()
    model_output = relu(input_to_final_layer)
    return(model_output)


# create empty list to store prediction results
results = []
input_data = np.array([np.array([3,5]),
                       np.array([1,-1]),
                       np.array([0,0]),
                       np.array([8,4])])
for input_data_row in input_data:
    results.append(predict_with_network(input_data_row, weights))

print(results) # [52, 63, 0, 148]


# Lesson 1.4 Deeper networks

# the modern deep learning models have more than one successive hidden layers
# deep networks internally build representation of patterns in the data
# it partially replace the need for feature engineering

# EXERCISE:
# 1) multi-layer neural network


def predict_with_network(input_data):
    # calculate node 0 in the first hidden layer
    node_0_0_input = (input_data * weights['node_0_0']).sum()
    node_0_0_output = relu(node_0_0_input)

    # calculate node 1 in the first hidden layer
    node_0_1_input = (input_data * weights['node_0_1']).sum()
    node_0_1_output = relu(node_0_1_input)

    # put node values into array
    hidden_0_outputs = np.array([node_0_0_output, node_0_1_output])

    # calculate node 0 in the second hidden layer
    node_1_0_input = (hidden_0_outputs * weights['node_1_0']).sum()
    node_1_0_output = relu(node_1_0_input)

    # calculate node 1 in the second hidden layer
    node_1_1_input = (hidden_0_outputs * weights['node_1_1']).sum()
    node_1_1_output = relu(node_1_1_input)

    # put node values into array
    hidden_1_outputs = np.array([node_1_0_output, node_1_1_output])

    # calculate model output
    model_output = (hidden_1_outputs * weights['output']).sum()
    return(model_output)


import numpy as np
input_data = np.array([3,5])

weights = {'node_0_0': np.array([2, 4]),
           'node_0_1': np.array([4, -5]),
           'node_1_0': np.array([-1, 2]),
           'node_1_1': np.array([1, 2]),
           'output': np.array([2, 7])}
output = predict_with_network(input_data)
print(output)





# CHAPTER 2. Optimizing a neural network with backward propagation


# Lesson 2.1 The need for optimization

# it is hard to make accurate predictions with more data points
# we use loss function to calculate aggregate errors in prediction to a single number
# it can measure performance of a model
# the best model is the one with lowest loss function

# Gradient descent is a method to find the lowest point in the dark
# take a small step downhill, repeat until it is uphill in every direction
# Gradient descent steps:
    # 1. start at random point
    # 2. Until you are somewhere flat (get slope, take a step downhill)

# EXERCISE:
# 1) coding how weight changes affect accuracy
# input data
import numpy as np
input_data = np.array([0,3])

# weights
weights_0 = {'node_0': [2,1],
             'node_1': [1,2],
             'output': [1,1]}

# target value
target_actual = 3

# define make predictions function
def predict_with_network(input_data_row, weights):

    # calculate node 0 value
    node_0_input = (input_data_row * weights['node_0']).sum()
    node_0_output = relu(node_0_input)

    # calculate node 1 value
    node_1_input = (input_data_row * weights['node_1']).sum()
    node_1_output = relu(node_1_input)

    # put node values into array:
    hidden_layer_outputs = np.array([node_0_output, node_1_output])

    # calculate model output
    input_to_final_layer = (hidden_layer_outputs * weights['output']).sum()
    model_output = relu(input_to_final_layer)
    return(model_output)

# make prediction with original weights
model_output_0 = predict_with_network(input_data, weights_0)

# calculate error
error_0 = model_output_0 - target_actual

# create new weights
weights_1 = {'node_0': [2,1],
             'node_1': [1,0],
             'output': [1,1]}

# make predictions with new weights
model_output_1 = predict_with_network(input_data, weights_1)

# calculate new error
error_1 = model_output_1 - target_actual

print(error_0)
print(error_1)

# 2) scaling up to multiple data points
from sklearn.metrics import mean_squared_error

# create model_output_0
model_output_0 = []

# create model_output_1
model_output_1 = []

# input_data, weights_0, weights_1, target_actuals
input_data = np.array([
    np.array([0,3]),
    np.array([1,2]),
    np.array([-1,-2]),
    np.array([4,0])
])
weights_0 = {'node_0': [2,1],
             'node_1': [1,2],
             'output': [1,1]}
weights_1 = {'node_0': [2,1],
             'node_1': [1,1.5],
             'output': [1,1.5]}
target_actuals = [1, 3, 5, 7]

# loop over input_data
for row in input_data:
    # append prediction to model_output_0
    model_output_0.append(predict_with_network(row, weights_0))
    # append prediction to model_output_1
    model_output_1.append(predict_with_network(row, weights_1))

# calculate the mean squared error for model_output_0 and model_output_1
mse_0 = mean_squared_error(target_actuals, model_output_0)
mse_1 = mean_squared_error(target_actuals, model_output_1)
print("Mean squared error with weights_0: %f" %mse_0)
print("Mean squared error with weights_1: %f" %mse_1)


# Lesson 2.2 Gradient descent

# repeatedly found a slope capturing how your loss function changes as weight changes
# then, you made a small change to the weight
# repeat this process until you couldn't go downhill any more

# if the slope is positive, going opposite the slope means moving to lower numbers
# subtract the slope from current value achieve this, but this might be too big
# we multiply the slope by a small number, called the learning rate

# how to calculate a slope?
    # 1) slope of the loss function with respect to value at the node we feed into
    # 2) the value of the node that feeds into our weight
    # 3) slope of activation function with respect to value we feed into

# example:
# node: 3 * weight: 2 = output: 6 (Actual Target Value = 10)
# 1) slope of mean-squared loss function with respect to prediction: 2 Error = 2 * (6 - 10) = 2 * (-4) = -8

# 2) the value of the node that feeds into our weight: 3

# 3) slope of activation function with respect to value we feed into (we can leave it out here)

# result of slope: -8 * 3 = -24


# EXERCISE:
# 1) calculate slopes
# input_data
import numpy as np
input_data = np.array([1,2,3])

# weights
weights = np.array([0,2,1])

# target
target = 0

# calculate the predictions
preds = (input_data * weights).sum()

# calculate the error
error = preds - target

# calculate the slope:
slope = input_data * error * 2

# 2) improve model weights
# set learning rate
learning_rate = 0.01

# update the weights
weights_updated = weights - learning_rate * slope

# update predictions
preds_updated = (input_data * weights_updated).sum()

# update error
error_updated = preds_updated - target

# compare
print(error) # 7
print(error_updated) # 5.04

# 3) make multiple updates to weights
# define helper functions for this exercise


def get_slope(input_data, target, weights):
    preds = (input_data * weights).sum()
    error = preds - target
    slope = input_data * error * 2
    return(slope)


def get_mse(input_data, target, weights):
    preds = (input_data * weights).sum()
    error = preds - target
    mse = error ** 2
    return(mse)



# make 20 updates to weights
n_updates = 20
mse_hist = []

# iterate over the number of updates
for i in range(n_updates):
    # calculate the slope
    slope = get_slope(input_data, target, weights)
    # update the weights
    weights = weights - 0.01 * slope
    # calculate mse with new weights
    mse = get_mse(input_data, target, weights)
    # append the mse to mse_hist
    mse_hist.append(mse)

# plot the mse history
import matplotlib.pyplot as plt
plt.plot(mse_hist)
plt.xlabel('Iterations')
plt.ylabel('Mean Squared Error')
plt.show()


# Lesson 2.3 Backpropagation

# use backpropagation to calculate the slopes we need to optimize more complex deep learning models
# backpropagation starts from error
# it goes through hidden layers all the way to input data
# it allows Gradient descent to update all weights in neural networks
# it comes from chain rule of calculus

# backpropagation process:
    # try to estimate the slope of the loss function with respect to each weight
    # do forward propagation to calculate predictions and errors before backpropagation

# EXERCISE:
# input_data: [1, 3]
# weights: [1, 2]
# target: 4
# prediction: 7
# error: 7 - 4 = 3
# slope: [2 * 3 * 1, 2 * 3 * 3] = [6, 18]

# stochastic gradient descent
# it is common to calculate slopes on only a subset of the data (a batch)
# use a different batch of data to calculate the next update
# start over the beginning once all data is used
# each time through the training data is called an epoch





# CHAPTER 3. Building deep learninng models with keras


# Lesson 3.1 Creating a keras model

# keras model building workflow has four steps:
    # 1) specify architecture
    # 2) compile the model
    # 3) fit the model
    # 4) predict

# EXERCISE:
# 1) dataset: wages
import pandas as pd
df = pd.read_csv('Deep Learning Datasets/hourly_wages.csv')
print(df.head())
print(df.info())

# separate df to predictors and target variable and convert them to numpy matrix
predictors = df.drop(['wage_per_hour'], axis=1)
predictors = predictors.to_numpy()

target = df[['wage_per_hour']]
target = target.to_numpy()

# import necessary modules
from keras.layers import Dense
from keras.models import Sequential

# save the number of columns in predictors
n_cols = predictors.shape[1]

# specify the model
model = Sequential()
model.add(Dense(50, activation='relu', input_shape=(n_cols,)))
model.add(Dense(32, activation='relu'))
model.add(Dense(1))


# Lesson 3.2 Compiling and fitting a model

# compile your model:
    # 1) specify you optimizer, "adam" is usually a good choice
    # 2) loss function, "mean_squared_error" common for regression

# fit your model:
    # applying backpropagation and gradient descent with your data
    # update the weights
    # scaling data before fitting can help optimization

# EXERCISE:
# 1) compile and fit the model
# dataset: predictors, target

# import necessary modules
from keras.layers import Dense
from keras.models import Sequential

# save the number of columns in predictors
n_cols = predictors.shape[1]

# specify the model
model = Sequential()
model.add(Dense(50, activation='relu', input_shape=(n_cols,)))
model.add(Dense(32, activation='relu'))
model.add(Dense(1))

# compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# fit the model
model.fit(predictors, target)


# Lesson 3.3 Classification model

# deep learning works similarly for classification
# predicting outcomes from a set of discrete options

# differences compared to regression problems:
    # 1) loss function: "categorical_crossentropy" instead of "mean_squared_error"
    # 2) add metrics: "accuracy" to compile step for easy-to-understand diagnostics
    # 3) outputer layer has multiple nodes, one for each possible outcome and use "softmax" activation
    # 4) one-hot encoding to transform the categorical column to binary columns with method 'to_categorical'

# EXERCISE:
# 1) classification models
# dataset: titanic
import pandas as pd
titanic = pd.read_csv('Deep Learning Datasets/titanic.csv')
print(titanic.head())
print(titanic.info())

# convert dataframe to predictors and target

predictors = titanic.drop(['survived'], axis=1)
predictors = predictors.to_numpy()

from keras.utils.np_utils import to_categorical
target = to_categorical(titanic.survived)

# set up the model
from keras.layers import Dense
from keras.models import Sequential
n_cols = predictors.shape[1]
model = Sequential()
model.add(Dense(32, activation='relu', input_shape=(n_cols,)))
model.add(Dense(2, activation='softmax'))

# compile the model
model.compile(optimizer='sgd',
              loss='categorical_crossentropy', metrics=['accuracy'])

# fit the model
model.fit(predictors, target)


# Lesson 3.4 Using models

# save a model (.h5 format)
# reload a model (.h5 format)
# make predictions with a model
# verify model structure with .summary() method

# EXERCISE:
# 1) make predictions
# dataset: new data called 'pred_data'
pred_data = ([[2, 34.0, 0, 0, 13.0, 1, False, 0, 0, 1],
       [2, 31.0, 1, 1, 26.25, 0, False, 0, 0, 1],
       [1, 11.0, 1, 2, 120.0, 1, False, 0, 0, 1],
       [3, 0.42, 0, 1, 8.5167, 1, False, 1, 0, 0],
       [3, 27.0, 0, 0, 6.975, 1, False, 0, 0, 1],
       [3, 31.0, 0, 0, 7.775, 1, False, 0, 0, 1],
       [1, 39.0, 0, 0, 0.0, 1, False, 0, 0, 1],
       [3, 18.0, 0, 0, 7.775, 0, False, 0, 0, 1],
       [2, 39.0, 0, 0, 13.0, 1, False, 0, 0, 1],
       [1, 33.0, 1, 0, 53.1, 0, False, 0, 0, 1],
       [3, 26.0, 0, 0, 7.8875, 1, False, 0, 0, 1],
       [3, 39.0, 0, 0, 24.15, 1, False, 0, 0, 1],
       [2, 35.0, 0, 0, 10.5, 1, False, 0, 0, 1],
       [3, 6.0, 4, 2, 31.275, 0, False, 0, 0, 1],
       [3, 30.5, 0, 0, 8.05, 1, False, 0, 0, 1],
       [1, 29.69911764705882, 0, 0, 0.0, 1, True, 0, 0, 1],
       [3, 23.0, 0, 0, 7.925, 0, False, 0, 0, 1],
       [2, 31.0, 1, 1, 37.0042, 1, False, 1, 0, 0],
       [3, 43.0, 0, 0, 6.45, 1, False, 0, 0, 1],
       [3, 10.0, 3, 2, 27.9, 1, False, 0, 0, 1],
       [1, 52.0, 1, 1, 93.5, 0, False, 0, 0, 1],
       [3, 27.0, 0, 0, 8.6625, 1, False, 0, 0, 1],
       [1, 38.0, 0, 0, 0.0, 1, False, 0, 0, 1],
       [3, 27.0, 0, 1, 12.475, 0, False, 0, 0, 1],
       [3, 2.0, 4, 1, 39.6875, 1, False, 0, 0, 1],
       [3, 29.69911764705882, 0, 0, 6.95, 1, True, 0, 1, 0],
       [3, 29.69911764705882, 0, 0, 56.4958, 1, True, 0, 0, 1],
       [2, 1.0, 0, 2, 37.0042, 1, False, 1, 0, 0],
       [3, 29.69911764705882, 0, 0, 7.75, 1, True, 0, 1, 0],
       [1, 62.0, 0, 0, 80.0, 0, False, 0, 0, 0],
       [3, 15.0, 1, 0, 14.4542, 0, False, 1, 0, 0],
       [2, 0.83, 1, 1, 18.75, 1, False, 0, 0, 1],
       [3, 29.69911764705882, 0, 0, 7.2292, 1, True, 1, 0, 0],
       [3, 23.0, 0, 0, 7.8542, 1, False, 0, 0, 1],
       [3, 18.0, 0, 0, 8.3, 1, False, 0, 0, 1],
       [1, 39.0, 1, 1, 83.1583, 0, False, 1, 0, 0],
       [3, 21.0, 0, 0, 8.6625, 1, False, 0, 0, 1],
       [3, 29.69911764705882, 0, 0, 8.05, 1, True, 0, 0, 1],
       [3, 32.0, 0, 0, 56.4958, 1, False, 0, 0, 1],
       [1, 29.69911764705882, 0, 0, 29.7, 1, True, 1, 0, 0],
       [3, 20.0, 0, 0, 7.925, 1, False, 0, 0, 1],
       [2, 16.0, 0, 0, 10.5, 1, False, 0, 0, 1],
       [1, 30.0, 0, 0, 31.0, 0, False, 1, 0, 0],
       [3, 34.5, 0, 0, 6.4375, 1, False, 1, 0, 0],
       [3, 17.0, 0, 0, 8.6625, 1, False, 0, 0, 1],
       [3, 42.0, 0, 0, 7.55, 1, False, 0, 0, 1],
       [3, 29.69911764705882, 8, 2, 69.55, 1, True, 0, 0, 1],
       [3, 35.0, 0, 0, 7.8958, 1, False, 1, 0, 0],
       [2, 28.0, 0, 1, 33.0, 1, False, 0, 0, 1],
       [1, 29.69911764705882, 1, 0, 89.1042, 0, True, 1, 0, 0],
       [3, 4.0, 4, 2, 31.275, 1, False, 0, 0, 1],
       [3, 74.0, 0, 0, 7.775, 1, False, 0, 0, 1],
       [3, 9.0, 1, 1, 15.2458, 0, False, 1, 0, 0],
       [1, 16.0, 0, 1, 39.4, 0, False, 0, 0, 1],
       [2, 44.0, 1, 0, 26.0, 0, False, 0, 0, 1],
       [3, 18.0, 0, 1, 9.35, 0, False, 0, 0, 1],
       [1, 45.0, 1, 1, 164.8667, 0, False, 0, 0, 1],
       [1, 51.0, 0, 0, 26.55, 1, False, 0, 0, 1],
       [3, 24.0, 0, 3, 19.2583, 0, False, 1, 0, 0],
       [3, 29.69911764705882, 0, 0, 7.2292, 1, True, 1, 0, 0],
       [3, 41.0, 2, 0, 14.1083, 1, False, 0, 0, 1],
       [2, 21.0, 1, 0, 11.5, 1, False, 0, 0, 1],
       [1, 48.0, 0, 0, 25.9292, 0, False, 0, 0, 1],
       [3, 29.69911764705882, 8, 2, 69.55, 0, True, 0, 0, 1],
       [2, 24.0, 0, 0, 13.0, 1, False, 0, 0, 1],
       [2, 42.0, 0, 0, 13.0, 0, False, 0, 0, 1],
       [2, 27.0, 1, 0, 13.8583, 0, False, 1, 0, 0],
       [1, 31.0, 0, 0, 50.4958, 1, False, 0, 0, 1],
       [3, 29.69911764705882, 0, 0, 9.5, 1, True, 0, 0, 1],
       [3, 4.0, 1, 1, 11.1333, 1, False, 0, 0, 1],
       [3, 26.0, 0, 0, 7.8958, 1, False, 0, 0, 1],
       [1, 47.0, 1, 1, 52.5542, 0, False, 0, 0, 1],
       [1, 33.0, 0, 0, 5.0, 1, False, 0, 0, 1],
       [3, 47.0, 0, 0, 9.0, 1, False, 0, 0, 1],
       [2, 28.0, 1, 0, 24.0, 0, False, 1, 0, 0],
       [3, 15.0, 0, 0, 7.225, 0, False, 1, 0, 0],
       [3, 20.0, 0, 0, 9.8458, 1, False, 0, 0, 1],
       [3, 19.0, 0, 0, 7.8958, 1, False, 0, 0, 1],
       [3, 29.69911764705882, 0, 0, 7.8958, 1, True, 0, 0, 1],
       [1, 56.0, 0, 1, 83.1583, 0, False, 1, 0, 0],
       [2, 25.0, 0, 1, 26.0, 0, False, 0, 0, 1],
       [3, 33.0, 0, 0, 7.8958, 1, False, 0, 0, 1],
       [3, 22.0, 0, 0, 10.5167, 0, False, 0, 0, 1],
       [2, 28.0, 0, 0, 10.5, 1, False, 0, 0, 1],
       [3, 25.0, 0, 0, 7.05, 1, False, 0, 0, 1],
       [3, 39.0, 0, 5, 29.125, 0, False, 0, 1, 0],
       [2, 27.0, 0, 0, 13.0, 1, False, 0, 0, 1],
       [1, 19.0, 0, 0, 30.0, 0, False, 0, 0, 1],
       [3, 29.69911764705882, 1, 2, 23.45, 0, True, 0, 0, 1],
       [1, 26.0, 0, 0, 30.0, 1, False, 1, 0, 0],
       [3, 32.0, 0, 0, 7.75, 1, False, 0, 1, 0]])

# specify, compile, and fit the model
from keras.layers import Dense
from keras.models import Sequential
model = Sequential()
model.add(Dense(32, activation='relu', input_shape=(n_cols,)))
model.add(Dense(2, activation='softmax'))
model.compile(optimizer='sgd',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
model.fit(predictors, target)

# calculate predictions
predictions = model.predict(pred_data)

# calculate predicted probabilities of survival
predicted_prob_true = predictions[:, 1]
print(predicted_prob_true)





# CHAPTER 4. Fine-tuning keras models


# Lesson 4.1 Understanding model optimization

# optimization is difficult
    # simultaneously optimizing thousands of parameters
    # updates may not improve model meaningfully
    # updates too small or too large (because of learning rate)

# try to use the simplest model first
# then, we try with different learning rates and compare their results

# we may also meet "dying neuron" problem
# that ReLU activation will turn any node with negative input into output of 0 and slope of 0
# so the weights on this node will not get updated

# "vanishing gradients" problem
# that non-zero but small slopes in deep networks, updates to backprop were close to 0

# EXERCICSE:
# 1) change optimization parameters
# import SGD optimizer
from keras.optimizers import SGD
from keras.layers import Dense
from keras.models import Sequential

# Save the number of columns in predictors: n_cols
n_cols = predictors.shape[1]
input_shape = (n_cols,)

# create function to create new model


def get_new_model(input_shape=input_shape):
    model = Sequential()
    model.add(Dense(100, activation='relu', input_shape=input_shape))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(2, activation='softmax'))
    return(model)


# create list of learning rates: lr_to_test
lr_to_test = [0.00001, 0.01, 1]

# loop over learning rates
for lr in lr_to_test:
    print('\n\nTesting model with learning rate: %f\n' %lr)

    # build new model to test, unaffected by previous models
    model = get_new_model()

    # create new SGD optimizer with specified learning rate:
    my_optimizer = SGD(lr=lr)

    # compile the model
    model.compile(optimizer=my_optimizer, loss='categorical_crossentropy')

    # fit the model
    model.fit(predictors, target)


# Lesson 4.2 Model Validation

# performance on training data is not a good indication of performance on new data
# we use validation data/ test data to test model performance
# people do not use cross-validation in deep learning, they use validation split in a single run
# when score isn't improving, we use early stopping to stop the model

# you can experiment with different architectures:
    # more or fewer layers
    # more or fewer nodes in each layer
# create a great model requires experimentation

# EXERCISE:
# 1) evaluating model accuracy on validation dataset
# save the number of columns in predictors: n_cols
n_cols = predictors.shape[1]
input_shape = (n_cols,)

# specify the model
model = Sequential()
model.add(Dense(100, activation='relu', input_shape = input_shape))
model.add(Dense(100, activation='relu'))
model.add(Dense(2, activation='softmax'))

# compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# fit the model
hist = model.fit(predictors, target, validation_split=0.3)

# 2) early stopping: optimizing the optimization
# import EarlyStopping
from keras.callbacks import EarlyStopping

# save the number of columns in predictors: n_cols
n_cols = predictors.shape[1]
input_shape = (n_cols,)

# specify the model
model = Sequential()
model.add(Dense(100, activation='relu', input_shape = input_shape))
model.add(Dense(100, activation='relu'))
model.add(Dense(2, activation='softmax'))

# compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# define early_stopping_monitor
early_stopping_monitor = EarlyStopping(patience=2)

# fit the model
model.fit(predictors, target, validation_split=0.3, epochs=30, callbacks=[early_stopping_monitor])

# You can also experiment with adding layers or nodes


# Lesson 4.3 Thinking about model capacity

# "model capacity" is key considerations when you think about what models to try
# it is related to over-fitting or under-fitting
# model capacity is a model's ability to capture predictive patterns in data

# workflow for optimizing model capacity:
    # start with a small network
    # gradually increase capacity (adding layers and nodes)
    # until validation score is no longer improving


# Lesson 4.4 Stepping into images

# MNIST dataset is a good starting point for image dataset
# it has handwritten digits images (28*28 pixels)
# value in each part of array denotes darkness of that pixel

# EXERCISE:
# 1) building your own digit recognition model
# Get X and y from dataset
import pandas as pd
digits = pd.read_csv('Deep Learning Datasets/MNIST.csv', header=None)
print(digits.head())

X = digits.drop([0], axis=1)
y = digits[0]
print(X.head())
print(y.head())

# Create the model
from keras.layers import Dense
from keras.models import Sequential
model = Sequential()
model.add(Dense(50, activation='relu', input_shape=(784,)))
model.add(Dense(50, activation='relu'))
model.add(Dense(10, activation='softmax'))

# compile the model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# fit the model
model.fit(X, y, validation_split=0.3)

# keras.io for excellent documentation
# Graphical processing unit(GPU) provides dramatic speedups in model training times
