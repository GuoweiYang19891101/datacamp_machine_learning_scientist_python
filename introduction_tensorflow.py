# Introduction to TensorFlow

# setting working directory
import os
os.chdir('E:/DataCampProjects/Deep Learning')





# CHAPTER 1. Introduction to TensorFlow


# Lesson 1.1 Constants and variables

# TensorFlow is an open-source library for graph-based numerical computation
# tensor is described as a generalization of vectors and matrices
# you can think of it as a collection of numbers into a specific shape

# constants in TensorFlow
# a constant does not change and cannot be trained, it can have any dimension

# variables in TensorFlow
# a variable's value can change during computation
# the value is shared, persistent, and modifiable
# the data type and shape are fixed

# EXERCISE:
# 1) defining data as constants
# import constant from TensorFlow
from tensorflow import constant

# convert the credit_numpy array into a tensorflow constant
import numpy as np

credit_numpy = np.array([[2.0000e+00, 1.0000e+00, 2.4000e+01, 3.9130e+03],
                         [2.0000e+00, 2.0000e+00, 2.6000e+01, 2.6820e+03],
                         [2.0000e+00, 2.0000e+00, 3.4000e+01, 2.9239e+04],
                         [2.0000e+00, 2.0000e+00, 3.7000e+01, 3.5650e+03],
                         [3.0000e+00, 1.0000e+00, 4.1000e+01, -1.6450e+03],
                         [2.0000e+00, 1.0000e+00, 4.6000e+01, 4.7929e+04]])
credit_constant = constant(credit_numpy)

print(credit_numpy)
print(credit_constant)
print('\n The datatyp of original is:', credit_numpy.dtype)
print('\n The shape of original is:', credit_numpy.shape)
print('\n The datatype is:', credit_constant.dtype)
print('\n The shape is:', credit_constant.shape)

# 2) defining variables
# import Variable from TensorFlow
from tensorflow import Variable

# define the 1-dimensional variable A1
A1 = Variable([1, 2, 3, 4])
print('\n A1:', A1)

# convert A1 to a numpy array and assign it to B1
B1 = A1.numpy()
print('\n B1:', B1)

# Lesson 1.2 Basic operations

# tensorflow graph contains edges and nodes, edges are tensors and nodes are operations
# perform add operation with add(), tensors should have same shape
# perform element-wise multiplication operation with multiply(), tensors should have same shape
# perform matrix multiplication operation with matmul(), columns of A equal rows of B
# perform summing over tensor dimension with reduce_sum(), sums over all dimension of tensor

# EXERCISE:
# 1) performing element-wise multiplication
# define tensor A1 and A23 as constants
from tensorflow import constant, ones_like, multiply

A1 = constant([1, 2, 3, 4])
A23 = constant([[1, 2, 3], [1, 6, 4]])

# define B1 and B23 to have the correct shape
B1 = ones_like(A1)
B23 = ones_like(A23)

# perform element-wise multiplication
C1 = multiply(A1, B1)
C23 = multiply(A23, B23)

# print the tensors C1 and C23
print('\n C1: {}'.format(C1.numpy()))
print('\n C23: {}'.format(C23.numpy()))

# 2) making predictions with matrix multiplication
# define features, params, and bill as constants
from tensorflow import constant, matmul

features = constant([[2, 24], [2, 26], [2, 57], [1, 37]])
params = constant([[1000], [150]])
bill = constant([[3913], [2682], [8617], [64400]])

# compute billpred using features and params
billpred = matmul(features, params)
print(billpred)

# compute and print the error
error = bill - billpred
print(error.numpy)

# Lesson 1.3 Advanced operations

# we have covered add(), multiply(), matmul(), and reduce_sum()
# we will cover gradient(), reshape(), and random()

# gradient() computes the slope of a function at a point
# reshape() reshapes a tensor (e.g. 10 * 10 to 100 * 1)
# random() populates tensor with entries drawn from a probability distribution

# gradient() can help to find minimum or maximum when find gradient = 0
# reshape() can help in data reshape for image classification problem

# EXERCISE:
# 1) reshaping tensors
# write two numpy arrays
import numpy as np

gray_tensor = np.array([])
color_tensor = np.array([])

# reshape the grayscale image tensor into a vector
from tensorflow import reshape

gray_vector = reshape(gray_tensor, (784, 1))

# reshape the color image tensor into a vector
color_vector = reshape(color_tensor, (2352, 1))

# 2) optimizing with gradients
from tensorflow import Variable, GradientTape, multiply


def compute_gradient(x0):
    # define x as a variable with an initial value of x0
    x = Variable(x0)
    with GradientTape() as tape:
        tape.watch(x)
        # define y using the multiply operation
        y = x * x
    # return the gradient of y with respect to x
    return tape.gradient(y, x).numpy()


# compute and print gradients at x = -1, 1, and 0
print(compute_gradient(-1.0))
print(compute_gradient(1.0))
print(compute_gradient(0.0))

# 3) working with image data
import numpy as np

letter = np.array([[1., 0., 1.],
                   [1., 1., 0.],
                   [1., 0., 1.]])
model = np.array([[1., 0., -1.]])

# reshape model from 1*3 to 3*1 tensor
from tensorflow import reshape, reduce_sum

model = reshape(model, (3, 1))

# multiply letter by model
output = matmul(letter, model)

# sum over output and print prediction using the numpy method
prediction = reduce_sum(output)
print(prediction.numpy())

# CHAPTER 2. Linear models


# Lesson 2.1 Input data

# in machine learning model, we will want to import data from external source
# data can be imported use 'tensorflow' module
# simple way is to import data using 'pandas', then convert data to numpy array
# read_csv() is used to import csv data
# parameters in read_csv(): filepath, sep, delim_whitespace, encoding
# np.array() can set the data type (e.g. np.float32, np.bool)
# tf.cast() can also set the data type (e.g. tf.float32, tf.bool)

# EXERCISE:
# 1) loading data using pandas
# import pandas
import pandas as pd

# assign the path to a string variable 'data_path'
data_path = 'TensorFlow Datasets/kc_house_data.csv'

# load the dataset as a dataframe 'housing'
housing = pd.read_csv(data_path)
print(housing.columns)
print(housing['price'])

# 2) setting the data type
# import numpy and tensorflow
import numpy as np
import tensorflow as tf

# use a numpy array to define price as a 32-bit float
price = np.array(housing['price'], np.float32)

# define waterfront as a boolean using cast
waterfront = tf.cast(housing['waterfront'], tf.bool)

# print them
print(price)
print(waterfront)

# Lesson 2.2 Loss functions

# loss function tells us how well our model explains the data
# we use this to adjust our model during training process
# loss function value is the lower, the better
# we want to minimize the loss function
# common loss functions in TensorFlow:
# mean squared error (MSE)
# mean absolute error (MAE)
# huber error

# EXERCISE:
# 1) loss function in TensorFlow
# import the keras module from tensorflow
from tensorflow import keras

# compute the mean squared error (MSE)
price = np.array(housing['price'], np.float32)
predictions = np.array([])
loss = keras.losses.mse(price, predictions)
print(loss.numpy())

# Compute the mean absolute error (mae)
loss = keras.losses.mae(price, predictions)
print(loss.numpy())

# 2) modifying loss function
# initiate a variable named scalar
import tensorflow as tf

scalar = tf.cast(1.0, tf.float32)
features = np.array([1., 2., 3., 4., 5.])
targets = np.array([2., 4., 6., 8., 10.])


# define the model


def model(scalar, features=features):
    return scalar * features


# define a loss function


def loss_function(scalar, features=features, targets=targets):
    # compute the predicted values
    predictions = model(scalar, features)
    # return the mean absolute error loss
    return keras.losses.mae(targets, predictions)


# evaluate the loss function and print the loss
print(loss_function(scalar).numpy())

# Lesson 2.3 Linear regression

# linear regression model assumes that relationship between variables can be captured by a line
# two parameters: intercept, slope

# EXERCISE:
# 1) set up a linear regression
# get the data
import pandas as pd

data_path = 'TensorFlow Datasets/kc_house_data.csv'
housing = pd.read_csv(data_path)
print(housing.info())
print(housing.head())

# define target and features
import numpy as np

price_log = np.array(housing['price'], np.float32)
size_log = np.array(housing['sqft_living'], np.float32)


# define a linear regression model


def linear_regression(intercept, slope, features=size_log):
    return intercept + slope * features


# set loss_function() to take the variables as arguments
from tensorflow import keras


def loss_function(intercept, slope, features=size_log, targets=price_log):
    # set the prediction value
    predictions = linear_regression(intercept, slope, features)
    # return the mean squared error loss
    return keras.losses.mse(targets, predictions)


# compute the loss for different slope and intercept values
print(loss_function(0.1, 0.1).numpy())
print(loss_function(0.1, 0.5).numpy())

# 2) train a linear model
# initialize an optimizer
opt = keras.optimizers.Adam(0.5)

# pick up intercept and slope value
import tensorflow as tf

intercept = tf.Variable(5)
slope = tf.Variable(0.001)
print(intercept)
print(slope)

for j in range(100):
    # apply minimize, pass the loss function, and supply variables
    opt.minimize(lambda: loss_function(intercept, slope), var_list=[intercept, slope])

    # print every 10th value of the loss
    if j % 10 == 0:
        print(loss_function(intercept, slope).numpy())

# 3) multiple linear regression
# get your data
price_log = np.array(housing['price'], np.float32)
size_log = np.array(housing['sqft_living'], np.float32)
bedrooms = np.array(housing['bedrooms'], np.float32)
params = tf.Variable([0.1, 0.05, 0.2])


# define the linear regression model


def linear_regression(params, feature1=size_log, feature2=bedrooms):
    return params[0] + feature1 * params[1] + feature2 * params[2]


# define the loss function


def loss_function(params, targets=price_log, feature1=size_log, feature2=bedrooms):
    # set the predicted values
    predictions = linear_regression(params, feature1, feature2)
    # return the mean absolute error loss
    return keras.losses.mae(targets, predictions)


# define the optimize operation
opt = keras.optimizers.Adam()

# perform minimization and print trainable variables
for j in range(10):
    opt.minimize(lambda: loss_function(params), var_list=[params])
    print(loss_function(params).numpy())

# Lesson 2.4 Batch training

# use batch training to handle large datasets
# read_csv() allows us to load data in batches with 'chunksize' parameter
# high level APIs can automate the batch process

# EXERCISE:
# 1) preparing to batch train
# define the intercept and slope
from tensorflow import Variable
from tensorflow import keras

intercept = Variable(10.0, tf.float32)
slope = Variable(0.5, tf.float32)


# define the model


def linear_regression(intercept, slope, features):
    # define the predicted values
    return intercept + slope * features


# define the loss function


def loss_function(intercept, slope, targets, features):
    # define the predicted values
    predictions = linear_regression(intercept, slope, features)
    # define the MSE loss
    return keras.losses.mse(targets, predictions)


# 2) training a linear model in batches
# initialize adam optimizer
opt = keras.optimizers.Adam()

# load data in batches
import pandas as pd
import numpy as np

for batch in pd.read_csv('TensorFlow Datasets/kc_house_data.csv', chunksize=100):
    size_batch = np.array(batch['sqrt_lot'], np.float32)
    # extract the price values for the current batch
    price_batch = np.array(batch['price'], np.float32)
    # compute the loss
    opt.minimize(lambda: loss_function(intercept, slope, price_batch, size_batch),
                 var_list=[intercept, slope])

# print trained parameters
print(intercept.numpy(), slope.numpy())

# CHAPTER 3. Neural Networks


# Lesson 3.1 Dense layers

# we add hidden layer to make linear regressions to neural network
# the process of generating a prediction is called a forward propagation
# a dense layer applies weights to all nodes from the previous layer
# high-level approach: high-level API operations
# low-level approach: linear-algebraic operations

# EXERCISE:
# 1) the linear algebra of dense layers / low-level approach
from tensorflow import Variable, ones, matmul, keras

# initialize bias1
bias1 = Variable(1.0)

# initialize weights1 as 3*2 variable of ones
weights1 = Variable(ones(3, 2))

# perform matrix multiplication of borrower_features and weights1
import numpy as np

borrower_features = np.array([[2., 2., 43.]], np.float32)
product1 = matmul(borrower_features, weights1)

# apply sigmoid activation function to product1 + bias1
dense1 = keras.activations.sigmoid(product1 + bias1)
print("\n dense1's output shape: {}".format(dense1.shape))

# initialize bias2 and weights2
bias2 = Variable(1.0)
weights2 = Variable(ones(2, 1))

# perform matrix multiplication of dense1 and weights2
product2 = matmul(dense1, weights2)

# apply activation to product2 + bias2 and print the prediction
prediction = keras.activations.sigmoid(product2 + bias2)
print('\n prediction: {}'.format(prediction.numpy()[0, 0]))
print('\n actual: 1')

# 2) using the dense layer operation  / high-level approach
from tensorflow import keras

# define the first dense layer
dense1 = keras.layers.Dense(7, activation='sigmoid')(borrower_features)

# define a dense layer with 3 output nodes
dense2 = keras.layers.Dense(3, activation='sigmoid')(dense1)

# define a dense layer with 1 output node
predictions = keras.layers.Dense(1, activation='sigmoid')(dense2)
print('\n shape of dense1: ', dense1.shape)
print('\n shape of dense2: ', dense2.shape)
print('\n shape of predictions: ', predictions.shape)

# Lesson 3.2 Activation functions

# components of a typical hidden layer:
# linear: matrix multiplication
# nonlinear: activation function
# there are three most common activation functions: sigmoid, relu, and softmax
# sigmoid activation function is primarily used on binary classification problem
# softmax activation function is used in the output layer in classification problems
# with more than two classes

# EXERCISE:
# 1) binary classification problem
# get the data
import pandas as pd

credit_card_data = pd.read_csv('TensorFlow Datasets/uci_credit_card.csv')
print(credit_card_data.info())

# get three latest bill_amount
data = credit_card_data[['BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6']]
bill_amounts = data.to_numpy()
print(bill_amounts)

# get targets: default
default = credit_card_data[['default.payment.next.month']]
default = default.to_numpy()
print(default)

# construct input layer from features
from tensorflow import constant

inputs = constant(bill_amounts)

# define first dense layer
from tensorflow import keras

dense1 = keras.layers.Dense(3, activation='relu')(inputs)

# define second dense layer
dense2 = keras.layers.Dense(2, activation='relu')(dense1)

# define output layer
outputs = keras.layers.Dense(1, activation='sigmoid')(dense2)

# print error for first five elements
error = default[:5] - outputs.numpy()[:5]
print(error)

# 2) multiclass classification problem
# construct input layer from borrower features
inputs = constant(borrower_features)

# define first dense layer
dense1 = keras.layers.Dense(10, activation='sigmoid')(inputs)

# define second dense layer
dense2 = keras.layers.Dense(8, activation='relu')(dense1)

# define output layer
outputs = keras.layers.Dense(6, activation='softmax')(dense2)

# print first five predictions
print(outputs.numpy()[:5])

# Lesson 3.3 Optimizers

# minimization problem might go wrong
# you might get stuck in a local minima with gradient descent algorithm
# stochastic gradient descent (SGD) algorithm is an improved version of gradient descent
# SGD is less likely to get stuck in local minima and it is fast
# RMS can apply different learning rates to each feature, useful for high diemnsional problem
# RMS also allows you to build momentum and allows it to decay
# adam can set momentum and decay with beta1 parameter
# adam performs better with default parameter values, it is a good first choice

# EXERCISE:
# 1) the dangers of local minima
# initialize x_1 and x_2
from tensorflow import Variable

x_1 = Variable(6.0, float)
x_2 = Variable(0.3, float)

# define loss_function
import math
from math import pi


def loss_function(x):
    return 4.0 * math.cos(x - 1) + math.cos(2.0 * pi * x) / x


# define the optimization operation
from tensorflow import keras

opt = keras.optimizers.SGD(learning_rate=0.01)

for j in range(100):
    # perform minimization using the loss function and x_1
    opt.minimize(lambda: loss_function(x_1), var_list=[x_1])
    # perform minimization using the loss function and x_2
    opt.minimize(lambda: loss_function(x_2), var_list=[x_2])

print(x_1.numpy(), x_2.numpy())

# 2)avoiding local minima
# initialize x_1 and x_2
from tensorflow import Variable

x_1 = Variable(0.05, float)
x_2 = Variable(0.05, float)

# define the optimization operation for opt1_ and opt_2
opt_1 = keras.optimizers.RMSprop(learning_rate=0.01, momentum=0.99)
opt_2 = keras.optimizers.RMSprop(learning_rate=0.01, momentum=0.00)

for j in range(100):
    opt_1.minimize(lambda: loss_function(x_1), var_list=[x_1])
    opt_2.minimize(lambda: loss_function(x_2), var_list=[x_2])

print(x_1.numpy(), x_2.numpy())

# Lesson 3.4 Training a network in TensorFlow

# in minimizing problem, we often need to initialize thousands of variables
# tf.ones() perform poorly, we draw initial values from distribution, such as normal or uniform
# also, be careful with over-fitting
# you can apply dropout layer to reduce the likely of over-fitting

# EXERCISE:
# 1) initialization in TensorFlow
from tensorflow import Variable, random, ones, keras

# define the layer 1 weights
w1 = Variable(random.normal([23, 7]))

# initialize the layer 1 bias
b1 = Variable(ones([7]))

# define the layer 2 weights
w2 = Variable(random.normal([7, 1]))

# define the layer 2 bias
b2 = Variable(0.0)


# 2) defining the model and loss function
# define the model


def model(w1, b1, w2, b2, features=borrower_features):
    # apply relu activation functions to layer1
    layer1 = keras.activations.relu(matmul(features, w1) + b1)
    # apply dropout
    dropout = keras.layers.Dropout(0.25)(layer1)
    return keras.activations.sigmoid(matmul(dropout, w2) + b2)


# define the loss function


def loss_function(w1, b1, w2, b2, features=borrower_features, targets=default):
    predictions = model(w1, b1, w2, b2)
    # pass targets and predictions to the cross entropy class
    return keras.losses.binary_crossentropy(targets, predictions)


# 3) training neural networks with TensorFlow
# initiate the optimizer
opt = keras.optimizers.adam()

# train the model
for j in range(100):
    # complete the optimizer
    opt.minimize(lambda: loss_function(w1, b1, w2, b2),
                 var_list=[w1, b1, w2, b2])

# make predictions with model
test_features = borrower_features
test_targets = default
model_predictions = model(w1, b1, w2, b2, test_features)

# construct the confusion matrix
from sklearn.metrics import confusion_matrix

confusion_matrix(test_targets, model_predictions)

# CHAPTER 4. High Level APIs


# Lesson 4.1 Defining neural networks with Keras

# use sequential API to define neural network is simple and strong
# the input layer, hidden layer, and output layer are ordered in sequence
# functional API to combine multiple models

# EXERCISE:
# 1) the sequential model in Keras
# import module
from tensorflow import keras

# define a Keras sequential model
model = keras.Sequential()

# define the first dense layer
model.add(keras.layers.Dense(16, activation='relu'), input_shape=(784,))

# define the second dense layer
model.add(keras.layers.Dense(8, activation='relu'))

# define the output layer
model.add(keras.layers.Dense(4, activation='softmax'))

# print the model architecture
print(model.summary())

# 2) compiling a sequential model
# import module
from tensorflow import keras

# define a Keras sequential model
model = keras.Sequential()
# define the first dense layer
model.add(keras.layers.Dense(16, activation='sigmoid'), input_shape=(784,))

# apply dropout to the first layer's output
model.add(keras.layers.Dropout(0.25))

# define the output layer
model.add(keras.layers.Dense(4, activation='softmax'))

# compile the model
model.compile('adam', loss='categorical_crossentropy')

# print a model summary
print(model.summary())

# 3) define a multiple input model
# define input layers for model 1 and model 2
m1_inputs = keras.Input(shape=(28 * 28,))
m2_inputs = keras.Input(shape=(10,))

# for model 1, pass the input layer to layer 1 and layer 1 to layer 2
m1_layer1 = keras.layers.Dense(12, activation='sigmoid')(m1_inputs)
m1_layer2 = keras.layers.Dense(4, activation='softmax')(m1_layer1)

# for model 2, pass the input layer to layer 1 and layer 1 to layer 2
m2_layer1 = keras.layers.Dense(12, activation='relu')(m2_inputs)
m2_layer2 = keras.layers.Dense(4, activation='softmax')(m2_layer1)

# merge model outputs and define a functional model
merged = keras.layers.add([m1_layer2, m2_layer2])
model = keras.Model(inputs=[m1_inputs, m2_inputs], output=merged)

# print a model summary
print(model.summary())

# Lesson 4.2 Training and validation with Keras

# steps of training and evaluation
# 1. load and clean data
# 2. define model
# 3. train and validate model
# 4. evaluate model

# EXERCISE:
# 1) training with Keras
# import module
from tensorflow import keras

# load and get the data
import pandas as pd

data = pd.read_csv('TensorFlow Datasets/slmnist.csv', header=None)
print(data.head(5))
print(data.shape)

sing_language_features = data.iloc[:, 1:].to_numpy()
# print(sing_language_features)
# print(sing_language_features.shape)

sing_language_labels = data.iloc[:, 0]
sing_language_labels = pd.get_dummies(sing_language_labels).to_numpy()
# print(sing_language_labels)
# print(sing_language_labels.shape)

# define a sequential model
model = keras.Sequential()

# define a hidden layer
model.add(keras.layers.Dense(16, activation='relu'), input_shape=(784,))

# define the output layer
model.add(keras.layers.Dense(4, activation='softmax'))

# compile the model
model.compile('SGD', loss='categorial_crossentropy')

# complete the fitting operation
model.fit(sing_language_features, sing_language_labels, epochs=5)

# 2) metrics and validation with Keras
# define sequential model
model = keras.Sequential()

# define the first layer
model.add(keras.layers.Dense(32, activation='sigmoid', input_shape=(784,)))

# add activation function to classifier
model.add(keras.layers.Dense(4, activation='softmax'))

# set the optimizer, loss function, and metrics
model.compile(optimizer='RMSprop', loss='categorical_crossentropy', metrics=['accuracy'])

# add the number of epochs and the validation split
model.fit(sing_language_features, sing_language_labels, epochs=10, validation_split=0.1)

# 3) over-fitting detection
# define sequential model
model = keras.Sequential()

# define the first layer
model.add(keras.layers.Dense(1024, activation='relu', input_shape=(784,)))

# add activation function to classifier
model.add(keras.layers.Dense(4, activation='softmax'))

# set the optimizer, loss function, and metrics
model.compile(optimizer=keras.optimizers.adam(learning_rate=0.001),
              loss='categorical_crossentropy', metrics=['accuracy'])

# complete the model fit operation
model.fit(sing_language_features, sing_language_labels, epochs=50, validation_split=0.5)

# 4) evaluating models
# set two models
small_model = keras.Sequential()
large_model = keras.Sequential()
# set train and test split
train_features, test_features = sing_language_features[:1599, :], sing_language_features[1600:, :]
train_labels, test_labels = sing_language_labels[:1599, :], sing_language_labels[1600:, :]

# evaluate the small model using the train data
small_train = small_model.evaluate(train_features, train_labels)

# evaluate the small model using the test data
small_test = small_model.evaluate(test_features, test_labels)

# evaluate the large model using the train data
large_train = large_model.evaluate(train_features, train_labels)

# evaluate the large model using the test data
large_test = large_model.evaluate(test_features, test_labels)

# print losses
print('\n Small - Train: {}, Test: {}'.format(small_train, small_test))
print('Large - Train: {}, Test: {}'.format(large_train, large_test))


# Lesson 4.3 Training models with the Estimators API

# Estimators API is a high level tensorflow submodule
# it allows for faster deployment with less code
# but it enforces a set of best practices by placing restrictions, lacks flexibility
# steps:
    # 1. define feature columns
    # 2. load and transform data
    # 3. define and train a estimator/ neural network

# EXERCISE:
# 1) preparing to train with Estimators
# import module
from tensorflow import feature_column
import pandas as pd
import numpy as np

# load data
housing = pd.read_csv('TensorFlow Datasets/kc_house_data.csv')
# define feature columns for bedrooms and bathrooms
bedrooms = feature_column.numeric_column("bedrooms")
bathrooms = feature_column.numeric_column("bathrooms")

# define the list of feature columns
feature_list = [bedrooms, bathrooms]


def input_fn():
    # define the labels
    labels = np.array(housing["price"])

    # define the features
    features = {'bedrooms': np.array(housing['bedrooms']),
                'bathrooms': np.array(housing['bathrooms'])}
    return labels, features

# 2) defining Estimators
from tensorflow import estimator
# define the model and set the number of steps
model = estimator.DNNRegressor(feature_columns=feature_list, hidden_units=[2, 2])
model.train(input_fn, steps=1)

# modify to LinearRegressor
model = estimator.LinearRegressor(feature_columns=feature_list)
model.train(input_fn, steps=2)



# Two other useful TensorFlow extensions:
# 1) TensorFlow Hub
# 2) TensorFlow Probability