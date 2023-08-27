# Deep Learning In Python
# Build multiple-input and multiple-output deep learning models using Keras
# Change directory
import os
os.chdir('E:/DataCampProjects/Deep Learning')



# CHAPTER 1. The Keras Functional API

# this chapter is a refresher to familiarize the functional APIs in Keras with simple models
# dataset: college basketball games

# Lesson 1.1 Keras input and dense layers

#　simple deep learning models:
    # input layer
    # output layer

# layers are used to construct deep learning models
# tensors are used to describe the data flows in deep learning models

# EXERCISE:
# 1) input layers & dense layers
# import Input, Dense from keras.layers
from keras.layers import Input, Dense

# input layer
input_tensor = Input(shape=(1, ))

# dense layer
output_layer = Dense(1)

# connect the dense layer to the input_tensor
output_tensor = output_layer(input_tensor)


# Lesson 1.2 Build and compile a model

# we can turn layers into a real model that can be used to predict new data

# EXERCISE:
# 1) build a model
# Input/dense/output layers
from keras.layers import Input, Dense
input_tensor = Input(shape=(1, ))
output_tensor = Dense(1)(input_tensor)

# build the model
from keras.models import Model
model = Model(input_tensor, output_tensor)

# 2) compile a model
model.compile(optimizer='adam', loss='mean_absolute_error')

# 3) visualize a model
# import the plotting function
from keras.utils import plot_model
import matplotlib.pyplot as plt

# summarize the model
model.summary()

# plot the model
plot_model(model, to_file='model.png')
data = plt.imread('model.png')
plt.imshow(data)
plt.show()


# Lesson 1.3 Fit and evaluate a model

# we have college basketball tournament dataset
# we have an input: seed difference
# we need an output: score difference

# we also need to evaluate model using a new dataset with .evaluate() method

# EXERCISE:
# 1) fit the model to the tournament basketball data
# get data
import pandas as pd
games_tourney = pd.read_csv('Deep Learning in Python Datasets/basketball_data/games_tourney.csv')
print(games_tourney.head())
print(games_tourney.shape) # (4324, 9)

# split the data
games_tourney_train = games_tourney.iloc[:3430, :]
games_tourney_test = games_tourney.iloc[3430:, :]
print(games_tourney_train.shape)
print(games_tourney_test.shape)

# build model
from keras.layers import Input, Dense
from keras.models import Model
input_tensor = Input(shape=(1,))
output_tensor = Dense(1)(input_tensor)
model = Model(input_tensor, output_tensor)

# fit the model
model.fit(games_tourney_train['seed_diff'],
          games_tourney_train['score_diff'],
          epochs=1,
          batch_size=128,
          validation_split=0.1,
          verbose=True)

# 2) evaluate the model on a test set
# load the X variable from the test data
X_test = games_tourney_test['seed_diff']

# load the Y variable from the test data
y_test = games_tourney_test['score_diff']

# evaluate the model on the test data
print(model.evaluate(X_test, y_test, verbose=False))





# CHAPTER 2. Two Input Networks Using Categorical Embeddings, Shared Layers, and Merge Layers


# Lesson 2.1 Category embeddings

# input: integer (team ID)
# output: floats (team strength)

# create an embedding layer with Embedding() function
# the embedding layer increases the dimension by adding third dimension
# after embedding layer, we need to Flatten the layer from 3D to 2D with Flatten()
# the Flatten layer will be output layer

# EXERCISE:
# 1) define team lookup
# get the data
import pandas as pd
games_season = pd.read_csv('Deep Learning in Python Datasets/basketball_data/games_season.csv')
print(games_season.head())
print(games_season.shape)

# imports
from keras.layers import Embedding
from numpy import unique

# count the unique number of teams
n_teams = unique(games_season['team_1']).shape[0]

# create an embedding layer
team_lookup = Embedding(input_dim=n_teams,
                        output_dim=1,
                        input_length=1,
                        name='Team-Strength')

# 2) define team model
# imports
from keras.layers import Input, Embedding, Flatten
from keras.models import Model

# create an input layer for the team ID
teamid_in = Input(shape=(1,))

# lookup the input in the team strength embedding layer
strength_lookup = team_lookup(teamid_in)

# flatten the output
strength_lookup_flat = Flatten()(strength_lookup)

# combine the operations into a single, re-usable model
team_strength_model = Model(teamid_in, strength_lookup_flat, name='Team-Strength-Model')


# Lesson 2.2 Shared layers

# we will have two inputs: one for each team
# we want them to have the same embedding layer, we use shared layer
# EXERCISE:
# 1) define two inputs
# load the input layer from keras.layers
from keras.layers import Input

# input layer for team 1
team_in_1 = Input(shape=(1,), name='Team-1-In')

# separate input layer for team 2
team_in_2 = Input(shape=(1,), name='Team-2-In')

# lookup team 1 in the team strength model
team_1_strength = team_strength_model(team_in_1)

# lookup team 2 in the team strength model
team_2_strength = team_strength_model(team_in_2)


# Lesson 2.3 Merge layers

# when you need to combine multiple inputs into a single layer to a single output
# this requires a Merge layer
# this gives you a lot of flexibility to creatively design networks to solve problems
# different kinds of merge layers:
    # add
    # subtract
    # multiply
    # concatenate
# only concatenate can operate on layers that have different numbers of columns
# other operations can only apply to layers that have same numbers of columns


# EXERCISE:
# 1) output layer using shared layer
# import the Subtract layer from keras
from keras.layers import Subtract

# create a subtract layer using the inputs from the previous exercise
score_diff = Subtract()([team_1_strength, team_2_strength])

# 2) model using two inputs and one output
# imports
from keras.layers import Subtract
from keras.models import Model

# subtraction layer from previous exercise
score_diff = Subtract()([team_1_strength, team_2_strength])

# create the model
model = Model([team_in_1, team_in_2], score_diff)

# compile the model
model.compile(optimizer='adam', loss='mean_absolute_error')

# Lesson 2.4 Predict from your model

# fit with multiple inputs

# EXERCISE:
# 1) fit the model to the regular season training data
# get the team_1 column from the regular season data
input_1 = games_season['team_1']

# get the team_2 column from the regular season data
input_2 = games_season['team_2']

# fit the model to input 1 and 2, using score diff as a target
model.fit([input_1, input_2],
          games_season['score_diff'],
          epochs=1,
          batch_size=2048,
          validation_split=0.1,
          verbose=True)

# 2) evaluate the model on the tournament test data
# get team_1 from the tournament data
input_1 = games_tourney['team_1']

# get team_2 from the tournament data
input_2 = games_tourney['team_2']

# evaluate the model using these inputs
print(model.evaluate([input_1, input_2], games_tourney['score_diff'], verbose=False))


# CHAPTER 3. Multiple Inputs: 3 Inputs (and Beyond!)

# Lesson 3.1 Three-input models

# creating three-input models is the same as two-input models
# you can also create shared layers with three inputs
# in model fittting process, you also need to have three inputs

# exercise:
from keras.layers import Input, Concatenate

# 1) making an input layer for home vs. away
# create an Input for each team
team_in_1 = Input(shape=(1,), name='Team-1-In')
team_in_2 = Input(shape=(1,), name='Team-2-In')

# create an Input for home vs. away
home_in = Input(shape=(1,), name='Home-In')

# lookup the team inputs in the team strength model
team_1_strength = team_strength_model(team_in_1)
team_2_strength = team_strength_model(team_in_2)

# combine the team strengths with the home input using a Concatenate layer, then a Dense layer
out = Concatenate()([team_1_strength, team_2_strength, home_in])
out = Dense(1)(out)

# 2) make a model and compile it
from keras.models import Model

# make a model
model = Model([team_in_1, team_in_2, home_in], out)

# compile the model
model.compile(optimizer='adam', loss='mean_absolute_error')

# fit the model to the games_season dataset
model.fit([games_season['team_1'], games_season['team_2'], games_season['home']],
          games_season['score_diff'],
          epochs=1,
          verbose=True,
          validation_split=0.1,
          batch_size=2048)

# evaluate the model on the games_tourney dataset
print(model.evaluate([games_tourney['team_1'], games_tourney['team_2'], games_tourney['home']],
                     games_tourney['score_diff'],
                     verbose=False))


# Lesson 3.2 Summarizing and plotting models

# model summary shows you all the layers, and how many parameters each layer has
# Keras model has both untrainable and trainable parameters
# model plot can help you understand the sequences of the model

# exercise:
# 1) model summaries
model.summary()

# 2) plotting models
import matplotlib.pyplot as plt
from keras.utils import plot_model

# plot the model
plot_model(model, to_file='model.png')

# display the image
data = plt.imread('model.png')
plt.imshow(data)
plt.show()


# Lesson 3.3 Stacking models

# model stacking is to use the predictions from one model as an input to another model
# stacking models requires 2 datasets, it is important to use different datasets for each model
#

# exercise:
# 1) add model predictions to the tournament data
# predict
games_tourney['pred'] = model.predict([games_tourney['team_1,'], games_tourney['team_2'],
                                       games_tourney['home']])

# 2) create an input layer with multiple columns
from keras.layers import Input, Dense
# create an input layer with 3 columns
input_tensor = Input(shape=(3,))

# output layer
output_tensor = Dense(1)(input_tensor)

# create a model
model = Model(input_tensor, output_tensor)

# compile the model
model.compile(optimizer='adam', loss='mean_absolute_error')

# fit the model
model.fit(games_tourney_train[['home', 'seed_diff', 'pred']],
          games_tourney_train['score_diff'],
          epochs=1,
          verbose=True)

# evaluate the model
print(model.evaluate(games_tourney_test[['home', 'seed_diff', 'pred']],
                     games_tourney_test['score_diff'], verbose=False))





# CHAPTER 4. Multiple Outputs

# Lesson 4.1 Two-output models

# you can use a single model to have two outputs, it can even be both a classifier and regressor
# everything is similar, the only difference is the size of output layer

# exercise:
# 1) simple two-output model
from keras.layers import Input, Dense

# define the input
input_tensor = Input(shape=(2,))

# define the output
output_tensor = Dense(2)(input_tensor)

# create a model
model = Model(input_tensor, output_tensor)

# compile the model
model.compile(optimizer='adam', loss='mean_absolute_error')

# fit a model with two outputs
# fit the model
model.fit(games_tourney_train[['seed_diff', 'pred']],
          games_tourney_train[['score_1', 'score_2']],
          verbose=True,
          epochs=100,
          batch_size=16384)

# inspect the model
# print the model's weights
print(model.get_weights())

# print the column means of the training data
print(games_tourney_train.mean())

# evaluate the model on the tournament test data
print(model.evaluate(games_tourney_test[['seed_diff', 'pred']],
                     games_tourney_test[['score_1', 'score_2']],
                     verbose=False))


# Lesson 4.2 Single model for classification and regression

# in this case, we build two one-node output layer separately rather than one two-node output layer

# exercise:
# 1) classification and regression in one model
from keras.layers import Input, Dense
from keras.models import Model
# create an input layer with 2 columns
input_tensor = Input(shape=(2,))

# create the first output
output_tensor_1 = Dense(1, activation='linear', use_bias=False)(input_tensor)

# create the second output (use the first output as input here)
output_tensor_2 = Dense(1, activation='sigmoid', use_bias=False)(output_tensor_1)

# create a model with 2 outputs
model = Model(input_tensor, [output_tensor_1, output_tensor_2])

# compile and fit the model
# import the Adam optimizer
from keras.optimizers import Adam

# compile the model with 2 losses and the Adam optimizer with a high learning rate
model.compile(loss=['mean_absolute_error', 'binary_crossentropy'], optimizer=Adam(lr=0.01))

# fit the model to the tournament training data, with 2 inputs and 2 outputs
model.fit(games_tourney_train[['seed_diff', 'pred']],
          games_tourney_train[['score_diff']], games_tourney_train[['won']],
          epochs=10,
          verbose=True,
          batch_size=16384)

# inspect the model
# print the model weights
print(model.get_weights())

# print the training data means
print(games_tourney_train.mean())

# import the sigmoid function from scipy
from scipy.special import expit as sigmoid

# weight from the model
weight = 0.14

# print the approximate win probability predicted close game
print(sigmoid(1 * weight))

# print the approximate win probability predicted blowout game
print(sigmoid(10 * weight))

# evaluate on new data with two metrics
# evaluate the model on new data
print(model.evaluate(games_tourney_test[['seed_diff', 'pred']],
                     [games_tourney_test[['score_diff']], games_tourney_test[['won']]],
                     verbose=False))


# Lesson 4.3 Wrap-up

# So far,
# functional API
# shared layers (useful for making comparisons)
# multiple inputs (can be used for different kinds of input data, like text, numeric data, and images)
# multiple outputs (can do BOTH regression and classification)

