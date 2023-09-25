# Introduction to Deep Learning with Keras




# CHAPTER 1. Introducing Keras


# Lesson 1.1 What is Keras?

# keras is a high level deep learning framework
# it takes less code and enables fast experimentation
# it runs on top of other frameworks

# deep learning works great on unstructured data (image, audio, etc)
# deep learning can handle feature extraction + classification


# Lesson 1.2 Your first neural network

# keras allows you to build models in two different ways:
    # 1) functional API
    # 2) sequential API

# we will focus on sequential API in this course

# EXERCISE:
# 1) hello nets!
# import the sequential model and Dense layer
from keras.models import Sequential
from keras.layers import Dense

# create a sequential model
model = Sequential()

# add an input layer and a hidden layer with 10 neurons
model.add(Dense(10, input_shape=(2,), activation="relu"))

# add a 1-neuron output layer
model.add(Dense(1))

# summarise your model
model.summary()


# Lesson 1.3 Surviving a meteor strike

# a model needs to be compiled before training
# model.compile(optimizer=..., loss=...)

# a model is useless without training
# model.fit(X_train, y_train, epoch=...)

# after training, we can use it for prediction
# model.predict(X_test)

# evaluating model is also important
# model.evaluate(X_test, y_test)

# EXERCISE:
# 1) specifying a model
# instantiate a Sequential model
model = Sequential()

# add a Dense layer with 50 neurons and an input of 1 neuron
model.add(Dense(50, input_shape=(1,), activation='relu'))

# add two Dense layers with 50 neurons and relu activation
model.add(Dense(50, activation='relu'))
model.add(Dense(50, activation='relu'))

# end your model with a Dense layer and no activation
model.add(Dense(1))

# compile your model
model.compile(optimizer='adam', loss='mse')

# fit your model on your data for 30 epochs
import numpy as np
time_steps = np.array([-10.        ,  -9.989995  ,  -9.97998999,  9.97998999,
         9.989995  ,  10.        ])
y_positions = np.array([100.        ,  99.80000005,  99.6002003 ,  99.6002003 ,
        99.80000005, 100.        ])
model.fit(time_steps, y_positions, epochs=30)

# evaluate your model
model.evaluate(time_steps, y_positions)

# predict the twenty minutes orbit
twenty_min_orbit = model.predict(np.arange(-10, 10))





# CHAPTER 2. Going Deeper

# Lesson 2.1 Binary classification

# problems when you predict whether an observation belongs to one of two possible classes
# our dataset is two coordinates for two circles: one inside, one outside
# sigmoid activation function for the last output node, it converts the output to a floating number between 0 and 1
# sigmoid activation is common for output layer in binary classification problems
# when compiling model, use loss='binary_crossentropy' as loss function

# EXERCISE:
# 1) exploring dollar bills (goal is to distinguish real and fake dollar bills)
# import seaborn
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

# use pairplot and set the hue to be our class column
banknotes = pd.read_csv()
sns.pairplot(banknotes, hue='class')
plt.show()

# describe the data
print('Dataset info: \n', banknotes.info())
print('Dataset describe: \n', banknotes.describe())

# count the number of observations per class
print('Observations per class: \n', banknotes['class'].value_counts())

# 2) a binary classification model
# import sequential model and dense layer
from keras.models import Sequential
from keras.layers import Dense

# create a sequential model
model = Sequential()

# add a dense layer
model.add(Dense(1, input_shape=(4,), activation='sigmoid'))

# compile your model
model.compile(loss='binary_crossentropy',
              optimizer='sgd',
              metrics=['accuracy'])
model.summary()

# 3) is this dollar fake?
# split data
from sklearn.model_selection import train_test_split
X = banknotes.drop(['class'], axis=1)
y = banknotes['class']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# train your model for 20 epochs
model.fit(X_train, y_train, epochs=20)

# evaluate your model accuracy on the test set
accuracy = model.evaluate(X_test, y_test)[1]
print('Accuracy: ', accuracy)


# Lesson 2.2 Multi-class classification

# when there are more than two classes, we have a multi-class problem
# we need to change the number of nodes in the output layer
# we also need to make sure the sum of probabilities of all nodes is equal to 1
# we achieve this with the 'softmax' activation function
# when compile the model, we still need the loss function to be 'categorical_crossentropy'
# we also need to convert dataset response variable to categorical variable and do one-hot encoding

# EXERCISE:
# 1) a multi-class model
from keras.models import Sequential
from keras.layers import Dense

# instantiate a Sequential model
model = Sequential()

# add 3 dense layers of 128, 64, and 32 neurons each
model.add(Dense(128, input_shape=(2,), activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))

# add a dense layer with as many neurons as competitors
model.add(Dense(4, activation='softmax'))

# compile your model using categorical_crossentropy loss
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

# 2) prepare your dataset
import pandas as pd
from tensorflow.keras.utils import to_categorical
darts = pd.read_csv('')

# transform into a categorical variable
darts.competitor = pd.Categorical(darts.competitor)

# assign a number to each category (label encoding)
darts.competitor = darts.competitor.categories.codes

coordinates = darts.drop(['competitor'], axis=1)
# use to_categorical on your labels
competitor = to_categorical(darts.competitor)

# 3) training on dart throws
# split data
from sklearn.model_selection import train_test_split
coord_train, coord_test, competitor_train, competitor_test = train_test_split(coordinates, competitor,
                                                                              test_size=0.2, random_state=42)

# fit your model to the training data for 200 epochs
model.fit(coord_train, competitor_train, epochs=200)

# evaluate your model accuracy on the test data
accuracy = model.evaluate(coord_test, competitor_test)[1]
print('Accuracy: ', accuracy)

# 3) softmax predictions
coords_small_test = pd.read_csv('')
competitors_small_test = pd.read_csv('')
# predict on coords_small_test
preds = model.predict(coords_small_test)
print("{:45} | {}".format('Raw Model Predictions','True labels'))
for i,pred in enumerate(preds):
  print("{} | {}".format(pred,competitors_small_test[i]))

# extract the position of highest probability from each pred vector
preds_chosen = [np.argmax(pred) for pred in preds]
print("{:10} | {}".format('Rounded Model Predictions','True labels'))
for i,pred in enumerate(preds_chosen):
  print("{:25} | {}".format(pred,competitors_small_test[i]))


# Lesson 2.3 Multi-label classification

# in multi-label classification, a single input can be assigned to more than one class
# real world example is a movie's genres
# in a multi-label problem, each individual can have all, none, or subset of the available classes
# the difference here is we still use 'sigmoid' activation function in the output layer with more than one output

# EXERCISE:
# 1) an irrigation machine
from keras.models import Sequential
from keras.layers import Dense

# instantiate a Sequential model
model = Sequential()

# add a hidden layer of 64 neurons and a 20 neuron's input
model.add(Dense(64, input_shape=(20,), activation='relu'))

# add a hidden layer of 3 neurons with sigmoid activation
model.add(Dense(3, activation='sigmoid'))

# compile your model with binary crossentropy loss
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])
model.summary()

# 2) training with multiple labels
# get data
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

data = pd.read_csv('')
sensors = data.dropna(['parcel'], axis=1)
parcels = to_categorical(data['parcel'])
sensors_train, sensors_test, parcels_train, parcels_test = train_test_split(sensors, parcels,
                                                                            test_size=0.2, random_state=42)

# train for 100 epochs using a validation split of 0.2
model.fit(sensors_train, parcels_train, epochs=100, validation_split=0.2)

# predict on sensors_test and round up the predictions
import numpy as np
preds = model.predict(sensors_test)
preds_rounded = np.round(preds)
print('Rounded Predictions: \n', preds_rounded)

# evaluate your model's accuracy on the test data
accuracy = model.evaluate(sensors_test, parcels_test)[1]
print('Accuracy: ', accuracy)


# Lesson 2.4 Keras callbacks

# using callbacks to better control and supervise model training
# callback is a block of code that executed after each epoch during training or after training is finished
# we can save the model training history to store these information or display history plots
# early stopping can avoid the over-fitting problem
# model checkpoint can help to identify the best performing model

# EXERCISE:
# 1) the history callback
# train your model and save its history
h_callback = model.fit(X_train, y_train, epochs=50,
                       validation_data=(X_test, y_test))

# plot train vs test loss during training
import matplotlib.pyplot as plt

def plot_loss(data1, data2):
    plt.figure()
    plt.plot(data1)
    plt.plot(data2)
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'])
    plt.show()


plot_loss(h_callback.history['loss'], h_callback.history['val_loss'])

# plot train vs test accuracy during training

def plot_accuracy(data1, data2):
    plt.figure()
    plt.plot(data1)
    plt.plot(data2)
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'])
    plt.show()

plot_accuracy(h_callback.history['acc'], h_callback.history['val_acc'])

# 2) early stopping your model
# import the early stopping callback
from keras.callbacks import EarlyStopping

# define a callback to monitor val_acc
monitor_val_acc = EarlyStopping(monitor='val_acc',
                                patience=5)

# train your model using the early stopping callback
model.fit(X_train, y_train, epochs=100, validation_data=(X_test, y_test), callbacks=[monitor_val_acc])

# 3) a combination of callbacks
# import the EarlyStopping and ModelCheckpoint
from keras.callbacks import EarlyStopping, ModelCheckpoint

# early stop on validation accuracy
monitor_val_acc = EarlyStopping(monitor='val_acc',
                                patience=3)

# save the best model as best_banknote_model.hdf5
modelCheckpoint = ModelCheckpoint('best_banknote_model.hdf5', save_best_only=True)

# fit your model for a stupid amount of epochs
h_callback = model.fit(X_train, y_train, epochs=1000000000000,
                       callbacks=[monitor_val_acc, modelCheckpoint],
                       validation_data=(X_test, y_test))





# CHAPTER 3. Improving Your Model Performance


# Lesson 3.1 Learning curves

# two types of learning curves: loss learning curve, and accuracy learning curve
# loss learning curve goes down as epoch goes
# accuracy learning curve goes up as epoch goes
# these learning curves can tell us a lot of information to help improve our model

# EXERCISE:
# 1) learning the digits
import numpy as np
X_train = np.array([])
X_test = np.array([])
y_train = np.array([])
y_test = np.array([])

# instantiate a Sequential model
from keras.models import Sequential
from keras.layers import Dense
model = Sequential()

# input and hidden layer with input_shape, 16 neurons, and relu
model.add(Dense(16, input_shape=(64,), activation='relu'))

# output layer with 10 neurons (one per digit) and softmax
model.add(Dense(10, activation='softmax'))

# compile your model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# test if your model is well assembled by predicting before training
print(model.predict(X_train))

# 2) is the model over-fitting?
# train your model with 60 epochs, using X_test and y_test as validation data
h_callback = model.fit(X_train, y_train, epochs=60, validation_data=(X_test, y_test), verbose=0)

# extract from the h_callback object loss and val_loss to plot the learning curve
plot_loss(h_callback.history['loss'], h_callback.history['val_loss'])

# 3) do we need more data?
# prepare necessary variables
initial_weight = model.get_weights()
training_sizes = np.array([125, 502, 879, 1255])
early_stop = EarlyStopping(monitor='val_acc',
                                patience=3)
train_accs = []
test_accs = []

for size in training_sizes:
    # get a fraction of training data
    X_train_frac, y_train_frac = X_train[:size], y_train[:size]

    # reset the model to the initial weights and train it on the new training data fraction
    model.set_weights(initial_weight)
    model.fit(X_train_frac, y_train_frac, epochs=50, callbacks=[early_stop])

    # evaluate and store both
    train_accs.append(model.evaluate(X_train_frac, y_train_frac)[1])
    test_accs.append(model.evaluate(X_test, y_test)[1])

plot_accuracy(train_accs, test_accs) # Yes, we need more data!


# Lesson 3.2 Activation functions

# well known activation functions are:
    # sigmoid (0, 1)
    # tanh (-1, 1)
    # relu (0, infinity)
    # leaky_relu (allow for negative values)

# all activation functions come with cons and pros
# there is no easy way to determine which activation function is best to use
# relu is a great first choice
# avoid sigmoid for deep models
# tune with experimentation
# we can compare different activation functions' performance

# EXERCISE:
# 1) comparing activation functions
# activation functions to try
activations = ['relu', 'leaky_relu', 'sigmoid', 'tanh']

# define get_model function
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
np.random.seed(1)

def get_model(act_function):
    model = Sequential()
    model.add(Dense(4, input_shape=(2,), activation=act_function))
    model.add(Dense(1, activation='sigmoid'))
    return model


# create two dictionaries
activation_results = {}
val_loss_per_function = {}
val_acc_per_function = {}

# loop over the activation functions
for act in activations:
    # get a new model with the current activation
    model = get_model(act)
    # fit the model and store the history results
    h_callback = model.fit(X_train, y_train,
                           validation_data=(X_test, y_test),
                           epochs=20, verbose=0)
    activation_results[act] = h_callback
    val_loss_per_function[act] = h_callback['val_loss']
    val_acc_per_function[act] = h_callback['val_acc']



# create dataframes from val_cross_per_function & val_acc_per_function
val_loss = pd.DataFrame(val_loss_per_function)
val_acc = pd.DataFrame(val_acc_per_function)

# call plots on these dataframes
import matplotlib.pyplot as plt
val_loss.plot()
plt.show()
val_acc.plot()
plt.show()


# Lesson 3.3 Batch size and batch normalization

# mini-batches are subsets of data samples
# mini-batches advantages:
    # networks train faster
    # less RAM memory required, can train on huge datasets
    # noise can help networks reach a lower error, escaping local minima
# mini-batches disadvantages:
    # more iterations need to be run
    # need to be adjusted, we need to find a good batch size
# normalization is a common pre-processing step in machine learning algorithms
# normalized_data = (data - mean) / standard_deviation
# normalizing neural networks inputs improve our model
# batch normalization makes sure that independently of the changes, the inputs to the next layer

# batch normalization advantages:
    # improves gradient flow
    # allows higher learning rates
    # reduces dependence on weight initializations
    # acts as an unintended form of regularization
    # limits internal covariate shift

# batch normalization is played in Keras as a layer

# EXERCISE:
# 1) changing batch size
# get a fresh new model with get_model function
model = get_model('relu')

# fit your model
model.fit(X_train, y_train, epochs=5, batch_size=1)
print('\n The accuracy when using batch size of 1 is: ',
      model.evaluate(X_test, y_test)[1])

# change batch_size
model.fit(X_train, y_train, epochs=5, batch_size=len(X_train))
print('\n The accuracy when using the whole training set as batch-size was: ',
      model.evaluate(X_test, y_test)[1])

# 2) batch normalizing a familiar model
# import batch normalization from keras layers
from keras.layers import BatchNormalization
from keras.layers import Dense
from keras.models import Sequential

# build your deep model
batchnorm_model = Sequential()
batchnorm_model.add(Dense(50, input_shape=(64,), activation='relu', kernel_initializer='normal'))
batchnorm_model.add(BatchNormalization())
batchnorm_model.add(Dense(50, activation='relu', kernel_initializer='normal'))
batchnorm_model.add(BatchNormalization())
batchnorm_model.add(Dense(50, input_shape=(64,), activation='relu', kernel_initializer='normal'))
batchnorm_model.add(BatchNormalization())
batchnorm_model.add(Dense(10, activation='softmax', kernel_initializer='normal'))

# compile your model with sgd
batchnorm_model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])

# 3) batch normalization effects
# train your standard model, storing its history callback
h1_callback = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, verbose=0)

# train the batch normalized model you recently built, store its history callback
h2_callback = batchnorm_model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, verbose=0)

# call compare_histories_acc passing in both model histories
import matplotlib.pyplot as plt

def compare_histories_acc(data1, data2):
    plt.figure()
    plt.plot(data1.history['acc'])
    plt.plot(data1.history['loss_acc'])
    plt.plot(data2.history['acc'])
    plt.plot(data2.history['loss_acc'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test', 'Train with batch normalization', 'Test with batch normalization'])
    plt.show()

compare_histories_acc(h1_callback, h2_callback)


# Lesson 3.4 Hyper-parameter tuning

# neural network is full of parameters that can be tweaked
    # number of layers
    # number of neurons per layer
    # layer order
    # layer activations
    # batch size
    # learning rates
    # optimizers
    # ...
# in machine learning, we can use methods like RandomizedSearchCV or GridSearchCV from sklearn module
# we can do the same with our Keras models
# first, we have to transform them into sklearn estimators by creating function that creates our model
# then, we import KerasClassifier wrapper from keras.wrappers.scikit_learn (VERY IMPORTANT)
# from here, our model can work as any other sklearn estimator models
# tips for hyper-parameter tuning:
    # random search is preferred over grid search
    # don't use many epochs
    # use a smaller sample of your dataset
    # play with batch sizes, activations, optimizers and learning rates

# EXERCISE:
# 1) prepare a model for tuning
from keras.optimizers import Adam
# create a model given an activation and learning rate

def create_model(learning_rate, activation):

    # create optimizer
    opt = Adam(lr = learning_rate)

    # create your binary classification model
    model = Sequential()
    model.add(Dense(128, input_shape=(30,), activation=activation))
    model.add(Dense(256, activation=activation))
    model.add(Dense(1, activation='sigmoid'))

    # compile your model
    model.compile(optimizer=opt, loss='binray_crossentropy', metrics=['accuracy'])
    return model

# 2) tuning model parameters
# import KerasClassifier from keras scikit learn wrappers
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import RandomizedSearchCV

# create a KerasClassifier
model = KerasClassifier(build_fn=create_model)

# define the parameters to try out
params = {'activation': ['relu', 'tanh'], 'batch_size':[32, 128, 256], 'epochs':[50, 100, 200],
          'learning_rate':[0.1, 0.01, 0.001]}

# create a randomized search cv object passing in the parameters to try
random_search = RandomizedSearchCV(model, param_distributions=params, cv=3)
random_search_results = random_search.fit(X, y)
print(random_search_results.best_score_, random_search_results.best_params_)

# 3) training with cross-validation
# import KerasClassifier from keras wrappers
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score

# create a KerasClassifier
model = KerasClassifier(build_fn=create_model(learning_rate=0.001, activation='relu'), epochs=50,
                        batch_size=128, verbose=0)

# calculate the accuracy score for each fold
kfolds = cross_val_score(model, X, y, cv=3)
print('The mean accuracy was: ', kfolds.mean())
print('With a standard deviation of:', kfolds.std())





# CHAPTER 4. Advanced Model Architectures


# Lesson 4.1 Tensors, layers and auto-encoders

# we can access Keras layers and this layer's input, output and weights
    # e.g: model.layers[0]
# tensors are main data structures in deep learning, it is a multi-dimensional array of numbers
# auto-encoders are models that aim at producing the same inputs as outputs
# it can help to compress its input into a small set of neurons
# it is useful for things like: dimensionality reduction, de-noising, and anomaly detection and many other applications

# EXERCISE:
# 1) it's a flow of tensors
# import keras backend
import keras.backend as K

# input tensor from the 1st layer of the model
inp = model.layers[0].input

# output tensor from the 1st layer of the model
out = model.layers[0].output

# define a function from inputs to outputs
inp_to_out = K.function([inp], [out])

# print the results of passing X_test through the 1st layer
print(inp_to_out([X_test]))

# 2) neural separation
for i in range(0, 21):
    # train model for 1 epoch
    h = model.fit(X_train, y_train, batch_size=16, epochs=1, verbose=0)
    if i%4 ==0:
        # get the output of the first layer
        layer_output = inp_to_out([X_test])[0]
        
        # evaluate model accuracy for this epoch
        test_accuracy = model.evaluate(X_test, y_test)[1]
        
# 3) building an auto-encoder
# start with a sequential model
autoencoder = Sequential()

# add a dense layer with input the original image pixels and neurons the encoded representation
autoencoder.add(Dense(32, input_shape=(28*28, ), activation='relu'))

# add an output layer with as many neurons as the original image pixels
autoencoder.add(Dense(28*28, activation='sigmoid'))

# compile your model with adadelta
autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')

# summarize your model structure
autoencoder.summary()

# 4) de-noising like an auto-encoder
# build your encoder by using the first layer of your auto-encoder
encoder = Sequential()
encoder.add(autoencoder.layers[0])

# encode the noisy images and show the encodings for your favorite number [0-9]
encodings = encoder.predict(X_test)

# predict the noisy images with your autoencoders
decoded_images = autoencoder.predict(X_test)
# now the noise is gone, the auto-encoder helped us to construct a better image


# Lesson 4.2 Intro to CNNs

# CNNs is short for Convolutional Neural Networks (CNNs)
# a convolutional model uses convolutional layers
# it applies a filter known as kernel of a given size
# convolutional layers perform feature learning, then flatten the output into a vector
# it is used for feature extraction

# EXERCISE:
# 1) building a CNN model
# import the Conv2D and Flatten layers and instantiate model
from keras.layers import Conv2D, Flatten
model = Sequential()

# add a convolutional layer of 32 filters of size 3*3
model.add(Conv2D(32, kernel_size=3, input_shape=(28, 28, 1), activation='relu'))

# add a convolutional layer of 16 filters of size 3*3
model.add(Conv2D(16, kernel_size=3, activation='relu'))

# flatten the previous layer output
model.add(Flatten())

# add as many outputs as classes with softmax activation
model.add(Dense(10, activation='softmax'))

# 2) looking at convolutions
# obtain a reference to the outputs of the first layer
first_layer_output = model.layers[0].output

# build a model using the model's input and the first layer output
first_layer_model = Model(inputs = model.layers[0].input, outputs = first_layer_output)

# use this model to predict on X_test
activations = first_layer_model.predict(X_test)

# plot the activations of first digit of X_test for the 15th filter
axs[0].matshow(activations[, :, :, 14], cmap='viridis')

# do the same but for the 18th filter now
axs[1].matshow(activations[0, :, :, 17], cmap='viridis')
plt.show()

# 3) preparing your input image
# import image and preprocess_input
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input

# load the image with the right target size for your model
img_path = ''
img = image.load_img(img_path, target_size=(224, 224))

# turn it into an array
img_array = image.img_to_array(img)

# expand the dimensions of the image, this is so that it fits the expected model input format
img_expanded = np.expand_dims(img_array, axis=0)

# pre-process the img in the same way original images were
img_ready = preprocess_input(img_expanded)

# 4) using a real world model
# instantiate a ResNet50 model with 'imagenet' weights
from keras.applications.resnet50 import ResNet50, decode_predictions
model = ResNet50(weights='imagenet')

# predict with ResNet50 on your already processed img
preds = model.predict(img_ready)

# decode the first 3 predictions
print('Predicted:', decode_predictions(preds, top=3)[0])


# Lesson 4.3 Intro to LSTMs

# LSTMs is short for Long Short Term Memory networks
# LSTMs are a type of recurrent neural networks, RNN for short
# RNN can use past predictions in order to infer new ones

# LSTM neurons have an internal state that is passed between units, as a memory of past steps
# a unit receives the internal state, an output from previous unit, and a new input at time t
# it updates the state and produces a new output that is returned, passes it as input for following unit

# LSTM units perform several operations
    # learn what to ignore
    # learn what to keep
    # select the most important pieces of past information

# LSTMs are used for image captioning, speech to text, text translation, document summarization, text generation,
# musical composition, and many more

# EXERCISE:

# 1) text predictions with LSTMs
# split text into an array of words
text = 'it is not the strength of the body but the strength of the spirit it is useless to meet revenge with revenge it will heal nothing even the smallest person can change the course of history all we have to decide is what to do with the time that is given us the burned hand teaches best after that advice about fire goes to the heart'
words = text.split()

# make sentences of 4 words each, moving one word at a time
sentences = []
for i in range(4, len(words)):
    sentences.append(' '.join(words[i-4:i]))

# instantiate a Tokenizer, then fit it on the sentences
from keras.preprocessing.text import Tokenizer
tokenizer = Tokenizer()
tokenizer.fit_on_texts(sentences)

# turn sentences into a sequence of numbers
sequences = tokenizer.texts_to_sequences(sentences)
print("Sentences: \n {} \n Sequences: {}".format(sentences[:5], sequences[:5]))

# 2) build your LSTM model
# import the Embedding, LSTM and Dense layer
from keras.layers import Embedding, LSTM, Dense
model = Sequential()
vocab_size = len(tokenizer.index_word) + 1
# add an Embedding layer with the right parameters
model.add(Embedding(input_dim=vocab_size, output_dim=8, input_length=5))

# add a 32 unit LSTM layer
model.add(LSTM(32))

# add a hidden Dense layer of 32 units and an output layer of vocab_size with softmax
model.add(Dense(32, activation='relu'))
model.add(Dense(vocab_size, activation='softmax'))
model.summary()

# 3) decode your predictions

def predict_text(test_text, model=model):
    if len(test_text.split()) !=3:
        print('Text input should be 3 words!')
        return False

    # turn the test_text into a sequence of numbers
    test_seq = tokenizer.texts_to_sequences([test_text])
    test_seq = np.array(test_seq)

    # use the model passed as a parameter to predict the next word
    pred = model.predict(test_seq).argmax(axis=1)[0]

    # return the word that maps to the prediction
    return tokenizer.index_word[pred]

# 4) test your model!
predict_text('meet revenge with')
predict_text('the course of')
predict_text('strength of the')
