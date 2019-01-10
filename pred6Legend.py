""""This Text Prediction model has been drained on Gemmel's Legend, a 345 page Epic Fantasy novel
Starting time 8:28"""
import numpy as np
import tensorflow as tf
from keras.models import Sequential, load_model
from keras.layers import LSTM
from keras.layers.core import Dense, Activation
from keras.optimizers import RMSprop, Adadelta, Adam
import matplotlib.pyplot as plt
import pickle
import seaborn as sns
from pylab import rcParams

np.random.seed(42)
tf.set_random_seed(42)

# Adjusting the settings for the performance graphs
sns.set(style='darkgrid', palette='deep', font='sans-serif', font_scale=1.5)
rcParams['figure.figsize'] = 12, 5

"""The data-set we are using is the book Legend by David Gemmel
It has 345 pages and 624816 characters after removing special characters """

LEGEND = "/home/amna/Downloads/Gemmel, David - [Drenai Saga 1] - Legend .txt"
# All uppercase letters are converted to lowercase to reduce the vocabulary the model has to learn
text = open(LEGEND).read().lower()


chars = sorted(list(set(text)))
# These are the 2 dictionaries we are using in our code
# The first maps each character to a integer number
char_indices = dict((c, i) for i, c in enumerate(chars))
# The second maps the integer back to the character
indices_char = dict((i, c) for i, c in enumerate(chars))


# The sequence length is the hard-coded length of the string of words the model uses at a time
SEQUENCE_LENGTH = 40
# The time shift has been taken as 3 to reduce the number of training examples
# and consequently the training time
step = 3
# Creating an empty list to store the training examples
sentences = []
# Creating another empty list to store the next character associated with each training example
next_chars = []

# Iterating over the entire data set to create distinct 100 character string patterns
# Each pattern is shifted by 1 character from its adjacent pattern
for i in range(0, len(text) - SEQUENCE_LENGTH, step):
    sentences.append(text[i: i + SEQUENCE_LENGTH])
    next_chars.append(text[i + SEQUENCE_LENGTH])

# Details of the data set
print('unique chars:', {len(chars)})
print('num training examples:', {len(sentences)})

# Creating numpy arrays for the input X and output y
# X = [# of training example x length of each example x # of possible values]
X = np.zeros((len(sentences), SEQUENCE_LENGTH, len(chars)), dtype=np.bool)
# y = [# of training examples x #of possible values]
y = np.zeros((len(sentences), len(chars)), dtype=np.bool)
"""Since we have a character level model, the sequence_length for y will be 1, as only 1 character is predicted at a time.
Clearly defining the sizes of the numpy arrays at this stage prevents unexpected errors when we feed these to the LSTM layer"""

# Now we iterate over the lists[[]] of sentences and the list[] of characters to copy the training examples into X and y
for i, sentence in enumerate(sentences):
    for t, char in enumerate(sentence):
        X[i, t, char_indices[char]] = 1
    y[i, char_indices[next_chars[i]]] = 1

# We can print the shape of X and y to check that they are as expected
print('X size:  ', X.shape)
print('y size:  ', y.shape)

# Printing a random sentence from the book and its corresponding next_char
print('A random sentence sampled', sentences[49])
print('The next char is:  ', next_chars[49])

"""We now define the architecture of our model,
It is difficult to choose an optimum between training time and accuracy"""

# The Keras sequential model is a linear stack of layers
model = Sequential()
# We add 128 LSTM layers in our DEEP Learning model
model.add(LSTM(128, input_shape=(SEQUENCE_LENGTH, len(chars))))
# This is followed by a dense layer
model.add(Dense(len(chars)))
# soft max activation function is used in our final layer
# The softmax function takes an un-normalized vector, and normalizes it into a probability distribution
model.add(Activation('softmax'))

# These are the different tyoes of optimizers which are commonly in use
optimizer = RMSprop(lr=0.01)
# These are the default values of parameters for these models
opt2 = Adadelta(lr=1.0, rho=0.95, epsilon=None, decay=0.0)
opt3 = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
"""For our particular problem, Adam Optimizer was found to have the best result
Loss is the quantity the model tries to reduce with training, cross_entropy loss is 
frequently chosen
The metric chosen is accuracy"""
model.compile(loss='categorical_crossentropy', optimizer=opt2, metrics=['accuracy'])

# The fit function is used to train the model
# We iterate over the data set in batches of 128 for 20 epochs
history = model.fit(X, y, validation_split=0.05, batch_size=128, epochs=20, shuffle=True).history

# Now, finally, we save our models so that the weights do not have to be re-calculated each time it is run
model.save('keras_model_legend6.h5')
pickle.dump(history, open("history_legend6.p", "wb"))

# Loading the model to check it is working
model = load_model('keras_model_legend6.h5')
history = pickle.load(open("history_legend6.p", "rb"))

"""We plot some graphs to visually see how well our model is doing
The first graph plots accuracy of both training and testing predictions against number of epochs
We should see an increasing trend for a model which is correctly learning
Examining this graph will also allow us to check for overfitting"""
plt.plot(history['acc'])
plt.plot(history['val_acc'])
plt.title('ACCURACY')
plt.ylabel('Accuracy')
plt.xlabel('Iterations')
plt.legend(['training', 'testing'], loc='upper left');
plt.show()

"""The second graph plots error for both training and testing predictions against number of epochs
We should see a decreasing trend for a model which is correctly learning"""
plt.plot(history['loss'])
plt.plot(history['val_loss'])
plt.title('LOSS')
plt.ylabel('Loss')
plt.xlabel('Iterations')
plt.legend(['training', 'testing'], loc='upper left');
plt.show()

