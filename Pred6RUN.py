"""This file loads the model
and uses Tkinter to add GUI to the Text Predictor
we used Pickle to save the model and used the epic fantasy novel Legend as our dataset"""
from Tkinter import *
import numpy as np
from keras.models import load_model
import pickle
import heapq


bText=[' ', ' ', ' ', ' ', ' ']
# Creating the main GUI window
root = Tk()
root.title("Text Predictor")
# This is the font that will be used throughout the program
large_font = ('Verdana', 40)
# Creating Tkinter objects for display
entry1Var = StringVar(value='Large Font!')
display = StringVar()
output_window = Label(root, textvariable=display, font=large_font)
output_window.pack()
entry1 = Entry(root, font=large_font, textvariable=display)
# These are the StringVar() instances which will be updates each tim the user presses enter
svarB1= StringVar()
svarB2= StringVar()
svarB3= StringVar()
svarB4= StringVar()
svarB5= StringVar()
LEGEND = "/home/amna/Downloads/Gemmel, David - [Drenai Saga 1] - Legend .txt"
text = open(LEGEND).read().lower()

chars = sorted(list(set(text)))
# These are the 2 dictionaries we are using in our code
# The first maps each character to a integer number
char_indices = dict((c, i) for i, c in enumerate(chars))
# The second maps the integer back to the character
indices_char = dict((i, c) for i, c in enumerate(chars))

# Printing some details of our data set
print('corpus length:', len(text))
print('unique chars:', {len(chars)})

# The sequence length is the hard-coded length of the string of words the model uses at a time
SEQUENCE_LENGTH = 40
# The time shift has been taken as 3 to reduce the number of training examples
# and consequently the training time
step = 3

"""We're loading the model we have trained earlier
That model ha been defined in the pred6Legend.py file and has been saved using pickle"""
model = load_model('keras_model_legend6.h5')
history = pickle.load(open("history_legend6.p", "rb"))

# Let's put our cooking to the TEST
def prepare_input(text):
    """This function inputs a string variable entered by the user
    It maps all the characters to their corresponding integer values in the char_indices dictionary
    It returns a numpy array of size [1 x length of a sequence x number of characters]
    for this model, that is [1 x 40 x 624816]"""
    x = np.zeros((1, SEQUENCE_LENGTH, len(chars)))

    for t, char in enumerate(text):
        x[0, t, char_indices[char]] = 1.

    return x

    # SAMPLE our finest delicacies
def sample(preds, top_n=3):
    """This function finds the likelihood of each prediction
    It then finds the probabilities of each prediction by dividing by the sum of total probabilities
    It uses a maxheap to return the 5 words with the highest probability"""
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds)
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    # heapq is used to implement the max heap which returns the 5 words with greatest probability
    return heapq.nlargest(top_n, range(len(preds)), preds.take)

def predict_completion(text):
    """This function takes the input of the string entered by the user
    It then shapes the input according to the neural network requirements by calling the prepare_input function
    It then uses the model to predict a single prediction for the next character
    It keeps predicting repeatedly until a space or fullstop is predicted and atleast 2 chars have been predicted"""
    original_text = text
    generated = text
    completion = ''
    # This loop predicts characters repeatedly to get a single word with multiple characters
    # It is crucial step in our character level model
    # It keeps predicting repeatedly until a space or fullstop is predicted and atleast 2 chars have been predicted
    while True:
        x = prepare_input(text)
        preds = model.predict(x, verbose=0)[0]
        next_index = sample(preds, top_n=1)[0]
        next_char = indices_char[next_index]

        text = text[1:] + next_char
        completion += next_char

        if len(original_text + completion) + 2 > len(original_text) and (next_char == ' ' or next_char == '.'):
            return completion
# multiple predictons
def predict_completions(text, n=3):
    """This function takes the text string variable entered by the user as input
    and also the number of predictions required which is 3 by default
    For our model we pass the value 5 to n to obtain 5 predictions for each word
    It calls the predict_completion function as many times as required"""
    x = prepare_input(text)
    preds = model.predict(x, verbose=0)[0]
    next_indices = sample(preds, n)
    return [indices_char[idx] + predict_completion(text[1:] + indices_char[idx]) for idx in next_indices]

b = ' '
def chooseOption(n):
    global b
    a = display.get()
    if n == 1:
        b = svarB1.get()
    if n == 2:
        b=svarB2.get()
    if n == 3:
        b = svarB3.get()
    if n == 4:
        b = svarB4.get()
    if n == 5:
        b = svarB5.get()
    a = a+b
    display.set(a)

# button = Tk.Button(master=frame, text='press', command= lambda: action(someNumber))
# Defining the buttons for predictions
b1 = Button(root, textvariable=svarB1, bg="blue", font=large_font, padx=80, pady=40, command= lambda: chooseOption(1))
b2 = Button(root, textvariable=svarB2, bg="blue", font=large_font, padx=80, pady=40, command= lambda: chooseOption(2))
b3 = Button(root, textvariable=svarB3, bg="blue", font=large_font, padx=80, pady=40, command= lambda: chooseOption(3))
b4 = Button(root, textvariable=svarB4, bg="blue", font=large_font, padx=80, pady=40, command= lambda: chooseOption(4))
b5 = Button(root, textvariable=svarB5, bg="blue", font=large_font, padx=80, pady=40, command= lambda: chooseOption(5))

def main():
    global bText
    # Taking the text entered by the user in the entry box
    q = entry1.get()
    # Clipping the text t get the first 40 chars in lowercase
    if q < 40:
        seq = q[:40].lower()
    else:
        # if longer sentence is entered, take 40 most recently inputted characters
        seq = q[-40:].lower()
    print(seq)
    # The list bText stored the 5 predictions predicted by the predict_completion button
    bText = predict_completions(seq, 5)
    # Setting the strings in the bText list as the value for Tkinter's StringVar() instances
    svarB1.set(bText[0])
    svarB2.set(bText[1])
    svarB3.set(bText[2])
    svarB4.set(bText[3])
    svarB5.set(bText[4])

    # Displaying the prediction buttons on the page
    b1.pack(side=LEFT)
    b2.pack(side=LEFT)
    b3.pack(side=LEFT)
    b4.pack(side=LEFT)
    b5.pack(side=LEFT)

def pressEnter():
    """The main function is called when the user presses Submit button"""
    main()



#  Creating and displaying the output label and the submit button
l1 = Label(root, text="Type here:", height=10, width=30, font=large_font)
submit = Button(root, text="Enter", font=large_font, command=pressEnter)

l1.pack(side=TOP)
entry1.pack(side=TOP, fill=X)
submit.pack(side=TOP)


root.mainloop()




