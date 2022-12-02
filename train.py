import nltk
nltk.download('omw-1.4')
nltk.download('punkt')
nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
import json
import pickle

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras.optimizers import SGD


import random

#tensorboard
%load_ext tensorboard
from datetime import datetime
from packaging import version

import tensorflow as tf
from tensorflow import keras
from keras import backend as K

import numpy as np


words=[]
classes = []
documents = []
ignore_words = ['?', '!']
data_file = open('intents.json').read()
intents = json.loads(data_file)


for intent in intents['intents']:
    for pattern in intent['patterns']:

        # take each word and tokenize it
        w = nltk.word_tokenize(pattern)
        words.extend(w)
        # adding documents
        documents.append((w, intent['tag']))

        # adding classes to our class list
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_words]
words = sorted(list(set(words)))

classes = sorted(list(set(classes)))

print (len(documents), "documents")

print (len(classes), "classes", classes)

print (len(words), "unique lemmatized words", words)


pickle.dump(words,open('words.pkl','wb'))
pickle.dump(classes,open('classes.pkl','wb'))

# initializing training data
training = []
output_empty = [0] * len(classes)
for doc in documents:
    # initializing bag of words
    bag = []
    # list of tokenized words for the pattern
    pattern_words = doc[0]
    # lemmatize each word - create base word, in attempt to represent related words
    pattern_words = [lemmatizer.lemmatize(word.lower()) for word in pattern_words]
    # create our bag of words array with 1, if word match found in current pattern
    for w in words:
    bag.append(1) if w in pattern_words else bag.append(0)

    # output is a '0' for each tag and '1' for current tag (for each pattern)
    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1

    training.append([bag, output_row])
# shuffle our features and turn into np.array
random.shuffle(training)
training = np.array(training)
# create train and test lists. X - patterns, Y - intents
X = list(training[:,0])
y = list(training[:,1])

print("Training data created")

#log tensorboard
logdir = "logs/scalars/" + datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logdir)

# Create model - 3 layers. First layer 128 neurons, second layer 64 neurons and 3rd output layer contains number of neurons
# equal to number of intents to predict output intent with softmax
model = Sequential()
model.add(Dense(128, input_shape=(len(X[0]),), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5)) #tambah
model.add(Dense(len(y[0]), activation='softmax'))


# Compile model. Stochastic gradient descent with Nesterov accelerated gradient gives good results for this model
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
model.summary()


#fitting and saving the model
#hist = model.fit(np.array(X), np.array(y), epochs=200, batch_size=5, verbose=1, 
                 #validation_data=(X, y), callbacks=[tensorboard_callback],)

from sklearn.model_selection import train_test_split
#train_size = 0.8
X_train, X_test, y_train, y_test = train_test_split(X,y, train_size=0.8)
#test_size = 0.5
X_valid, X_test, y_valid, y_test = train_test_split(X_test,y_test, test_size=0.5)

hist = model.fit(np.array(X_train), np.array(y_train), epochs=300, batch_size=5, verbose=1,
                 validation_data=(X_valid, y_valid),
                 callbacks=[tensorboard_callback],)


#X_train, X_test, y_train, y_test = train_test_split(
    #X_valid, y_valid, test_size=0.1, random_state=0)

#hist = model.fit(np.array(X_train), np.array(y_train), epochs=200, batch_size=5, verbose=1, 
                 #validation_data=(X_valid, y_valid), callbacks=[tensorboard_callback],)



#hist = model.fit(np.array(train_x), np.array(train_y), epochs=200, batch_size=5, verbose=1, 
                 #validation_data=(train_x, train_y), callbacks=[tensorboard_callback],)



#Save model h5 to drive
model.save('/content/drive/MyDrive/Colab Notebooks/Chatbot/model h5/chatbot_model.h5', hist) 


print("model created")