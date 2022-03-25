import random
import warnings
import keras
import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Activation, Dense, LSTM
from syllable import Encoder

# https://www.neuralnine.com/generating-texts-with-recurrent-neural-networks-in-python/
from keras.optimizer_experimental.rmsprop import RMSprop

warnings.filterwarnings("ignore")

filepath = tf.keras.utils.get_file('C:\\Users\\User\\PycharmProjects\\Poetry2\\dataset\\PoemsText.txt',
                                   'https://raw.githubusercontent.com/Mrjavaci/Turkish-Poems/main/PoemsText.txt')
# C:\\Users\\Ahmet\\.keras\\datasets\\dataset\\PoemsText.txt
text = open(filepath, 'rb').read().decode(encoding='utf-8').lower()

characters = sorted(set(text))

char_to_index = dict((c, i) for i, c in enumerate(characters))
index_to_char = dict((i, c) for i, c in enumerate(characters))

SEQ_LENGTH = 40
STEP_SIZE = 3

sentences = []
next_char = []

for i in range(0, len(text) - SEQ_LENGTH, STEP_SIZE):
    sentences.append(text[i: i + SEQ_LENGTH])
    next_char.append(text[i + SEQ_LENGTH])

x = np.zeros((len(sentences), SEQ_LENGTH,
              len(characters)), dtype=np.bool)
y = np.zeros((len(sentences),
              len(characters)), dtype=np.bool)

for i, satz in enumerate(sentences):
    for t, char in enumerate(satz):
        x[i, t, char_to_index[char]] = 1
    y[i, char_to_index[next_char[i]]] = 1

model = Sequential()
model.add(LSTM(128,
               input_shape=(SEQ_LENGTH,
                            len(characters))))
model.add(Dense(len(characters)))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer=RMSprop(lr=0.01))

model.fit(x, y, batch_size=256, epochs=20)

model.save("C:\\Users\\User\\PycharmProjects\\Poetry2")


def sample(preds, temperature=1.0):
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)

def split(word):
    return [char for char in word]

def generate_text(startSentence, length, temperature):
    generated = ""
    sentence = []
    seed = startSentence.split()
    for i in range(SEQ_LENGTH):
        sentence.append("a")

    seedChars = []
    for word in seed:
        seedChars += split(word)

    for i in range(len(seedChars)):
        sentence[SEQ_LENGTH-i-1] = seedChars[len(seedChars)-i-1]

    generated += startSentence

    for i in range(length):
        x_predictions = np.zeros((1, SEQ_LENGTH, len(characters)))
        for t, char in enumerate(sentence):
            x_predictions[0, t, char_to_index[char]] = 1

        predictions = model.predict(x_predictions, verbose=0)[0]
        next_index = sample(predictions,
                                 temperature)
        next_character = index_to_char[next_index]

        generated += next_character
        sntnc = sentence[1:]
        sntnc.append(next_character)
        sentence = sntnc
    return generated


print(generate_text("ah bu ben kendimi nerelere atsam", 300, 0.2))
"""print(generate_text(300, 0.4))
print(generate_text(300, 0.5))
print(generate_text(300, 0.6))
print(generate_text(300, 0.7))
print(generate_text(300, 0.8))"""
