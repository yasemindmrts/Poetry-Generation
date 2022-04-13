import random
import warnings
import keras
import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Activation, Dense, LSTM
from syllable import Encoder
from keras.optimizer_experimental.rmsprop import RMSprop
from turkish.deasciifier import Deasciifier

warnings.filterwarnings("ignore")

filepath = tf.keras.utils.get_file('C:\\Users\\User\\PycharmProjects\\Poetry2\\dataset\\PoemsText.txt',
                                   'https://raw.githubusercontent.com/Mrjavaci/Turkish-Poems/main/PoemsText.txt')
# C:\\Users\\Ahmet\\.keras\\datasets\\dataset\\PoemsText.txt
chars = ['\n', '\r', ' ']

text = open(filepath, 'rb').read().decode(encoding='utf-8').lower()
remove = text.translate(str.maketrans('', '', " abcçdefgğhıijklmnoöprsştuüvyz\n\r"))
result = text.translate(str.maketrans('', '', remove))
text = result

vowels = ['a', 'e', 'i', 'o', 'ö', 'u', 'ü']


def countVowels(word):
    count = 0
    for char in word:
        if char in vowels:
            count += 1
    return count


words = text.split()
wordLengths = [len(word) for word in words]
vowelNumberOfWords = [countVowels(word) for word in words]
avgWordLength = round(np.mean(wordLengths))
avgVowel = round(np.mean(vowelNumberOfWords))

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

"""
model = Sequential()
model.add(LSTM(128,
               input_shape=(SEQ_LENGTH,
                            len(characters))))
model.add(Dense(len(characters)))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer=RMSprop(lr=0.01))

model.fit(x, y, batch_size=256, epochs=30)

model.save("C:\\Users\\User\\PycharmProjects\\Poetry2\\savedModel") """

model = keras.models.load_model("C:\\Users\\User\\PycharmProjects\\Poetry2\\savedModel")


def sample(preds, temperature=1.0):
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)


def split(word):
    return [char for char in word]


def generate_text(startSentence, length, temperature, syllable=0):
    generated = ""
    sentence = []
    seed = startSentence.split()

    for i in range(SEQ_LENGTH):
        sentence.append("a")

    seedChars = []
    for word in seed:
        seedChars += split(word)

    for i in range(len(seedChars)):
        sentence[SEQ_LENGTH - i - 1] = seedChars[len(seedChars) - i - 1]

    generated += startSentence
    newSentence = startSentence
    count = 0

    for i in range(length):
        x_predictions = np.zeros((1, SEQ_LENGTH, len(characters)))
        for t, char in enumerate(sentence):
            x_predictions[0, t, char_to_index[char]] = 1

        predictions = model.predict(x_predictions, verbose=0)[0]
        next_index = sample(predictions,
                            temperature)
        next_character = index_to_char[next_index]

        if syllable != 0:
            vowel = ['a', 'e', 'ı', 'i', 'o', 'ö', 'u', 'ü']
            for letter in newSentence:
                if letter in vowel:
                    count += 1

            if next_character != "\r":
                if count == syllable:
                    next_character = "\n"

                elif count < syllable and (next_character == "\n"):
                    next_character = " "

                elif count == syllable - 1 and next_character == " ":
                    next_character = "\n"

                if next_character == "\n":
                    newSentence = ""
                else:
                    newSentence += next_character

                generated += next_character
                sntnc = sentence[1:]
                sntnc.append(next_character)
                sentence = sntnc
            else:
                i = i - 1

            count = 0
        else:
            generated += next_character
            sntnc = sentence[1:]
            sntnc.append(next_character)
            sentence = sntnc
    return generated


print("Write start sentence: ")
sentence = input()
print("Number of line of poem: ")
lineNumber = int(input())
print("Syllable (yes or no):")
ans = input()
syllable = 0

if ans == "yes":
    print("Number of syllable:")
    syllable = int(input())

lineLength = 0
if syllable == 0:
    lineLength = int(avgWordLength * 7)
else:
    lineLength = int((syllable / avgVowel * avgWordLength))

start = sentence
print("Poetry generation started. Please wait... \n\n ")
poetry = (generate_text(start, lineLength * (lineNumber - 1), 0.7, syllable))
print(Deasciifier(poetry).convert_to_turkish())

errorLineNumber = poetry.count('\n')
lineErr = 1
syllableErr = 1

if lineNumber < errorLineNumber:
    lineErr = float((errorLineNumber - lineNumber)/lineNumber)
elif errorLineNumber < lineNumber:
    lineErr = float(errorLineNumber/lineNumber)

if syllable != 0:
    lines = poetry.split("\n")
    for line in lines:
        countVowel = countVowels(line)

        if syllable < countVowel:
            syllableErr = float((countVowel - syllable) / syllable)
        elif countVowel < syllable:
            syllableErr = float(countVowel / syllable)
else:
    syllableErr = 0

print("\nAverage error:",round(float(lineErr+syllableErr)/2,2))
