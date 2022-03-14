import json
import os
import pathlib
import string

import nltk
import zeyrek

lemmas = {}
analyzer = zeyrek.MorphAnalyzer()


# RUN IF NECESSARY
#nltk.download('punkt')

def createLemma(words):
    result = []
    for word in words:
        word = word.lower()
        result.append((analyzer.lemmatize(word))[0][1][0])
        try:
            lemmas[(analyzer.lemmatize(word))[0][1][0]].add(word)
        except:
            lemmas[(analyzer.lemmatize(word))[0][1][0]] = {word}
    return result


def readDataset():
    path = "dataset"
    dir_list = os.listdir(path)

    for file in dir_list:
        f = open(path + '/' + file, encoding="utf8")
        suffix = pathlib.Path(file).suffix

        if suffix == ".json":
            loadedData = json.load(f)
            for i in range(0, len(loadedData)):
                poem = []
                for sentence in loadedData[i]['icerik']:
                    sentence = sentence.translate(str.maketrans('', '', string.punctuation))
                    words = sentence.split()
                    poem = poem + words
                lemmatizedPoem = createLemma(poem)
                print(lemmatizedPoem)

        """elif suffix == ".txt":
            line = f.readline()
            words = line.split()
            words = list(set(words) - {'.', ',', '/', '"', "-"})
            createLemma(words)"""

        f.close()


def main():
    if __name__ == "__main__":
        readDataset()


main()
