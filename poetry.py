import json
import os
import pathlib

import nltk
import zeyrek

lemmas = {}
analyzer = zeyrek.MorphAnalyzer()


# RUN IF NECESSARY
# nltk.download('punkt')

def createLemma(words):
    for word in words:
        lemmas[word] = (analyzer.lemmatize(word))[0][1]


def readDataset():
    path = "dataset"
    dir_list = os.listdir(path)

    for file in dir_list:
        f = open(path + '/' + file)
        suffix = pathlib.Path(file).suffix

        if suffix == ".json":
            loadedData = json.load(f)
            for i in range(0, len(loadedData)):
                for word in loadedData[i]['icerik']:
                    words = word.split()
                    words = list(set(words) - {'.', ','})
                    createLemma(words)

        elif suffix == ".txt":
            line = f.readline()
            words = line.split()
            words = list(set(words) - {'.', ',', '/', '"', "-"})
            createLemma(words)

        f.close()


def main():
    if __name__ == "__main__":
        readDataset()


main()
