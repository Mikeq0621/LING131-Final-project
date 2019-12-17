import csv
import json

import nltk
from nltk.corpus import wordnet
from nltk.corpus.reader.wordnet import *



def word_tokenize(text):
    return nltk.re.findall(r"""(
            \w+         # word 
            |\.\.\.     # elipsis
            |\n{2,}     # white lines
            |-+         # dashes
            |[(){}[\]]  # brackets
            |[!?.,:;]   # punctuation
            |['"`]      # quotes
            |\S+        # fall through pattern
            )""",
                           text, nltk.re.VERBOSE)


def lemmatizer(word, pos):
    # Need to specify pos=NOUN|ADJ|VERB|
    if pos==None:
        lemmas = wordnet._morphy(word, NOUN)
    else:
        lemmas = wordnet._morphy(word, pos)
    return min(lemmas, key=len) if lemmas else word


def stem(word):
    return nltk.SnowballStemmer("english").stem(word)


def read_json(file):
    with open(file) as f:
        return json.load(f)


def write_json(file, obj):
    with open(file, 'w') as f:
        json.dump(obj, f)


def csv2dict(file):
    dic = {}
    with open(file, newline='') as f:
        reader = csv.DictReader(f)
        for idx, row in enumerate(reader):
            dic[idx] = dict(row)
    return dic


def read_cvs(path):
    song_lyrics = []
    with open(path) as file:
        songdata = csv.reader(file, delimiter=",")
        for row in songdata:
            song_lyrics.append(row[3])
    return song_lyrics[1:-1]


if __name__ == '__main__':
    lyrics = read_cvs("data/songdata.csv")
    print(lemmatizer("dogs", NOUN))
