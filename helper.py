import math
import os
from collections import defaultdict

from util import *

SONG_CORPUS = 'data/song.json'
POSTINGS_LIST = 'data/postings_list.json'
TF_IDF_DICT = 'data/tf_idf_dict.json'
COS_NORM_LIST = 'data/cos_norm_list.json'
TF_IDF_NORM_DICT = 'data/tf_idf_norm_dict.json'

STOPWORDS = {'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours', 'yourself',
             'yourselves', 'he', 'him', 'his', 'himself', 'she', 'her', 'hers', 'herself', 'it', 'its', 'itself',
             'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', 'these',
             'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do',
             'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while',
             'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before',
             'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again',
             'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each',
             'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than',
             'too', 'very', 's', 't', 'can', 'will', 'just', 'don', 'should', 'now'}


def is_stopwords(word):
    return word in STOPWORDS


def get_song_corpus():
    """
    get song corpus as dict
    """
    if not os.path.isfile(SONG_CORPUS):
        write_json(SONG_CORPUS, csv2dict('data/songdata.csv'))
    return read_json(SONG_CORPUS)


def get_postings_list(corpus):
    """
    get a postings list from corpus
    """
    if not os.path.isfile(POSTINGS_LIST):
        postings_list = {}
        for i in corpus:
            for token in word_tokenize(corpus[i]['text']):
                token = stem(token)
                if token not in postings_list:
                    postings_list[token] = [i]
                else:
                    if i not in postings_list[token]:
                        postings_list[token].append(i)
        write_json(POSTINGS_LIST, postings_list)
    return read_json(POSTINGS_LIST)


def get_tf_dict(corpus):
    """
    get a dict that shows term freq (tf) of a doc

    return: {term: {doc_id: freq}}
    """
    tf_dict = defaultdict(dict)
    for i in corpus:
        for word in corpus[i]['text']:
            word = word.lower()
            tf_dict[word][i] = tf_dict[word].get(i, 0) + 1
    return dict(tf_dict)


def get_tf_idf_dict(corpus):
    """
    get a dict that shows tf*idf of a doc
    W(t,d) = (1 + log10(tf(t,d))) * log10(N / df(t))

    return: {term: {doc_id: tf*idf}}
    """
    N = len(corpus)
    tf_dict = get_tf_dict(corpus)
    tf_idf_dict = defaultdict(dict)
    for key, val in tf_dict.items():  # key -> term, val -> {doc_id: freq}
        for idx in val:
            tf_idf_dict[key][idx] = (1 + math.log10(tf_dict[key].get(idx))) * math.log10(N / len(tf_dict[key]))
    return dict(tf_idf_dict)


def get_tf_idf_norm_dict(corpus):
    """
    weight doc terms using logarithms tf*idf formula with cosine length normalization
    normalization: wi / sqrt(w1^2 + w2^2 + ... + wn^2)

    return: {term: {doc_id: tf*idf w/ norm}}
    """
    if not os.path.isfile(TF_IDF_NORM_DICT):
        tf_idf_dict = get_tf_idf_dict(corpus)
        tf_idf_norm_dict = defaultdict(dict)
        for idx in corpus:
            tokens = [stem(token) for token in word_tokenize(corpus[idx]['text'])]
            weight = 0
            for token in tokens:
                if not is_stopwords(token):
                    weight += tf_idf_dict[token][idx] ** 2
            for token in tokens:
                tf_idf_norm_dict[token][idx] = tf_idf_dict[token][idx] / math.sqrt(weight)
        write_json(TF_IDF_NORM_DICT, dict(tf_idf_norm_dict))
    return read_json(TF_IDF_NORM_DICT)
