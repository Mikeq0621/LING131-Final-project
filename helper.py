import os

from util import *

SONG_CORPUS = 'data/song_test.json'
STEMMED_SONG_CORPUS = 'data/song_stemmed.json'
POSTINGS_LIST = 'data/postings_list.json'
TF_IDF_DICT = 'data/tf_idf_dict.json'
TF_IDF_NORM_DICT = 'data/tf_idf_norm_dict.json'
COS_NORM_LIST = 'data/cos_norm_list.json'

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
    # return word in STOPWORDS
    return False


def get_song_corpus():
    """
    get song corpus as dict
    """
    if not os.path.isfile(SONG_CORPUS):
        write_json(SONG_CORPUS, csv2dict('data/songdata.csv'))
    return read_json(SONG_CORPUS)


def get_stemmed_song_corpus():
    if not os.path.isfile(STEMMED_SONG_CORPUS):
        corpus = get_song_corpus()
        for i in corpus:
            corpus[i].pop('link', None)
            corpus[i]['artist'] = [word.lower() for word in word_tokenize(corpus[i]['artist'])]
            corpus[i]['song'] = [word.lower() for word in word_tokenize(corpus[i]['song'])]
            corpus[i]['text'] = [stem(word) for word in word_tokenize(corpus[i]['text'])]
        write_json(STEMMED_SONG_CORPUS, corpus)
    return read_json(STEMMED_SONG_CORPUS)


def get_flat_dict_values(dic):
    res = []
    for i in dic.values():
        res.extend(i)
    return res


def get_postings_list(stemmed_corpus):
    """
    get a postings list from corpus
    """
    if not os.path.isfile(POSTINGS_LIST):
        postings_list = defaultdict(list)  # {token: [id]}
        invert_postings_list = defaultdict(set)  # {id: (token)}, used to expedite append
        for i in stemmed_corpus:
            for token in [token for value in stemmed_corpus[i].values() for token in value]:
                if token not in invert_postings_list[i]:
                    postings_list[token].append(i)
                    invert_postings_list[i].add(token)
        write_json(POSTINGS_LIST, dict(postings_list))
    return read_json(POSTINGS_LIST)


def get_tf_dict(stemmed_corpus):
    """
    get a dict that shows term freq (tf) of a doc

    return: {term: {doc_id: freq}}
    """
    tf_dict = defaultdict(dict)
    for i in stemmed_corpus:
        for token in [token for value in stemmed_corpus[i].values() for token in value]:
            tf_dict[token][i] = tf_dict[token].get(i, 0) + 1
    return dict(tf_dict)


def get_tf_idf_dict(stemmed_corpus):
    """
    get a dict that shows tf*idf of a doc
    W(t,d) = (1 + log10(tf(t,d))) * log10(N / df(t))

    return: {term: {doc_id: tf*idf}}
    """
    N = len(stemmed_corpus)
    tf_dict = get_tf_dict(stemmed_corpus)
    tf_idf_dict = defaultdict(dict)
    for key, val in tf_dict.items():  # key -> term, val -> {doc_id: freq}
        for idx in val:
            tf_idf_dict[key][idx] = (1 + math.log10(tf_dict[key].get(idx))) * math.log10(N / len(tf_dict[key]))
    return dict(tf_idf_dict)


def get_cos_norm(stemmed_corpus):
    if not os.path.isfile(COS_NORM_LIST):
        tf_idf_dict = get_tf_idf_dict(stemmed_corpus)
        cos_norm_list = {}
        for i in stemmed_corpus:
            weight = 0
            for token in [token for value in stemmed_corpus[i].values() for token in value]:
                if not is_stopwords(token):
                    weight += tf_idf_dict[token][i] ** 2
            cos_norm_list[i] = math.sqrt(weight)
        write_json(COS_NORM_LIST, cos_norm_list)
    return read_json(COS_NORM_LIST)


def get_tf_idf_norm_dict(stemmed_corpus):
    """
    weight doc terms using logarithms tf*idf formula with cosine length normalization
    normalization: wi / sqrt(w1^2 + w2^2 + ... + wn^2)

    return: {term: {doc_id: tf*idf w/ norm}}
    """
    if not os.path.isfile(TF_IDF_NORM_DICT):
        tf_idf_dict = get_tf_idf_dict(stemmed_corpus)
        cos_norm_list = get_cos_norm(stemmed_corpus)
        tf_idf_norm_dict = defaultdict(dict)
        for idx in stemmed_corpus:
            for token in [token for value in stemmed_corpus[idx].values() for token in value]:
                tf_idf_norm_dict[token][idx] = tf_idf_dict[token][idx] / cos_norm_list[idx]
        write_json(TF_IDF_NORM_DICT, dict(tf_idf_norm_dict))
    return read_json(TF_IDF_NORM_DICT)


def build_index():
    clear_index()
    get_tf_idf_norm_dict(get_stemmed_song_corpus())


def clear_index():
    for f in [SONG_CORPUS, STEMMED_SONG_CORPUS, POSTINGS_LIST, TF_IDF_DICT, TF_IDF_NORM_DICT, COS_NORM_LIST]:
        if os.path.isfile(f):
            os.remove(f)
