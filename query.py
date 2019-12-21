import datetime
import heapq

from helper import *

now = datetime.datetime.now()
clear_index()
corpus = get_song_corpus()
stemmed_corpus = get_stemmed_song_corpus()
postings_list = get_postings_list(stemmed_corpus)
cos_norm_list = get_cos_norm(stemmed_corpus)
tf_idf_norm_dict = get_tf_idf_norm_dict(stemmed_corpus)
print(datetime.datetime.now() - now)


def parse_query_str(query_str):
    return [term for term in [stem(token) for token in word_tokenize(query_str)] if
            (not is_stopwords(term) and term in postings_list)]


def get_largest_score_doc(query_terms, page_idx, largest=10):
    matched_docs = {}  # {id: [score, [matched_terms]]}
    for term in query_terms:
        # query_tf_idf = get_tf_idf_dict({0: {'text': query_terms}})
        for i in postings_list.get(term, []):
            if i not in matched_docs:
                matched_docs[i] = [0, []]
            # matched_docs[i][0] += query_tf_idf[term][0] * tf_idf_norm_dict[term][i]
            matched_docs[i][0] += tf_idf_norm_dict[term][i]
            matched_docs[i][1].append(term)
    # for i in matched_docs:
    # matched_docs[i][0] /= cos_norm_list[i]
    # matched_docs[i][1] = [term for term in query_terms if term not in matched_docs[i][1]]

    largest_score_docs = heapq.nlargest(page_idx * largest, matched_docs.items(), lambda x: x[1][0])  # sort by score
    return largest_score_docs, len(matched_docs)


def get_corpus_data(i):
    return corpus.get(i)


def highlight_snippet(text, matched_terms):
    token = text.split()
    for i, w in enumerate(token):
        if stem(w) in matched_terms:
            token[i] = '<mark>%s</mark>' % w
    return ' '.join(token)


def get_doc_snippet(doc):
    """
    Return a snippet for the results page.
    Needs to include a title and a short description.
    Your snippet does not have to include any query terms, but you may want to think about implementing
    that feature. Consider the effect of normalization of index terms (e.g., stemming), which will affect
    the ease of matching query terms to words in the text.
    """
    data = get_corpus_data(doc[0])
    matched_terms = set([stem(token) for token in doc[1][1]])
    song = highlight_snippet(data['song'], matched_terms)
    artist = highlight_snippet(data['artist'], matched_terms)
    text = highlight_snippet(data['text'].replace('\n', '<br>'), matched_terms)
    # song id, song name, artist, lyric snippet, result score, matched terms
    return doc[0], song, artist, text, round(doc[1][0], 5), doc[1][1]
