import heapq

from helper import *

corpus = get_song_corpus()
stemmed_corpus = get_stemmed_song_corpus()
postings_list = get_postings_list(stemmed_corpus)
cos_norm_list = get_cos_norm(stemmed_corpus)
tf_idf_norm_dict = get_tf_idf_norm_dict(stemmed_corpus)


def parse_query_str(query_str):
    return [term for term in [stem(token) for token in word_tokenize(query_str)] if
            (not is_stopwords(term) and term in postings_list)]

    # query_terms = []
    # skipped_terms = []
    # unknown_terms = []
    # for term in [stem(token) for token in word_tokenize(query)]:
    #     if is_stopwords(term):
    #         skipped_terms.append(term)
    #     elif term not in postings_list:
    #         unknown_terms.append(term)
    #     else:
    #         query_terms.append(term)
    # return query_terms, skipped_terms, unknown_terms


def get_largest_cos_score(query_terms, largest=10):
    scores = {}
    for term in query_terms:
        # query_tf_idf = get_tf_idf_dict({0: {'text': query_terms}})
        for idx in postings_list.get(term, []):
            if idx not in scores:
                scores[idx] = [0, []]
            # scores[idx][0] += query_tf_idf[term][0] * tf_idf_norm_dict[term][idx]
            scores[idx][0] += tf_idf_norm_dict[term][idx]
            scores[idx][1].append(term)
    for idx in scores:
        scores[idx][0] /= cos_norm_list[idx]
        scores[idx][1] = [term for term in query_terms if term not in scores[idx][1]]
    largest_score = heapq.nlargest(largest, scores.items(), lambda x: [1][0])
    return largest_score, len(scores)


if __name__ == '__main__':
    s = 'hello how are you'
    get_largest_cos_score(parse_query_str(s))
