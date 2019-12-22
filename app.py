from flask import Flask, render_template, request, jsonify
from query import *

app = Flask(__name__)


@app.route('/')
def query():
    return render_template('query_page.html')


@app.route('/query')
def results():
    query_str = request.args.get('query_str', '', type=str)
    page_num = request.args.get('page_num', 1, type=int)

    matched_res, num_hits = get_largest_score_doc(parse_query_str(query_str), page_num)
    docs_data = [get_doc_snippet(doc) for doc in matched_res]
    card_html = render_template('result_card.html', docs_data=docs_data, matched_num=len(matched_res))

    return jsonify(query_str=query_str, total_hits=num_hits, page_num=page_num, card_html=card_html)


@app.route('/song/<song_id>')
def song_data(song_id):
    data = get_corpus_data(song_id)  # Get all of the info for a single movie
    data['text'] = data['text'].replace('\n', '<br>')
    return render_template('doc_data_page.html', data=data)


if __name__ == '__main__':
    app.run()
