from flask import Flask, render_template, request
from query import *

app = Flask(__name__)


@app.route('/')
def query():
    return render_template('query_page.html')


@app.route("/result", methods=['POST'])
def results():
    """Generate a result set for a query and present the 10 results starting with <page_num>."""

    query_str = request.form['query']
    page_num = int(request.form['page_num'])

    matched_res, num_hits = get_largest_score_doc(parse_query_str(query_str), page_num)

    # (song id, song name, artist, lyric snippet, result score, matched terms)
    docs_data = [get_doc_snippet(doc) for doc in matched_res]

    return render_template('results_page.html', query_str=query_str, docs_data=docs_data, page_num=page_num,
                           matched_num=len(matched_res), total_hits=num_hits)


@app.route('/song/<song_id>')
def movie_data(song_id):
    """Given the doc_id for a film, present the title and text (optionally structured fields as well)
    for the movie."""
    data = get_corpus_data(song_id)  # Get all of the info for a single movie
    data['text'] = data['text'].replace('\n', '<br>')
    return render_template('doc_data_page.html', data=data)


if __name__ == '__main__':
    app.run(debug=True)
