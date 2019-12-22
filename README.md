# LING131-Final-project: Song Search Engine

Our project is a song search engine based on vector space model and TF-IDF values. Users are required to input a piece of
lyrics with or without artist/title. Search results will be displayed in the descending similarity order. For further 
details, please refer to report.pdf.
    
## How to run
To install dependencies

    $ pip3 install -r requirements.txt

To start the application 
    
    $ python3 app.py

To use the application 
    
    open http://127.0.0.1:5000/ in browser 

To quit the application
    
    press ctrl + c

To rebuild the index

    $ python3 index.py
    
_Due to the indexing process, the first time to startup will take longer time_
