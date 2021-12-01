from flask import Flask,render_template, redirect, url_for,request
from flask_restful import Api, Resource
from sklearn.feature_extraction.text import CountVectorizer
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import time


app  = Flask(__name__)
api = Api(app)

@app.route('/')
def my_form():
    return render_template('home.html')

@app.route('/',methods =['POST'])
def article_post():
    raw_data = request.form['text']
    # processed_text = text.upper()
    # return {"data":"processed_text"}
    start = time.time()
    n_gram_range = (1, 1)
    stop_words = "english"

    # raw_data = open('test2.txt','r').read()

        # Extract candidate words/phrases
    count = CountVectorizer(ngram_range=n_gram_range,stop_words=stop_words).fit([raw_data])
    candidates = count.get_feature_names_out()


    model = SentenceTransformer('distilbert-base-nli-mean-tokens')
    doc_embedding = model.encode([raw_data])
    candidate_embeddings = model.encode(candidates)
    top_n = 10
    distances = cosine_similarity(doc_embedding, candidate_embeddings)
    keyword = [candidates[index] for index in distances.argsort()[0][-top_n:]]
    # print(keywords)
    print('seconds -------',time.time()-start)
    return ({"keywords":keyword})



# class HelloWorld(Resource):
#     def get(self):
#         return {'data':'Hello World'}

#     def post(self):
#         return {'data':'Posted'}

# api.add_resource(HelloWorld,"/helloworld")

if __name__ == "__main__":
    app.run(debug=True)

