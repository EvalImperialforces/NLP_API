from tfidf_assets import tfidf
from tfidf_esrc import tfidf_esrc
from flask import jsonify, request
from flask import Flask
import json
from flask_cors import CORS


app = Flask(__name__)
CORS(app)

################## Model Functions ######################

@app.route("/")

@app.route("/api/tfidf_assets",  methods=['GET', 'POST'])
def best_match_asset():
    
    try:
        query = request.json['query']
        
        return tfidf(query)
            
    except:
        return 'Query not submitted'

@app.route("/api/links",  methods=['GET', 'POST'])
def best_match_links():
    
    try:
        query = request.json['query']
        
        return links(query)
            
    except:
        return 'Query not submitted'

@app.route("/api/tfidf_esrc",  methods=['GET', 'POST'])
def best_match_esrc():
    
    try:
        question = request.json['question']
        
        return tfidf_esrc(question)
            
    except:
        return 'Query not submitted'

   
#@app.route("/api/bert", methods=['POST'])
#def esrc_bert():
#    try:
#        question = request.json['esrc_question']
#        no_answers = request.json['no_answers']
        
#        return bert_answers(question, no_answers)
#    except:
#        return 'Question not submitted'

#@app.route("/api/cdQA", methods=["GET"])
#def cdQA():

#    query = request.args.get("query")
#    n_predictions = request.json['no_answers']
    
#    prediction = make_prediction(query, n_predictions)

#    return jsonify(
#        query=query, answer=prediction[0], title=prediction[1], paragraph=prediction[2]
#    )



if __name__ == "__main__":
    app.run(debug=True)