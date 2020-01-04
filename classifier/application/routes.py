from application import app
from flask import request, jsonify
from spam_classifier import classify


@app.route('/classify_text', methods=['POST'])
def classify_text():
    data = request.json
    text = data.get('text') 
    if text is None:
        params = ', '.join(data.keys()) 
        return jsonify({'message': f'Parametr "{params}" is invalid'}), 400 

    else:
        result = classify(text)
        return jsonify({'result': result})