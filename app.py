import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template 
import pickle

app = Flask(__name__)
model = pickle.load(open('gymModel.pkl', 'rb'))

@app.route('/', methods=["POST"])
def home():
    
    data = request.get_json(force=True)
    
    array = np.array([[int(data['height']), int(data['weight']), int(data['age']), int(data['gender'])]])
    #array = np.array([[174, 72, 22, 1]])

    index_values = [0]

    column_values = ['height', 'weight', 'age', 'gender']

    df = pd.DataFrame(data = array, 
                      index = index_values, 
                      columns = column_values)
    
    prediction = model.predict(df)
    
    return jsonify(int(prediction[0]))

if __name__ == '__main__':
    app.run()