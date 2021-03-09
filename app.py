
# coding: utf-8

# In[ ]:


import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import pandas as pd

app = Flask(__name__)
model = pickle.load(open('pred.sav', 'rb'))

@app.route('/')
def home():
    return render_template('pred.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    int_features = [int(x) for x in request.form.values()]
    index_counter=0
    us_dict={}
    x=['location', 'gender', 'age_desc', 'time_desc', 'chance']
    for c in x:
        us_dict[c] = int_features[index_counter]
        index_counter= index_counter + 1

    
    user_input = pd.DataFrame(data=us_dict,index=[0])
    pred = model.predict_proba(user_input).round(3)[0]
    cr=['kiddnap', 'murder', 'rape', 'snatching', 'theft']
    pr=dict(zip(cr,pred))
    
    output = pr

    return render_template('output.html', prediction_text='crime rate should be for {}'.format(output))


if __name__ == "__main__":
    app.run(debug=True)

