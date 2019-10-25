#!/usr/bin/env python
# coding: utf-8

# In[1]:


import json
import pandas as pd
from sklearn.externals import joblib


# In[2]:


def lambda_handler(event, context):
    print(event['body'])
    
    req = json.loads(event['body'])
    #req = json.loads("{\"fixed acidity\":6.0,\"volatile acidity\":0.34,\"citric acid\":0.24,\"residual sugar\":5.4,\"chlorides\":0.06,\"free sulfur dioxide\":23,\"total sulfur dioxide\":126,\"density\":0.9951,\"pH\":3.25,\"sulphates\":0.44,\"alcohol\":9}")
    input = pd.DataFrame.from_dict(req, orient='index').T

    model = joblib.load('wine.pkl')
    result = model.predict(input)[0]
    print(f"result: {result}")
    
    return {
        'isBase64Encoded': False,
        'statusCode': 200,
        'headers': {},
        'body': json.dumps({
            "quality": result.item()
        })
    }