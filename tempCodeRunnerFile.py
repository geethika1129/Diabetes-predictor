from flask import Flask,render_template,request
import pickle
import numpy as np

app=Flask(__name__)
sc=pickle.load('sc.pkl','rb')
model=pickle.load(open('classifier.pkl','rb'))

@app.route('/home')
def home():
    return render_template('main.html')

@app.route('/predict',methods=['POST'])
def predict():
    float_features=[float(x) for x in request.form.values()]
    final_features=[np.array(float_features)]
    pred=model.predict(sc.transform(final_features))
    return render_template('result.html',prediction=pred)

if __name__=="__main__":
    app.run(debug=True)