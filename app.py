from flask import Flask,render_template,request
import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()

app=Flask(__name__)
model=pickle.load(open('classifier.pkl','rb'))

@app.route('/')
def KK():
    return render_template('main.html')

@app.route('/predict',methods=['POST'])
def home():
    aa=request.form['a']
    bb=request.form['b']
    cc=request.form['c']
    dd=request.form['d']
    ee=request.form['e']
    ff=request.form['f']
    gg=request.form['g']
    hh=request.form['h']
    data=(aa,bb,cc,dd,ee,ff,gg,hh)
    arr=np.asarray(data)
    arrre=arr.reshape(1,-1)
    
   
    pred=model.predict([arrre])[0]
    return render_template('result.html',prediction=pred)

if __name__=="__main__":
    app.run(debug=True)

    