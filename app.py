from flask import Flask, render_template, request, redirect, url_for
import numpy as np 
import xgboost_model

app = Flask(__name__)

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/about')
def about():
    return render_template("about.html")

#Set a post method to yield predictions on page
@app.route('/home')  
def inputForm():
     return render_template("home.html")

@app.route('/input', methods = ['POST']) 
def predict():
    nervous = int(request.form['nervous'])
    panic = int(request.form['panic'])   
    concentrate = int(request.form['concentrate'])
    hope = int(request.form['hope'])
    anger = int(request.form['anger'])
    socialmed = int(request.form['social-media'])
    weight = int(request.form['weight'])
    nightmare = int(request.form['nightmare'])
    negative = int(request.form['negative'])
    blaming = int(request.form['blaming'])
    y_pred = [[nervous, panic, concentrate, hope, anger, socialmed, weight, nightmare, negative, blaming]]

    trained_model = xgboost_model.training_model()
    prediction_value = trained_model.predict(y_pred)
    
    if prediction_value == 0 :
        return render_template('home.html', anxiety="Your result is you might have a symptom of Anxiety", nervous=nervous, panic=panic,
                                concentrate=concentrate, hope=hope, anger=anger, socialmed=socialmed, weight=weight,
                                nightmare=nightmare, negative=negative, blaming=blaming)
    elif prediction_value == 1:
        return render_template('home.html', depress="Your result is you might have a symptom of Depression", nervous=nervous, panic=panic,
                                concentrate=concentrate, hope=hope, anger=anger, socialmed=socialmed, weight=weight,
                                nightmare=nightmare, negative=negative, blaming=blaming)
    elif prediction_value == 2:
        return render_template('home.html', lonely="Your result is you experience Loneliness", nervous=nervous, panic=panic,
                                concentrate=concentrate, hope=hope, anger=anger, socialmed=socialmed, weight=weight,
                                nightmare=nightmare, negative=negative, blaming=blaming)
    elif prediction_value == 4:
        return render_template('home.html', stress="Your result is you experience Stress", nervous=nervous, panic=panic,
                                concentrate=concentrate, hope=hope, anger=anger, socialmed=socialmed, weight=weight,
                                nightmare=nightmare, negative=negative, blaming=blaming)
    else:
        return render_template('home.html', normal="Your result is you are Normal", nervous=nervous, panic=panic,
                                concentrate=concentrate, hope=hope, anger=anger, socialmed=socialmed, weight=weight,
                                nightmare=nightmare, negative=negative, blaming=blaming) 

#Run app
if __name__ == "__main__":
    app.run(debug=True)