from flask import Flask, render_template, request, url_for
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import joblib
import pickle

app=Flask(__name__)
cv=pickle.load(open('transform.pkl','rb'))
model=pickle.load(open('nlp_model.pkl','rb'))

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        message =request.form['message']
        data=[message]
        vect=cv.transform(data).toarray()
        my_predictions=model.predict(vect)
    return render_template('result.html', predictions=my_predictions)



if __name__=='__main__':
    app.run(debug=True)
