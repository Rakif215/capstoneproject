from flask import Flask, render_template, request
import pandas as pd
import joblib
from openai import OpenAI
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

app = Flask(__name__)


client = OpenAI(api_key="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")

bilstm_tokenizer = Tokenizer()
bilstm_tokenizer.fit_on_texts(pd.read_csv("data/train.csv"))
bilstm_model = load_model('model/bi_lstm_model.h5')

multioutput_model = joblib.load('model/model2.joblib')
vectorizer_tfidf = joblib.load('model/vectorizer.joblib')

# Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/score', methods=['POST'])
def score():
    if request.method == 'POST':
        input_text = request.form['input_text']
        # Vectorize the processed text
        essay_tfIdf = vectorizer_tfidf.transform([input_text])
        # Make predictions
        predictions = multioutput_model.predict(essay_tfIdf)
        # Extract prediction scores
        second_model_prediction = predictions

        # Process input_text using the BiL"STM model
        sequence = bilstm_tokenizer.texts_to_sequences([input_text])
        padded_sequence = pad_sequences(sequence, maxlen=bilstm_model.input_shape[1], padding='post')
        prediction = bilstm_model.predict(padded_sequence)
        
        # Process input_text using the second model (replace this with your actual second model)

        return render_template('score.html', input_text=input_text, 
                               prediction=prediction[0], 
                               second_model_prediction=second_model_prediction[0]
                              )

@app.route('/process_text', methods=['POST'])
def process_text():
    if request.method == 'POST':
        input_text = request.form['input_text']

        prompt = "Correct the grammar of  the following essay: "+ input_text 
        completion = client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            model="gpt-3.5-turbo-1106",
            response_format={"type": "text"},
        )
        corrected_essay = completion.choices[0].message.content.replace("\n", "")
        # here 
        return render_template('process_text.html', input_text=input_text , corrected_essay=corrected_essay)

@app.route('/error', methods=['GET'])
def error():
    return render_template('error.html')

if __name__ == '__main__':
    app.run(debug=False)
