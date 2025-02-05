import joblib
from flask import Flask, render_template, request

app = Flask(__name__)

# Load trained model (which includes the vectorizer)
model = joblib.load("spam_classifier.pkl")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    message = request.form.get('message', '')

    if not message:
        return render_template('index.html', prediction="Please enter a message.")

    # Debug: Print the message and prediction flow
    print(f"Received message: {message}")
    
    # Use the model to predict
    prediction = model.predict([message])[0]
    
    # Debug: Print the prediction result
    print(f"Prediction: {prediction}")
    
    result = "Spam" if prediction == 'spam' else "Not Spam"

    return render_template('index.html', prediction=result)

if __name__ == '__main__':
    app.run(debug=True)