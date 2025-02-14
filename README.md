# Spam Guard 3000

Spam Guard 3000 is a Flask-based web application that detects spam messages using a machine learning model trained on SMS spam data.

## 🚀 Features

- 🏷️ Classifies messages as "Spam" or "Not Spam"
- 🏗 Trained using Naïve Bayes with TF-IDF vectorization
- 🌐 Simple web UI for input and classification

## 📦 Setup Instructions

1. Clone the repository:

```bash
git clone https://github.com/MylesJMoore/SpamGuard3000.git
cd spam-guard-3000
```

2. Install Dependencies
   Ensure you have Python installed. Then, run:
   pip install -r requirements.txt
3. Train the Model
   python model.py
4. Run the Flask App
   Start the Flask server:
   python app.py
5. Visit http://127.0.0.1:5000/ in your browser.

## 🛠 Tech Stack

- Flask – Web framework
- Scikit-learn – Machine learning library
- NLTK – Text processing
- Pandas – Data handling
- Joblib – Model persistence

## 🔍 How It Works

1. The model.py script loads and processes SMS spam data.
2. It trains a Naïve Bayes classifier with TF-IDF vectorization.
3. The trained model is saved as spam_classifier.pkl.
4. The app.py script loads this model and provides a web interface for message classification.

## 📝 Example Predictions

Message Expected Classification Model Output
"Congratulations! You've won a free iPhone!" Spam Spam
"Hey, are we still on for lunch tomorrow?" Not Spam Not Spam

## 📜 License

This project is open-source and free to use.
