from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.utils.class_weight import compute_class_weight
import joblib
import preprocess
import numpy as np
import pandas as pd

# Load and preprocess data
df = preprocess.load_data("data/SMSSpamCollection")

# Convert numerical labels to string labels
df['label'] = df['label'].map({0: 'ham', 1: 'spam'})

# Compute class weights to balance spam and ham
class_weights = compute_class_weight('balanced', classes=np.unique(df['label']), y=df['label'])
class_weight_dict = {'ham': class_weights[0], 'spam': class_weights[1]}

# Create text classification model with adjusted class weights
# stop_words='english' removes common English words that don’t add much value (like "the", "is", "in").
# max_df=0.95 will ignore terms that appear in more than 95% of the messages, which helps with very common words that are likely not useful for classification.
# min_df=5 makes the model ignore words that appear in fewer than 5 messages, helping to reduce noise from rare words.
pipeline = Pipeline([
    ('vectorizer', TfidfVectorizer(stop_words='english', max_df=0.95, min_df=5)),
    ('classifier', MultinomialNB(class_prior=[class_weight_dict['ham'], class_weight_dict['spam']]))
])

print(df['label'].value_counts())  # Print label distribution
print(df.head())  # Print first few rows of the dataset

# Train the model
pipeline.fit(df['message'], df['label'])

# Test the model before saving
test_messages = [
    "Congratulations! You've won a free iPhone!",
    "Claim your free lottery prize now!",
    "Hey, are we still on for lunch tomorrow?",
    "Reminder: Your doctor's appointment is at 3 PM."
]

predictions = pipeline.predict(test_messages)
print("\nTesting Model Predictions:")
for msg, pred in zip(test_messages, predictions):
    print(f"Message: {msg} → Prediction: {pred}")

# Save the trained model
joblib.dump(pipeline, "spam_classifier.pkl")

print("Model training complete. Saved as spam_classifier.pkl")