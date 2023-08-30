import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model

# Load your dataset (replace 'your_dataset.csv' with your actual dataset)
df = pd.read_csv('generated_dataset.csv')

# Load your trained Keras model (replace 'model.h5' with the path to your actual model file)
model = load_model('model.h5')


# Define a function to recommend topics based on user input
def recommend_topics(user_age, user_gender):
    # Ensure user age is numeric
    user_age = pd.to_numeric(user_age, errors='coerce')

    # Encode user gender (assuming 'M' is 0 and 'F' is 1)
    user_gender = 0 if user_gender == 'M' else 1

    # Preprocess the data for prediction
    max_topics_length = 50
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(df['LikedTopic'])

    # Create a dummy sequence for topics (you can modify this based on user input)
    dummy_topic_sequence = tokenizer.texts_to_sequences(['your_dummy_topic'])
    dummy_topic_padded = pad_sequences(dummy_topic_sequence, maxlen=max_topics_length)

    # Use the model to predict topic scores
    predicted_scores = model.predict([[user_age], [user_gender], dummy_topic_padded])

    # Define a threshold for recommendations (adjust as needed)
    threshold = 0.5

    # Get recommended topics based on the threshold
    recommended_topics = [topic for i, topic in enumerate(df['LikedTopic']) if predicted_scores[i] >= threshold]

    return recommended_topics


# Example usage:
user_age_input = input("Enter user's age: ")
user_gender_input = input("Enter user's gender (M/F): ")

recommended_topics = recommend_topics(user_age_input, user_gender_input)

if recommended_topics:
    print("Recommended Topics:")
    for topic in recommended_topics:
        print(topic)
else:
    print("No topics recommended.")
