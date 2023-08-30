import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model

# Load your dataset
df = pd.read_csv('generated_dataset.csv')

# Load your trained Keras model
model = load_model('model.h5')


# Define a function to recommend topics for a user
def recommend_topic(model, user_age, user_gender):
    # Create input data for the model
    user_age_input = np.array([[user_age]])
    user_gender_input = np.array([[user_gender]])

    # Get all unique topics from the dataset
    topics = df['LikedTopic'].unique()

    # Generate dummy topics data with all zeros
    num_topics = len(topics)
    dummy_topics = np.zeros((1, num_topics))  # Placeholder for topics data

    # Use the model to predict topic scores
    topic_scores = model.predict([user_age_input, user_gender_input, dummy_topics])

    # Get the recommended topic based on the highest score
    recommended_topic_index = np.argmax(topic_scores)
    recommended_topic = topics[recommended_topic_index]

    return recommended_topic


# Example usage:
user_age_input = input("Enter user's age: ")
user_gender_input = input("Enter user's gender (M/F): ")

recommended_topics = recommend_topic(model, user_age_input, user_gender_input)

if recommended_topics:
    print("Recommended Topics:")
    for topic in recommended_topics:
        print(topic)
else:
    print("No topics recommended.")
