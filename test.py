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

    # Generate dummy topics data (you can replace this with actual topics)
    num_topics = 50  # Adjust this based on your dataset
    dummy_topics = np.zeros((1, num_topics))  # Placeholder for topics data

    # Use the model to predict topic scores
    topic_scores = model.predict([user_age_input, user_gender_input, dummy_topics])

    # Assuming you have a list of topics, you can select the topic with the highest score
    topics = ["Topic 1", "Topic 2", "Topic 3", ...]  # Replace with your actual topics
    recommended_topic_data = topics[np.argmax(topic_scores)]

    return recommended_topic_data


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
