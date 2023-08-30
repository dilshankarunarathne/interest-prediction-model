import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model

# Load your dataset
df = pd.read_csv('generated_dataset.csv')

# Load your trained Keras model
model = load_model('model.h5')


def format_topic(recommended_topic):
    # Remove leading and trailing whitespace
    formatted_topic = recommended_topic.strip()

    return formatted_topic


# Define a function to recommend a topic for a user
def recommend_topic(model, user_age, user_gender):
    # Ensure user_age is an integer
    user_age_input = np.array([[user_age]], dtype=int)

    # Ensure user_gender is encoded as 0 for male and 1 for female
    user_gender_input = np.array([[0 if user_gender == "M" else 1]], dtype=int)

    # Get all unique topics from the dataset
    topics = df['LikedTopic'].unique()

    # Ensure the number of topics matches the model's input shape (50 topics)
    num_topics = 50
    dummy_topics = np.zeros((1, num_topics))  # Placeholder for topics data

    # Use the model to predict topic scores
    topic_scores = model.predict([user_age_input, user_gender_input, dummy_topics])

    # Get the recommended topic index based on the highest score
    recommended_topic_index = np.argmax(topic_scores)
    recommended_topic = topics[recommended_topic_index]

    return recommended_topic


# Example usage:
age_input = input("Enter user's age: ")
gender_input = input("Enter user's gender (M/F): ")

recommended_topics = recommend_topic(model, age_input, gender_input)

if recommended_topics:
    print("Recommended Topics:")
    for topic in recommended_topics:
        print(topic)
else:
    print("No topics recommended.")
