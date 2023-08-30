import numpy as np
from tensorflow.python.keras.models import load_model

# Load your dataset and model (replace with your actual dataset and model loading code)
input_data = np.load('input_data.npy')
model = load_model('model.h5')

# Define a range of threshold values
threshold_values = np.arange(0.1, 1.1, 0.1)

# Initialize an empty dictionary to store results (threshold value => recommended topics)
threshold_to_topics = {}

# Iterate through threshold values
for threshold in threshold_values:
    # Apply the threshold to the model's output (replace with your actual model prediction code)
    predicted_scores = model.predict(input_data)  # Replace input_data with actual input
    recommended_topics = [topic for topic, score in topic_scores.items() if score >= threshold]

    # Store the threshold value and recommended topics
    threshold_to_topics[threshold] = recommended_topics

# Print the results
for threshold, topics in threshold_to_topics.items():
    print(f'Threshold: {threshold}, Recommended Topics: {topics}')
