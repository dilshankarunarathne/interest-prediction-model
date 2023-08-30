import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import load_model

# Load your dataset (replace 'your_dataset.csv' with your actual dataset)
df = pd.read_csv('your_dataset.csv')

# Preprocess the data
max_topics_length = 50
tokenizer = Tokenizer()
tokenizer.fit_on_texts(df['LikedTopics'])
topics_seq = tokenizer.texts_to_sequences(df['LikedTopics'])
topics_pad = pad_sequences(topics_seq, maxlen=max_topics_length)

# Encode UserGender
encoder = LabelEncoder()
df['UserGender'] = encoder.fit_transform(df['UserGender'])

# Define your input data
X = [df['UserAge'], df['UserGender'], topics_pad]

# Load your trained model (replace 'model.h5' with your actual model file)
# You need to provide custom_objects to handle the custom optimizer
model = load_model('model.h5', custom_objects={'CustomAdamOptimizer': 'adam'})

# Define a range of threshold values
threshold_values = np.arange(0.1, 1.1, 0.1)

# Initialize an empty dictionary to store results (threshold value => recommended topics)
threshold_to_topics = {}

# Iterate through threshold values
for threshold in threshold_values:
    # Apply the threshold to the model's output
    predicted_scores = model.predict(X)
    recommended_topics = [topic for topic, score in zip(df['LikedTopics'], predicted_scores) if score >= threshold]

    # Store the threshold value and recommended topics
    threshold_to_topics[threshold] = recommended_topics

# Print the results
for threshold, topics in threshold_to_topics.items():
    print(f'Threshold: {threshold}, Recommended Topics: {topics}')
