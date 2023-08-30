import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model

# Load your dataset (replace 'your_dataset.csv' with your actual dataset)
df = pd.read_csv('generated_dataset.csv')

# Preprocess the data
max_topics_length = 50
tokenizer = Tokenizer()
tokenizer.fit_on_texts(df['LikedTopics'])
topics_seq = tokenizer.texts_to_sequences(df['LikedTopics'])
topics_pad = pad_sequences(topics_seq, maxlen=max_topics_length)

# Load your trained Keras model (replace 'model.h5' with the path to your actual model file)
model = load_model('model.h5')

# Define input data for making predictions
X_user_age = df['UserAge']
X_user_gender = df['UserGender']
X_topics = topics_pad

# Make predictions using the model
predicted_scores = model.predict([X_user_age, X_user_gender, X_topics])

# Define a threshold value for recommendations (adjust as needed)
threshold = 0.5

# Get recommended topics based on the threshold
recommended_topics = [topic for i, topic in enumerate(df['LikedTopics']) if predicted_scores[i] >= threshold]

# Print the recommended topics
print("Recommended Topics:")
for topic in recommended_topics:
    print(topic)
