from sklearn.preprocessing import LabelEncoder
import joblib

with open('unique_topics.txt', 'r') as file:
    unique_topics = file.read().splitlines()

# Create a LabelEncoder
label_encoder = LabelEncoder()

# Fit the LabelEncoder with the unique topics
label_encoder.fit(unique_topics)

# Save the LabelEncoder to a file for future use
joblib.dump(label_encoder, 'label_encoder.joblib')