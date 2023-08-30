from sklearn.preprocessing import LabelEncoder

# Read the unique topics from the text file
with open('unique_topics.txt', 'r') as file:
    unique_topics = [line.strip() for line in file.readlines()]

# Create the LabelEncoder and fit it to the unique topics
label_encoder = LabelEncoder()
label_encoder.fit(unique_topics)

# Save the label_encoder using joblib
import joblib
joblib.dump(label_encoder, 'label_encoder.joblib')

print("LabelEncoder created and saved.")
# TODO - this code actually wrecks the encoder saved during training