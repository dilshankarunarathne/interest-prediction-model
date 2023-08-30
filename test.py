import warnings
import joblib
import numpy as np

# Load the trained model
model = joblib.load('model.joblib')

# Try to load the label encoder, or use a default one if not found
label_encoder_path = 'label_encoder.joblib'
label_encoder = joblib.load(label_encoder_path)


# Function to recommend a topic based on user's age and gender
def recommend_topic(user_age, user_gender):
    # Encode the gender input (assuming label_encoder was used during training)
    user_gender_encoded = label_encoder.transform([user_gender])

    # Predict the liked topic for the user by passing input as a NumPy array
    input_data = np.array([[user_age, user_gender_encoded[0]]], dtype=float)
    with warnings.catch_warnings():  # Suppress warnings temporarily
        warnings.simplefilter("ignore")
        predicted_topic = model.predict(input_data)

    # Convert the predicted label back to the original category, or use the predicted label directly
    if predicted_topic[0] in label_encoder.classes_:
        predicted_topic = label_encoder.inverse_transform(predicted_topic)
    else:
        predicted_topic = predicted_topic[0]

    return predicted_topic


# Get user input for age and gender
user_age = int(input("Enter your age: "))
user_gender = input("Enter your gender (M/F): ")

# Make a recommendation
recommended_topic = recommend_topic(user_age, user_gender)

print(f"Recommended Topic for User: {recommended_topic}")
