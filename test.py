import joblib

# Load the trained model
model = joblib.load('model.joblib')

# Load the label encoder used during training
label_encoder = joblib.load('label_encoder.joblib')  # If you saved it during training


# Function to recommend a topic based on user's age and gender
def recommend_topic(user_age, user_gender):
    # Reshape user_gender to (1, 1)
    user_gender = user_gender.reshape(1, 1)

    # Predict the liked topic for the user
    predicted_topic = model.predict([[user_age, user_gender]])

    # Convert the predicted label back to the original category
    predicted_topic = label_encoder.inverse_transform(predicted_topic)

    return predicted_topic[0] if predicted_topic else 'Unknown'


# Get user input for age and gender
user_age = int(input("Enter your age: "))
user_gender = input("Enter your gender (M/F): ")

# Encode the gender input (assuming label_encoder was used during training)
user_gender = label_encoder.transform([user_gender])

# Make a recommendation
recommended_topic = recommend_topic(user_age, user_gender)

print(f"Recommended Topic for User: {recommended_topic}")
