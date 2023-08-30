import joblib
import os

# Load the trained model
model = joblib.load('model.joblib')

# Try to load the label encoder, or use a default one if not found
label_encoder_path = 'label_encoder.joblib'
if os.path.exists(label_encoder_path):
    label_encoder = joblib.load(label_encoder_path)
else:
    # Create a default label encoder (assuming you know the classes in advance)
    from sklearn.preprocessing import LabelEncoder

    label_encoder = LabelEncoder()
    label_encoder.classes_ = ['art', 'books', 'business', 'fashion', 'food', 'movies', 'music', 'science', 'sports',
                              'technology', 'travel', 'video games']


# Function to recommend a topic based on user's age and gender
def recommend_topic(user_age, user_gender):
    # Encode the gender input (assuming label_encoder was used during training)
    user_gender_encoded = label_encoder.transform([user_gender])

    # Predict the liked topic for the user
    predicted_topic = model.predict([[user_age, user_gender_encoded[0]]])

    # Convert the predicted label back to the original category
    predicted_topic = label_encoder.inverse_transform(predicted_topic)

    return predicted_topic[0] if predicted_topic else 'Unknown'


# Get user input for age and gender
user_age = int(input("Enter your age: "))
user_gender = input("Enter your gender (M/F): ")

# Make a recommendation
recommended_topic = recommend_topic(user_age, user_gender)

print(f"Recommended Topic for User: {recommended_topic}")
