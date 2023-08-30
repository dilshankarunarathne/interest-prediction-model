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

    # Get the feature names from the label encoder
    feature_names = label_encoder.get_feature_names(['UserAge', 'UserGender'])

    # Create a dictionary with feature names and values
    user_data = dict(zip(feature_names, [user_age, user_gender_encoded[0]]))

    # Predict the liked topic for the user
    predicted_topic = model.predict([user_data])

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
