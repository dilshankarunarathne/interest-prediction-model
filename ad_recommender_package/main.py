import warnings
import joblib
import numpy as np
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
model_file_path = os.path.join(current_dir, '..', 'ad_recommender_model', 'model.joblib')
label_encoder_file_path = os.path.join(current_dir, '..', 'ad_recommender_model', 'label_encoder.joblib')
model = joblib.load(model_file_path)
label_encoder = joblib.load(label_encoder_file_path)


def recommend_topic(age, gender):
    """
    This method 
    :param age: age of the user
    :param gender: gender of the user
    :return: recommended topic
    """
    user_gender_encoded = label_encoder.transform([gender])

    input_data = np.array([[age, user_gender_encoded[0]]], dtype=float)
    with warnings.catch_warnings():  # Suppress warnings temporarily
        warnings.simplefilter("ignore")
        predicted_topic = model.predict(input_data)

    if predicted_topic[0] in label_encoder.classes_:
        predicted_topic = label_encoder.inverse_transform(predicted_topic)
    else:
        predicted_topic = predicted_topic[0]

    return predicted_topic
