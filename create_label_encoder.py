from sklearn.preprocessing import LabelEncoder
import joblib

lines = []

with open("unique_topics.txt", "r") as topics:
	lines = topics.readlines()

label_encoder = LabelEncoder()
label_encoder.classes_ = lines

# save the label encoder
joblib.dump(label_encoder, "label_encoder.joblib")

# TODO: This code is not working. Need to fix it.
