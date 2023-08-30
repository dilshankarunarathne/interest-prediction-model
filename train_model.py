import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.layers import Input, Embedding, Concatenate, Flatten, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical

# Load your dataset
df = pd.read_csv('generated_dataset.csv')

# Create a label encoder
label_encoder = LabelEncoder()

# Encode the 'LikedTopic' column into numerical labels
df['TopicClass'] = label_encoder.fit_transform(df['LikedTopic'])

# Tokenize and pad liked topics
max_topics_length = 50
tokenizer = Tokenizer()
tokenizer.fit_on_texts(df['LikedTopic'])
topics_seq = tokenizer.texts_to_sequences(df['LikedTopic'])
topics_pad = pad_sequences(topics_seq, maxlen=max_topics_length)

# Split the data
X_age = df['UserAge'].values
X_gender = df['UserGender'].values
X_topics = topics_pad
y = to_categorical(df['LikedTopic'])

# Split the data into train and test sets
X_age_train, X_age_test, X_gender_train, X_gender_test, X_topics_train, X_topics_test, y_train, y_test = train_test_split(
    X_age, X_gender, X_topics, y, test_size=0.2, random_state=42)

# Define the model
input_age = Input(shape=(1,), name='age_input')
input_gender = Input(shape=(1,), name='gender_input')
input_topics = Input(shape=(max_topics_length,), name='topics_input')

embedding_age = Embedding(input_dim=100, output_dim=16)(input_age)  # Adjust input_dim as needed
embedding_gender = Embedding(input_dim=2, output_dim=16)(input_gender)
embedding_topics = Embedding(input_dim=len(tokenizer.word_index) + 1, output_dim=16)(input_topics)

lstm_age = Flatten()(embedding_age)
lstm_gender = Flatten()(embedding_gender)
lstm_topics = Flatten()(embedding_topics)

concatenated = Concatenate()([lstm_age, lstm_gender, lstm_topics])
output = Dense(len(tokenizer.word_index) + 1, activation='softmax')(concatenated)

model = Model(inputs=[input_age, input_gender, input_topics], outputs=output)

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(
    {'age_input': X_age_train, 'gender_input': X_gender_train, 'topics_input': X_topics_train},
    y_train,
    epochs=10,
    batch_size=64,
    validation_split=0.2
)

# Evaluate the model
loss, accuracy = model.evaluate(
    {'age_input': X_age_test, 'gender_input': X_gender_test, 'topics_input': X_topics_test},
    y_test
)
print(f'Test loss: {loss:.4f}, Test accuracy: {accuracy:.4f}')

# Save the model
model.save('topic_recommendation_model.h5')
