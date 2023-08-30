import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.utils import to_categorical

# Load your dataset
df = pd.read_csv('generated_dataset.csv')

# Encode the 'LikedTopic' column into numerical labels
label_encoder = LabelEncoder()
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
y = to_categorical(df['TopicClass'])  # Use the 'TopicClass' column as labels

# Split the data into train and test sets
X_age_train, X_age_test, X_gender_train, X_gender_test, X_topics_train, X_topics_test, y_train, y_test = train_test_split(
    X_age, X_gender, X_topics, y, test_size=0.2, random_state=42)

# Define the model
model = Sequential()
model.add(Embedding(input_dim=len(tokenizer.word_index) + 1, output_dim=16, input_length=max_topics_length))
model.add(LSTM(16))
model.add(Dense(len(label_encoder.classes_), activation='softmax'))

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit({'age_input': X_age_train, 'gender_input': X_gender_train, 'topics_input': X_topics_train},
          y_train, epochs=10, batch_size=64, validation_split=0.2)

# Evaluate the model
loss, accuracy = model.evaluate({'age_input': X_age_test, 'gender_input': X_gender_test, 'topics_input': X_topics_test},
                                y_test)
print(f'Test loss: {loss:.4f}, Test accuracy: {accuracy:.4f}')

# Save the model
model.save('model_2.h5')
