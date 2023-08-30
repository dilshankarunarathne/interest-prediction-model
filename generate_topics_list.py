import pandas as pd

# Read the CSV file
df = pd.read_csv('generated_dataset.csv')

# Extract unique topics from the 'LikedTopic' column
unique_topics = df['LikedTopic'].unique()

# Write unique topics to a text file
with open('unique_topics.txt', 'w') as file:
    for topic in unique_topics:
        file.write(topic + '\n')

print("Unique topics have been written to 'unique_topics.txt'.")
