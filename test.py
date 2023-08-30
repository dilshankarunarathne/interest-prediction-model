import numpy as np

# Define a list of 50 topics
topics = ['Topic' + str(i) for i in range(1, 51)]

# Generate random scores for each topic (between 0 and 1)
topic_scores = {topic: np.random.rand(5) for topic in topics}

# Print the generated topic scores
for topic, scores in topic_scores.items():
    print(f'{topic}: {scores}')