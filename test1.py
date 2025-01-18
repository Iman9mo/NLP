import pandas as pd
from gensim.models import Word2Vec
from sklearn.neighbors import NearestNeighbors
from collections import Counter
import numpy as np
from bs4 import BeautifulSoup
import re

# Preprocessing function
def preprocess_text_simple(text):
    # Remove HTML tags
    text = BeautifulSoup(text, "lxml").get_text()
    # Lowercase the text
    text = text.lower()
    # Remove special characters and numbers
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Tokenize and join back as a string
    return ' '.join(text.split())

# Load datasets
train_data = pd.read_csv('processed_train.csv')
valid_data = pd.read_csv('processed_valid.csv')

# Ensure all entries are strings
train_data['Processed_Body'] = train_data['Processed_Body'].fillna("").astype(str)
valid_data['Processed_Body'] = valid_data['Processed_Body'].fillna("").astype(str)

# Preprocess validation data
valid_data['Processed_Body'] = valid_data['Body'].apply(preprocess_text_simple)

# Train Word2Vec model
sentences = train_data['Processed_Body'].apply(lambda x: x.split()).tolist()
word2vec_model = Word2Vec(sentences, vector_size=100, window=5, min_count=2, workers=4)

# Compute document vectors
def document_vector(doc):
    words = doc.split()
    vectors = [word2vec_model.wv[word] for word in words if word in word2vec_model.wv]
    return np.mean(vectors, axis=0) if vectors else np.zeros(word2vec_model.vector_size)

train_data['Doc_Vector'] = train_data['Processed_Body'].apply(document_vector)
valid_data['Doc_Vector'] = valid_data['Processed_Body'].apply(document_vector)

# Train KNN model
train_vectors = np.array(train_data['Doc_Vector'].tolist())
train_tags = train_data['Tags'].tolist()
k = 5
knn_model = NearestNeighbors(n_neighbors=k, metric='cosine')
knn_model.fit(train_vectors)

# Predict tags with weighted similarity
def find_nearest_neighbors(doc_vector):
    distances, indices = knn_model.kneighbors([doc_vector])
    return indices[0], 1 - distances[0]

def predict_tags_weighted(neighbors, similarities):
    neighbor_tags = [train_tags[i] for i in neighbors]
    flat_tags = []
    for tags, similarity in zip(neighbor_tags, similarities):
        for tag in tags.split():
            flat_tags.append((tag, similarity))
    tag_counter = Counter()
    for tag, weight in flat_tags:
        tag_counter[tag] += weight
    return [tag for tag, _ in tag_counter.most_common(3)]

valid_data['Predicted_Tags'] = valid_data.apply(
    lambda row: predict_tags_weighted(*find_nearest_neighbors(row['Doc_Vector'])),
    axis=1
)

# Evaluate accuracy
def calculate_accuracy(row):
    actual_tags = set(row['Tags'].split())
    predicted_tags = set(row['Predicted_Tags'])
    return len(actual_tags & predicted_tags) > 0

valid_data['Correct_Tag'] = valid_data.apply(calculate_accuracy, axis=1)
accuracy = valid_data['Correct_Tag'].mean()
print(f"Tagging Accuracy: {accuracy * 100:.2f}%")

# Show examples
successful_examples = valid_data[valid_data['Correct_Tag']].head(5)
unsuccessful_examples = valid_data[~valid_data['Correct_Tag']].head(5)
print("\nSuccessful Predictions:")
print(successful_examples[['Body', 'Tags', 'Predicted_Tags']])
print("\nUnsuccessful Predictions:")
print(unsuccessful_examples[['Body', 'Tags', 'Predicted_Tags']])
