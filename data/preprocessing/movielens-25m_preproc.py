import numpy as np
import pandas as pd
from pmf import PMF

np.random.seed(42)

ratings = pd.read_csv(filepath_or_buffer='./data/ml-25m/raw_data/ratings.csv')
ratings = ratings.drop('timestamp', axis=1)

# Filter out users with less than 100 ratings
user_rating_counts = ratings.groupby('userId').size()
qualified_users = user_rating_counts[user_rating_counts >= 100].index
ratings = ratings[ratings['userId'].isin(qualified_users)]

# Split data into train and test sets
train_users = np.random.choice(ratings['userId'].unique(), 6000, replace=False)
train = ratings[ratings['userId'].isin(train_users)]
test = ratings[~ratings['userId'].isin(train_users)]
test = test[test['movieId'].isin(train['movieId'])]  # Filter out movies not in train set

# Reindex movieId from 0 to num_movies - 1
item_id_index = {movie_id: i for i, movie_id in enumerate(train['movieId'].unique())}
train['movieId'] = train['movieId'].map(item_id_index)
test['movieId'] = test['movieId'].map(item_id_index)

# Pivot train data for PMF model
train_matrix = train.pivot(index='userId', columns='movieId', values='rating').fillna(0).values

# Train PMF model
pmf = PMF(num_factors=10, learning_rate=0.1, num_epochs=1, patience=20, batch_size=65536)
pmf.fit(train_matrix)

print(pmf.V.shape)
# Save movie embeddings and test set
np.save('./data/ml-25m/preproc_data/item_emb.npy', pmf.V)  # ith row is the embedding for movie i
test.to_csv('./data/ml-25m/preproc_data/test.csv', index=False)

# Print the top 10 movie embeddings
print(pmf.V[:10])
