import numpy as np
import pandas as pd
from pmf import PMF

np.random.seed(42)

ratings = pd.read_csv(filepath_or_buffer='../ml-25m/raw_data/ratings.csv')
ratings = ratings.drop('timestamp', axis=1)

# Filter out users with less than 100 ratings
user_rating_counts = ratings.groupby('userId').size()
qualified_movies = user_rating_counts[user_rating_counts >= 100].index
ratings = ratings[ratings['userId'].isin(qualified_movies)]

# Split data into train and test sets
train_user = np.random.choice(ratings['userId'].unique(), 6000, replace=False)
train = ratings[ratings['userId'].isin(train_user)]
test = ratings[~ratings['userId'].isin(train_user)]
test = test[test['movieId'].isin(train['movieId'])] # Filter out movies not in train set

train = train.pivot(index='userId', columns='movieId', values='rating').fillna(0).values

# Train PMF model
pmf = PMF(num_factors=10, learning_rate=0.001, num_epochs=100, patience=20, batch_size=65536)
pmf.fit(train)

# save movie embedding and test set
np.save('../ml-25m/preproc_data/item_emb.npy', pmf.V) # ith row is the embedding for movie i
test.to_csv('../ml-25m/preproc_data/test.csv', index=False)

# print the top 10 movies embedding
print(pmf.V[:10])