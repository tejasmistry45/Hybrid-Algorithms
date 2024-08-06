
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.linear_model import LinearRegression

# Load data
data = pd.read_csv('songsDataset.csv', nrows=1000)

# Preprocess data
data.columns = data.columns.str.strip().str.replace("'", "")

data.head()

data.info()

data.isna().sum()

data.describe()

# Create user-item matrix
user_item_matrix = data.pivot_table(index='userID', columns='songID', values='rating').fillna(0)
user_item_matrix.head()

# Compute user similarities
user_similarity_matrix = cosine_similarity(user_item_matrix)
user_similarity_df = pd.DataFrame(user_similarity_matrix, index=user_item_matrix.index, columns=user_item_matrix.index)

# Compute item similarities
item_similarity_matrix = cosine_similarity(user_item_matrix.T)
item_similarity_df = pd.DataFrame(item_similarity_matrix, index=user_item_matrix.columns, columns=user_item_matrix.columns)

# Function to get similar users
def get_user_similarity(target_user, user_similarity_df, top_n=5):
    similar_scores = user_similarity_df[target_user].sort_values(ascending=False)
    similar_users = similar_scores.iloc[1:top_n+1].index.tolist()
    return similar_users

# Function to get similar items
def get_similar_items(songID, item_similarity_df, top_n=5):
    similar_scores = item_similarity_df[songID].sort_values(ascending=False)
    similar_items = similar_scores.iloc[1:top_n+1].index
    return similar_items

# Function to recommend songs using user-based CF
def recommend_songs_ubcf(userID, user_item_matrix, user_similarity_df, num_recs=5):
    similar_users = get_user_similarity(userID, user_similarity_df)
    recommended_songs = []
    for similar_userID in similar_users:
        for songID, rating in user_item_matrix.loc[similar_userID].items():
            if rating > 0:
                recommended_songs.append((songID, rating))
    recommended_songs = sorted(recommended_songs, key=lambda x: x[1], reverse=True)
    return [songID for songID, _ in recommended_songs[:num_recs]]

# Function to recommend songs using item-based CF
def recommend_songs_ibcf(userID, user_item_matrix, item_similarity_df, num_recs=5):
    user_ratings = user_item_matrix.loc[userID]
    recommended_songs = pd.Series(dtype=float)
    for songID, rating in user_ratings.items():
        if rating > 0:
            similar_items = get_similar_items(songID, item_similarity_df)
            for similar_item in similar_items:
                if similar_item in recommended_songs:
                    recommended_songs[similar_item] += rating
                else:
                    recommended_songs[similar_item] = rating
    recommended_songs = recommended_songs.sort_values(ascending=False)
    return recommended_songs.head(num_recs).index

# Function to create the meta-model training data
def create_meta_model_data(user_item_matrix, user_similarity_df, item_similarity_df, top_n=5):
    meta_data = []
    for userID in user_item_matrix.index:
        ubcf_recs = recommend_songs_ubcf(userID, user_item_matrix, user_similarity_df, num_recs=top_n*2)
        ibcf_recs = recommend_songs_ibcf(userID, user_item_matrix, item_similarity_df, num_recs=top_n*2)
        for songID in user_item_matrix.columns:
            ubcf_score = 1 if songID in ubcf_recs else 0
            ibcf_score = 1 if songID in ibcf_recs else 0
            actual_rating = user_item_matrix.at[userID, songID]
            meta_data.append([ubcf_score, ibcf_score, actual_rating])
    return pd.DataFrame(meta_data, columns=['ubcf_score', 'ibcf_score', 'rating'])

# Create the meta-model training data
meta_model_data = create_meta_model_data(user_item_matrix, user_similarity_df, item_similarity_df)

# Train the meta-model
X = meta_model_data[['ubcf_score', 'ibcf_score']]
y = meta_model_data['rating']

meta_model = LinearRegression()

meta_model.fit(X, y)

meta_model.score(X,y)

# Hybrid recommendation function using stacking
def hybrid_recommendations(target_user, user_item_matrix, user_similarity_df, item_similarity_df, meta_model, top_n=5):
    ubcf_recs = recommend_songs_ubcf(target_user, user_item_matrix, user_similarity_df, num_recs=top_n*2)
    ibcf_recs = recommend_songs_ibcf(target_user, user_item_matrix, item_similarity_df, num_recs=top_n*2)

    recommendations = pd.Series(dtype=float)

    for songID in user_item_matrix.columns:
        ubcf_score = 1 if songID in ubcf_recs else 0
        ibcf_score = 1 if songID in ibcf_recs else 0
        hybrid_score = meta_model.predict(pd.DataFrame([[ubcf_score, ibcf_score]], columns=['ubcf_score', 'ibcf_score']))[0]
        recommendations[songID] = hybrid_score

    recommendations = recommendations.sort_values(ascending=False)
    return recommendations.head(top_n).index

# Test the hybrid model
target_user = int(input("Enter user ID: "))
recommended_songs = hybrid_recommendations(target_user, user_item_matrix, user_similarity_df, item_similarity_df, meta_model)
print(f"Recommendations for user {target_user}: {recommended_songs}")

# Recommendations for user 5: Index([95898, 24427, 99702, 98571, 33558], dtype='int64')





"""# Test 2"""

import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.linear_model import LinearRegression

# Load data
data = pd.read_csv('songsDataset.csv', nrows=1000)

# Preprocess data
data.columns = data.columns.str.strip().str.replace("'", "")
data.head()
data.info()
data.isna().sum()
data.describe()

# Create user-item matrix
user_item_matrix = data.pivot_table(index='userID', columns='songID', values='rating').fillna(0)
user_item_matrix.head()

# Compute user similarities
user_similarity_matrix = cosine_similarity(user_item_matrix)
user_similarity_df = pd.DataFrame(user_similarity_matrix, index=user_item_matrix.index, columns=user_item_matrix.index)

# Compute item similarities
item_similarity_matrix = cosine_similarity(user_item_matrix.T)
item_similarity_df = pd.DataFrame(item_similarity_matrix, index=user_item_matrix.columns, columns=user_item_matrix.columns)

# Function to get similar users
def get_user_similarity(target_user, user_similarity_df, top_n=5):
    similar_scores = user_similarity_df[target_user].sort_values(ascending=False)
    similar_users = similar_scores.iloc[1:top_n+1].index.tolist()
    return similar_users

# Function to get similar items
def get_similar_items(songID, item_similarity_df, top_n=5):
    similar_scores = item_similarity_df[songID].sort_values(ascending=False)
    similar_items = similar_scores.iloc[1:top_n+1].index
    return similar_items

# Function to recommend songs using user-based CF
def recommend_songs_ubcf(userID, user_item_matrix, user_similarity_df, num_recs=5):
    similar_users = get_user_similarity(userID, user_similarity_df)
    recommended_songs = []
    for similar_userID in similar_users:
        for songID, rating in user_item_matrix.loc[similar_userID].items():
            if rating > 0:
                recommended_songs.append((songID, rating))
    recommended_songs = sorted(recommended_songs, key=lambda x: x[1], reverse=True)
    return [songID for songID, _ in recommended_songs[:num_recs]]

# Function to recommend songs using item-based CF
def recommend_songs_ibcf(userID, user_item_matrix, item_similarity_df, num_recs=5):
    user_ratings = user_item_matrix.loc[userID]
    recommended_songs = pd.Series(dtype=float)
    for songID, rating in user_ratings.items():
        if rating > 0:
            similar_items = get_similar_items(songID, item_similarity_df)
            for similar_item in similar_items:
                if similar_item in recommended_songs:
                    recommended_songs[similar_item] += rating
                else:
                    recommended_songs[similar_item] = rating
    recommended_songs = recommended_songs.sort_values(ascending=False)
    return recommended_songs.head(num_recs).index

def create_meta_model_data(user_item_matrix, user_similarity_df, item_similarity_df, top_n=5):
    meta_data = []
    for userID in user_item_matrix.index:
        for songID in user_item_matrix.columns:
            ubcf_score = 1  # default score
            ibcf_score = 1  # default score
            actual_rating = user_item_matrix.at[userID, songID]
            meta_data.append([ubcf_score, ibcf_score, actual_rating])
    print("meta_data shape:", len(meta_data))
    meta_model_data = pd.DataFrame(meta_data, columns=['ubcf_score', 'ibcf_score', 'rating'])
    print("meta_model_data shape:", meta_model_data.shape)
    X = meta_model_data[['ubcf_score', 'ibcf_score']]
    y = meta_model_data['rating']
    print("X shape:", X.shape)
    print("y shape:", y.shape)
    return X, y

X, y = create_meta_model_data(user_item_matrix, user_similarity_df, item_similarity_df)

# Create the meta-model training data
# meta_model_data = create_meta_model_data(user_item_matrix, user_similarity_df, item_similarity_df)

meta_model = LinearRegression()

meta_model.fit(X, y)

meta_model.score(X,y)

# Hybrid recommendation function using stacking
def hybrid_recommendations(target_user, user_item_matrix, user_similarity_df, item_similarity_df, meta_model, top_n=5):
    ubcf_recs = recommend_songs_ubcf(target_user, user_item_matrix, user_similarity_df, num_recs=top_n*2)
    ibcf_recs = recommend_songs_ibcf(target_user, user_item_matrix, item_similarity_df, num_recs=top_n*2)

    recommendations = pd.Series(dtype=float)

    for songID in user_item_matrix.columns:
        ubcf_score = 1 if songID in ubcf_recs else 0
        ibcf_score = 1 if songID in ibcf_recs else 0
        hybrid_score = meta_model.predict(pd.DataFrame([[ubcf_score, ibcf_score]], columns=['ubcf_score', 'ibcf_score']))[0]
        recommendations[songID] = hybrid_score

    recommendations = recommendations.sort_values(ascending=False)
    return recommendations.head(top_n).index

# Test the hybrid model
target_user = int(input("Enter user ID: "))
recommended_songs = hybrid_recommendations(target_user, user_item_matrix, user_similarity_df, item_similarity_df, meta_model)
print(f"Recommendations for user {target_user}: {recommended_songs}")

# Recommendations for user 5: Index([319, 96037, 92459, 92523, 92547], dtype='int64')









