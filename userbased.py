
import pandas as pd
import numpy as np

data = pd.read_csv("songsDataset.csv",nrows=10000)
data.columns = data.columns.str.strip().str.replace("'", "")

data.head()

data.info()

data.isna().sum()

data.describe()

"""user-item matric"""

user_item_matrix = data.pivot_table(index='userID', columns='songID', values='rating').fillna(0)
user_item_matrix.head()

from sklearn.metrics.pairwise import cosine_similarity
user_similarity_matrix = cosine_similarity(user_item_matrix)
# user - user
user_similarity_df = pd.DataFrame(user_similarity_matrix, index=user_item_matrix.index, columns=user_item_matrix.index)
user_similarity_df

"""similar user"""

def get_user_similarity(target_user, user_similarity_df, top_n=5):
    similar_scores = user_similarity_df[target_user].sort_values(ascending=False)
    similar_users = similar_scores.iloc[1:top_n+1].index.tolist()
    return similar_users

"""Recommendation songs to target user"""

def recommend_songs(userID, user_item_matrix, user_similarity_df, num_recs=5):
    similar_users = get_user_similarity(userID, user_similarity_df)

    recommended_songs = []

    for similar_userID in similar_users:
        for songID, rating in user_item_matrix.loc[similar_userID].items():
            if rating > 0:
                recommended_songs.append((songID, rating))

    recommended_songs = sorted(recommended_songs, key=lambda x: x[1], reverse=True)

    return [songID for songID, _ in recommended_songs[:num_recs]]

target_user = int(input("Enter user ID: "))
recommended_songs = recommend_songs(target_user, user_item_matrix, user_similarity_df)
print(f"Recommendations for user {target_user}: {recommended_songs}")









