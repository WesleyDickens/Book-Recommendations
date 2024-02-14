import torch
import torch.nn as nn
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset, DataLoader
import joblib
import pandas as pd
import streamlit as st

cleaned_df = pd.read_csv('cleaned_df.csv')
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

import scipy.sparse as sp
import torch

def load_sparse_matrix_to_tensor(path):
    # Load the sparse matrix from disk
    sparse_matrix = sp.load_npz(path)
    
    # Convert the sparse matrix to a dense NumPy array
    dense_array = sparse_matrix.toarray()
    
    # Convert the dense NumPy array to a PyTorch tensor
    tensor = torch.tensor(dense_array, dtype=torch.float)
    
    return tensor

class CollabFiltModel(nn.Module):
    def __init__(self, num_users, num_items, emb_size=100):
        super().__init__()
        self.user_emb = nn.Embedding(num_users, emb_size)
        self.item_emb = nn.Embedding(num_items, emb_size)
    
    def forward(self, user, item):
        user_emb = self.user_emb(user)
        item_emb = self.item_emb(item)
        return (user_emb * item_emb).sum(1)
    
loaded_model = CollabFiltModel(num_users=cleaned_df['User_id'].nunique(),
                               num_items=cleaned_df['Title'].nunique()).to(device)

# Load the model state dictionary
loaded_model.load_state_dict(torch.load('collab_filt_model_state_dict.pth'))

# Ensure to switch the model to evaluation mode
loaded_model.eval()

# Load the encoders
user_encoder = joblib.load('user_encoder.joblib')
item_encoder = joblib.load('item_encoder.joblib')

loaded_model.eval()
loaded_model.to('cpu') ## Faster

# Extract item embeddings
item_embeddings = loaded_model.item_emb.weight.data.cpu().numpy()

filtered_df = cleaned_df[cleaned_df['categories'].isin(cleaned_df['categories'].value_counts()[cleaned_df['categories'].value_counts() > 20000].index)]

unique_genres = filtered_df['categories'].unique()

# Display the genres to the user
st.write("Please choose a genre from the following list:")
for i, genre in enumerate(unique_genres, 1):
    print(f"{i}. {genre}")

choice = int(st.number_input("Enter the number corresponding to your choice: ")) - 1  # Subtract 1 to match the list index

genre_choice = unique_genres[choice]


sample_titles = cleaned_df[cleaned_df['categories']==genre_choice]['Title'].sample(5).to_numpy()

decoded_titles = item_encoder.inverse_transform(sample_titles)

user_ratings = {}
st.write('Rate these books 1-5')
for title in decoded_titles:
    score = st.number_input(f"{title}: ")

    encoded_value = item_encoder.transform([title])[0]

    user_ratings[encoded_value] = float(score)

from sklearn.linear_model import Ridge
import numpy as np

# Prepare the data for ridge regression
rated_item_indices = list(user_ratings.keys())
X = item_embeddings[rated_item_indices]
y = np.array(list(user_ratings.values()))

# Fit the ridge regression model
ridge_model = Ridge(alpha=1.0)
ridge_model.fit(X, y)

# The user's "embedding" is approximated by the coefficients
user_preferences = ridge_model.coef_

predicted_ratings = np.dot(item_embeddings, user_preferences)

# Rank items by predicted rating, excluding already rated items
recommended_indices = np.argsort(-predicted_ratings)
top_recommendations = [index for index in recommended_indices if index not in rated_item_indices][:5]

# Decode the top recommended item indices to original IDs
top_recommended_item_ids = item_encoder.inverse_transform(top_recommendations)

st.write(top_recommended_item_ids)