import streamlit as st
import torch
import torch.nn as nn
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import Ridge
from torch.utils.data import Dataset, DataLoader
import joblib
import pandas as pd
import scipy.sparse as sp
import numpy as np

# Load data and models
@st.cache
def load_data():
    cleaned_df = pd.read_csv('/cleaned_df.csv')
    user_encoder = joblib.load('/user_encoder.joblib')
    item_encoder = joblib.load('/item_encoder.joblib')
    return cleaned_df, user_encoder, item_encoder

cleaned_df, user_encoder, item_encoder = load_data()

device = torch.device("cpu")

class CollabFiltModel(nn.Module):
    def __init__(self, num_users, num_items, emb_size=100):
        super().__init__()
        self.user_emb = nn.Embedding(num_users, emb_size)
        self.item_emb = nn.Embedding(num_items, emb_size)
    
    def forward(self, user, item):
        user_emb = self.user_emb(user)
        item_emb = self.item_emb(item)
        return (user_emb * item_emb).sum(1)

# Load model
def load_model():
    model = CollabFiltModel(num_users=cleaned_df['User_id'].nunique(),
                            num_items=cleaned_df['Title'].nunique()).to(device)
    model.load_state_dict(torch.load('/collab_filt_model_state_dict.pth', map_location=device))
    model.eval()
    return model

loaded_model = load_model()
item_embeddings = loaded_model.item_emb.weight.data.cpu().numpy()

# User Interface
st.title("Book Recommendation System")

filtered_df = cleaned_df[cleaned_df['categories'].isin(cleaned_df['categories'].value_counts()[cleaned_df['categories'].value_counts() > 20000].index)]
unique_genres = filtered_df['categories'].unique()
genre_options = list(unique_genres) + ["All genres"]
genre_choice = st.selectbox("Please choose a genre:", options=genre_options)

if genre_choice == "All genres":
    unique_df = cleaned_df.drop_duplicates(subset=['Title', 'authors'])
else:
    unique_df = cleaned_df[cleaned_df['categories'] == genre_choice].drop_duplicates(subset=['Title', 'authors'])

sample_data = unique_df[['Title', 'authors']].sample(10)
user_ratings = {}

for index, row in sample_data.iterrows():
    title, author = row['Title'], row['authors']
    score = st.number_input(f'Rate "{title}" by {author}:', min_value=1, max_value=5, value=3, step=1)
    user_ratings[title] = score

if st.button('Recommend Books'):
    rated_titles = list(user_ratings.keys())
    rated_item_indices = item_encoder.transform(rated_titles)
    X = item_embeddings[rated_item_indices]
    y = np.array(list(user_ratings.values()))
    
    ridge_model = Ridge(alpha=1.0)
    ridge_model.fit(X, y)
    user_preferences = ridge_model.coef_
    
    predicted_ratings = np.dot(item_embeddings, user_preferences)
    recommended_indices = np.argsort(-predicted_ratings)
    top_recommendations = [index for index in recommended_indices if index not in rated_item_indices][:5]
    
    top_recommended_item_ids = item_encoder.inverse_transform(top_recommendations)
    
    st.write("We recommend these 5 books based on your ratings: ")
    for i, book in enumerate(top_recommended_item_ids):
        st.write(f"{i+1}. {book}")
