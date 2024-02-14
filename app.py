import streamlit as st
import pandas as pd
import joblib
import torch
import torch.nn as nn
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset, DataLoader
from sklearn.linear_model import Ridge
import numpy as np
import scipy.sparse as sp

# Assuming all necessary imports are done and classes/functions are defined

# Load necessary data and models (adjust paths as necessary)
@st.cache
def load_data_and_models():
    cleaned_df = pd.read_csv('cleaned_df.csv')
    device = torch.device("cpu")
    loaded_model = CollabFiltModel(num_users=cleaned_df['User_id'].nunique(),
                                   num_items=cleaned_df['Title'].nunique()).to(device)
    loaded_model.load_state_dict(torch.load('collab_filt_model_state_dict.pth', map_location=device))
    loaded_model.eval()
    
    user_encoder = joblib.load('user_encoder.joblib')
    item_encoder = joblib.load('item_encoder.joblib')
    
    item_embeddings = loaded_model.item_emb.weight.data.cpu().numpy()
    
    return cleaned_df, loaded_model, user_encoder, item_encoder, item_embeddings

cleaned_df, loaded_model, user_encoder, item_encoder, item_embeddings = load_data_and_models()

# Filter genres
filtered_df = cleaned_df[cleaned_df['categories'].isin(cleaned_df['categories'].value_counts()[cleaned_df['categories'].value_counts() > 20000].index)]
unique_genres = filtered_df['categories'].unique()

# Streamlit user interface for genre selection
genre_choice = st.selectbox("Please choose a genre:", unique_genres)

# Displaying sample titles for the selected genre for user to rate
sample_titles = cleaned_df[cleaned_df['categories'] == genre_choice]['Title'].sample(5).to_numpy()
decoded_titles = item_encoder.inverse_transform(sample_titles)

user_ratings = {}
st.write('Rate these books 1-5:')
for title in decoded_titles:
    score = st.slider(f"{title}:", min_value=1, max_value=5, value=3)
    encoded_value = item_encoder.transform([title])[0]
    user_ratings[encoded_value] = float(score)

# Assuming your Ridge regression and recommendation logic is defined here

# Display recommendations to the user
st.write("We recommend the following titles based on your ratings:")
for item_id in top_recommended_item_ids:
    st.write(item_id)
