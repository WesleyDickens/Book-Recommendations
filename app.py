import streamlit as st
import pandas as pd
import joblib
import torch
import torch.nn as nn
import numpy as np
import scipy.sparse as sp
from sklearn.linear_model import Ridge

# Set device for model
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# Load sparse matrix to tensor function
def load_sparse_matrix_to_tensor(path):
    sparse_matrix = sp.load_npz(path)
    dense_array = sparse_matrix.toarray()
    tensor = torch.tensor(dense_array, dtype=torch.float)
    return tensor

# Collaborative Filtering Model Class
class CollabFiltModel(nn.Module):
    def __init__(self, num_users, num_items, emb_size=100):
        super(CollabFiltModel, self).__init__()
        self.user_emb = nn.Embedding(num_users, emb_size)
        self.item_emb = nn.Embedding(num_items, emb_size)
    
    def forward(self, user, item):
        user_emb = self.user_emb(user)
        item_emb = self.item_emb(item)
        return (user_emb * item_emb).sum(1)

# Load cleaned data
@st.cache
def load_data():
    cleaned_df = pd.read_csv('cleaned_df.csv')
    return cleaned_df

cleaned_df = load_data()

# Initialize the model and load state
@st.cache(allow_output_mutation=True)
def load_model():
    model = CollabFiltModel(num_users=cleaned_df['User_id'].nunique(), num_items=cleaned_df['Title'].nunique())
    model.load_state_dict(torch.load('collab_filt_model_state_dict.pth', map_location=torch.device('cpu')))
    model.eval()  # Set model to evaluation mode
    return model

model = load_model()

# Load encoders
user_encoder = joblib.load('user_encoder.joblib')
item_encoder = joblib.load('item_encoder.joblib')

# Filtering logic for genres
filtered_df = cleaned_df[cleaned_df['categories'].isin(cleaned_df['categories'].value_counts()[cleaned_df['categories'].value_counts() > 20000].index)]
unique_genres = filtered_df['categories'].unique()

# Streamlit UI for genre selection
genre_choice = st.selectbox("Please choose a genre:", unique_genres)

# Displaying sample titles for user ratings
sample_titles = cleaned_df[cleaned_df['categories'] == genre_choice]['Title'].sample(5).values
decoded_titles = item_encoder.inverse_transform(sample_titles)

user_ratings = {}
st.write('Rate these books (1-5):')
for title in decoded_titles:
    score = st.slider(f"{title}:", 1, 5, 3)
    encoded_value = item_encoder.transform([title])[0]
    user_ratings[encoded_value] = float(score)

# Ridge regression to predict user preferences
item_embeddings = model.item_emb.weight.data.numpy()
rated_item_indices = list(user_ratings.keys())
X = item_embeddings[rated_item_indices]
y = np.array(list(user_ratings.values()))

ridge_model = Ridge(alpha=1.0)
ridge_model.fit(X, y)
user_preferences = ridge_model.coef_

# Predicting ratings for all items and recommending top 5
predicted_ratings = np.dot(item_embeddings, user_preferences)
recommended_indices = np.argsort(-predicted_ratings)[:5]
top_recommended_item_ids = item_encoder.inverse_transform(recommended_indices)

# Displaying recommended books
st.write("Recommended Books for You:")
for book_id in top_recommended_item_ids:
    st.write(book_id)
