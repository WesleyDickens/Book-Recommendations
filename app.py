import streamlit as st
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import joblib

# Define your collaborative filtering model
class CollabFiltModel(nn.Module):
    def __init__(self, num_users, num_items, emb_size=100):
        super().__init__()
        self.user_emb = nn.Embedding(num_users, emb_size)
        self.item_emb = nn.Embedding(num_items, emb_size)

    def forward(self, user, item):
        user_emb = self.user_emb(user)
        item_emb = self.item_emb(item)
        return (user_emb * item_emb).sum(1)

# Load data and model
@st.cache(allow_output_mutation=True)
def load_data_and_model():
    # Load the dataset
    cleaned_df = pd.read_csv('cleaned_df.csv')
    device = torch.device('cpu')  # Change to 'cuda' if GPU is available

    # Load label encoders
    user_encoder = joblib.load('user_encoder.joblib')
    item_encoder = joblib.load('item_encoder.joblib')

    # Prepare model
    num_users = len(user_encoder.classes_)
    num_items = len(item_encoder.classes_)
    model = CollabFiltModel(num_users, num_items).to(device)
    model.load_state_dict(torch.load('collab_filt_model_state_dict.pth', map_location=device))
    model.eval()

    return cleaned_df, model, user_encoder, item_encoder

# Function to make recommendations
def make_recommendations(model, genre, cleaned_df, user_encoder, item_encoder, num_recommendations=5):
    # Filter books by genre if not 'All genres'
    if genre != 'All genres':
        genre_books = cleaned_df[cleaned_df['categories'] == genre]
    else:
        genre_books = cleaned_df

    # Here, you should implement your logic to select items based on the model's predictions
    # For simplicity, this example randomly selects books
    recommendations = genre_books.sample(n=num_recommendations)

    return recommendations[['Title', 'categories']]

# Streamlit application
def main():
    st.title("Book Recommendation System")

    cleaned_df, model, user_encoder, item_encoder = load_data_and_model()

    genres = ['All genres'] + sorted(cleaned_df['categories'].unique().tolist())
    selected_genre = st.selectbox("Select a genre for recommendations:", genres)

    if st.button("Recommend Books"):
        recommendations = make_recommendations(model, selected_genre, cleaned_df, user_encoder, item_encoder)
        st.write("Here are your recommendations:")
        st.dataframe(recommendations)

if __name__ == "__main__":
    main()
    
    
st.write("Here are your recommendations:")
