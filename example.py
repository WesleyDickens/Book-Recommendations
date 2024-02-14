# AUTOGENERATED! DO NOT EDIT! File to edit: Streamlit App.ipynb.

# %% auto 0
__all__ = []

# %% Streamlit App.ipynb 0
import torch
import torch.nn as nn
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset, DataLoader
import joblib
import pandas as pd
import streamlit as st

from streamlit_jupyter import StreamlitPatcher, tqdm

StreamlitPatcher().jupyter()  # register streamlit with jupyter-compatible wrappers

cleaned_df = pd.read_csv('cleaned_df.csv')
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# %% Streamlit App.ipynb 1
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


# %% Streamlit App.ipynb 2
class CollabFiltModel(nn.Module):
    def __init__(self, num_users, num_items, emb_size=100):
        super().__init__()
        self.user_emb = nn.Embedding(num_users, emb_size)
        self.item_emb = nn.Embedding(num_items, emb_size)
    
    def forward(self, user, item):
        user_emb = self.user_emb(user)
        item_emb = self.item_emb(item)
        return (user_emb * item_emb).sum(1)

