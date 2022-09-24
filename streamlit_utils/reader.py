import os
import pickle

import streamlit as st
import pandas as pd

def read_pickle(file_path: str) -> dict:            
    if not os.path.isabs(file_path):
        file_path = os.path.join(os.getcwd(), file_path)
    
    if os.path.exists(file_path) and file_path.endswith(".pkl"):    
        with open(file_path, "rb") as file:
            data = pickle.load(file)
            return data
        
    print(f"{file_path} is not a pickle file or doesn't exist.")
    return {}

@st.experimental_memo
def get_scores(pkl_file: str = "theia/storage/scores.pkl"):
    scores = pd.read_pickle(pkl_file)
    scores = scores.rename_axis('year').reset_index()
    scores.columns = scores.columns.str.replace("_", " ").str.title()
    scores['Code'] = scores['Code'].astype(str)
    scores = scores.round(2)
    return scores

@st.experimental_memo
def get_metrics(pkl_file: str = "theia/storage/metrics.pkl"):
    metrics = pd.read_pickle(pkl_file)
    metrics = metrics.sort_index(ascending=False).round(2)
    metrics.index = metrics.index.date
    metrics.columns = metrics.columns.str.replace("_", " ").str.title()
    return metrics

@st.experimental_memo
def get_metadata():    
    metadata = pd.read_csv("theia/storage/metadata.csv") 
    field_mappers = {
        'bursacode': 'Code', 
        'alias': 'Alias', 
        'name': 'Name', 
        'shariah': 'Is Shariah', 
        'economicsectorcode': 'Sector', 
        'industrygroupcode': 'Industry', 
        'description': 'Description',
        'hyperlink': 'Hyperlink',
        'address': 'Address',
    }
    
    metadata.economicsectorcode.fillna("Other", inplace=True)
    metadata.industrygroupcode.fillna("Other", inplace=True)
    
    metadata['shariah'] = metadata['shariah'].astype(bool)    
    is_four_digit = metadata.bursacode.str.isdigit() & (metadata.bursacode.str.len() == 4)
    is_klcc = metadata.bursacode.str.endswith("SS")
    
    metadata.rename(columns=field_mappers, inplace=True)
    metadata = metadata[field_mappers.values()]
    metadata = metadata[is_four_digit | is_klcc]
    metadata.reset_index(drop=True, inplace=True)
    metadata['Code'] = metadata['Code'].astype(str)
    return metadata

@st.experimental_memo
def get_share_prices_with_signals(signal_pickle_file="theia/storage/MA_3_Signals.pkl"):    
    print(f"Reading share price file {signal_pickle_file}...")
    prices = pd.read_pickle(signal_pickle_file)
    prices.columns = prices.columns.astype(str).str.title()
    prices['Code'] = prices['Code'].astype('category')
    prices['Date'] = prices['Date'].dt.date
    prices.set_index('Date', inplace=True)
    return prices