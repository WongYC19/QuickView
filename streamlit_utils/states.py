from signal import signal
from typing import Tuple
from datetime import datetime

import pandas as pd
import streamlit as st
from pandas.tseries.offsets import BDay

def set_daterange(initial_end_date: Tuple):
    date_range = st.session_state["date_range"]
    try:
        start_date, end_date = date_range                
    except ValueError:                
        start_date = date_range[0]
        end_date = initial_end_date
        
    if start_date == end_date:
        end_date = initial_end_date
        
    st.session_state["date_range"] = (start_date, end_date)                

def set_selected_stock_index(metadata):   
    selected_stock = st.session_state['selected_stock']
    name, code = selected_stock.split("(")
    name, code = name.strip(), code.replace(")", "").strip()
    index =  metadata[metadata['Code'] == code].index
    if index.shape[0] > 1:
        index = index[0]
    
    st.session_state['selected_index'] = index
    
@st.experimental_memo
def set_signal_statistics(metadata, signals_df, offset_periods):
    recent_date = datetime.today() - BDay(offset_periods)    
    signals_df = signals_df[signals_df.index >= recent_date]
    signal_stats = signals_df.groupby('Code').agg({'Buy': sum, 'Sell': sum})
    signal_stats_with_meta = metadata.merge(signal_stats, how='left', on='Code')
    signal_stats_with_meta = signal_stats_with_meta.sort_values(by=['Buy', 'Sell'], ascending=[False, True])
    score_cols = signal_stats_with_meta.columns[signal_stats_with_meta.columns.str.endswith("Score")].tolist()
    cols = ["Code", "Alias", "Name", "Buy", "Sell", "Sector", "Industry", "Is Shariah"] + score_cols
    signal_stats_with_meta = signal_stats_with_meta[cols]
    return signal_stats_with_meta

@st.experimental_memo
def set_signal(price_signal):    
    return price_signal[price_signal[price_signal.Buy | price_signal.Sell]].index
