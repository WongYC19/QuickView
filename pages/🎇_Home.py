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
def get_share_prices(price_pickle_file="prices.pkl"):
    print(f"Reading share price file {price_pickle_file}...")
    share_prices = read_pickle(price_pickle_file)
    return share_prices

@st.experimental_memo
def get_signals(signal_pickle_file="MA_3_Signals.pkl"):   
    print(f"Reading signal file {signal_pickle_file}...")
    signals_df = read_pickle(signal_pickle_file)
    if len(signals_df) == 0:
        signals_df = pd.DataFrame(columns=['date', 'code', 'decision'])
    print("Signals df:", signals_df)
    return signals_df

def page():    
    st.subheader("Signal Page")    
    signals_df = get_signals()    
    buy_section, _, sell_section = st.columns([5, 2, 5])
    
    with buy_section:        
        buy_signals = signals_df.query("decision == 'buy'")
        st.dataframe(buy_signals)
        
    with sell_section:
        #  and date > '20220712'
        sell_signals = signals_df.query("decision == 'sell'")
        st.dataframe(sell_signals)
            
    tab1, tab2 = st.tabs(["Tab 1", "Tab2"])
    tab1.write("this is tab 1")
    tab2.write("this is tab 2")

    """This is a markdown"""
    st.write({"a": '👩‍🦳', "b": '👨‍🦳', "c": '👱‍♀️'})
    st.dataframe(pd.DataFrame({"A": [1,2,3], "B": [4,5,6]}))
    
if __name__ == '__main__':
    page()