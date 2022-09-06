from turtle import onclick
import streamlit as st

from streamlit_utils.config import config_page
from streamlit_utils.reader import get_metadata, get_share_prices_with_signals
from streamlit_utils.states import set_daterange, set_signal_statistics
from streamlit_utils.components import render_icon, render_candlestick, render_dataframe
            
def page():    
    config_page()
    metadata = get_metadata()
    prices = get_share_prices_with_signals()
    
    with st.expander("List of stocks", expanded=True):        
        last_n_periods = st.number_input("Show signal for last # trading days:", key="offset_periods", min_value=1, max_value=40, value=10, step=1)
        sectors = metadata['Sector'].sort_values().unique()
        selected_sectors = st.multiselect("Select sector(s)", options=sectors, default=None)
                                
        if not selected_sectors:
            selected_sectors = sectors
            
        narrowed_metadata = metadata[metadata['Sector'].isin(selected_sectors)]
        signal_meta = set_signal_statistics(narrowed_metadata, prices, last_n_periods)
        
        if not metadata.empty:
            signal_grid = render_dataframe(signal_meta, key="consolidated_signals")
            st.info(f"""Matches record: {signal_meta.shape[0]}""")
        else:
            st.info("No record found.")
                                                                                                
    with st.expander(label=f"Chart Section", expanded=True):                           
        stocks = metadata['Name'] + " (" + metadata['Code'] + ")"
        stock_index = st.selectbox("Select a stock:", stocks.index, key="selected_stock", format_func=lambda x: stocks[x])
        
        row = metadata.iloc[stock_index, :]
        name, code = row['Name'], row['Code']
        price = prices[prices['Code'] == code]                      
        
        default_index = min(len(price), 60)        
        initial_start_date = price.index[-default_index]
        initial_end_date = price.index[-1]                                                    
        
        st.write(f"## {name} ({code})")
        sector_col, _, industry_col, _, shariah_col  = st.columns([4, 1, 4, 1, 4], gap="small")
        
        with sector_col:
            render_icon("sector", width=50, height=50, caption = f"{row['Sector']}")
        with industry_col:
            render_icon("industry", width=50, height=50, caption = f"{row['Industry']}")  
            
        if row['Is Shariah']:                                
            with shariah_col:                
                render_icon("shariah", width=50, height=50, caption="Shariah Compliance")
                    
        st.date_input(
                label="Pick a range (default to last 60 days)", 
                value=(initial_start_date, initial_end_date), 
                key="date_range",
                on_change=set_daterange,
                args=(initial_end_date,),
        )
        start_date, end_date = st.session_state['date_range']
        share_price = price.loc[start_date:end_date]
                    
        render_candlestick(share_price, name)
                            
if __name__ == '__main__':
    page()
    