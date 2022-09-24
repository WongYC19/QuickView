import streamlit as st
import pandas as pd

from streamlit_utils.config import config_page
from streamlit_utils.reader import get_metadata, get_share_prices_with_signals, get_scores, get_metrics
from streamlit_utils.states import set_daterange, set_signal_statistics
from streamlit_utils.components import render_icon, render_candlestick, render_dataframe

def update_code():
    full_stock_name = st.session_state["selected_stock"]
    if not "(" in full_stock_name:
        code = ""
    else:
        code = full_stock_name.split("(")[1].replace(")", "").strip()
    st.session_state['code'] = code
                        
def page():    
    config_page()    
    metadata = get_metadata()
    prices = get_share_prices_with_signals()
    scores = get_scores()    
    metrics = get_metrics()
        
    st.session_state.setdefault("code", "")
    st.session_state.setdefault("expand_chart", False)
    st.session_state.setdefault("selectedRows", [])
    
    latest_scores = scores.sort_values(by=['Year'], ascending=False).drop_duplicates(subset=['Code'], keep='first')
    f_range = st.sidebar.slider("Piotroski F-Score (greater than or equal)", value=3, min_value=0, max_value=9)
    z_range = st.sidebar.slider("Altman Z-Score (greater than or equal)", value=1.8, min_value=-5.0, max_value=5.0, step=0.5)
    o_range = st.sidebar.slider("Ohlson O-Score (less than or equal)", value=0.5, min_value=0.0, max_value =1.0, step=0.05)
    m_range = st.sidebar.slider("Beneish M-Score (less than or equal)", value=-1.78, min_value=-5.0, max_value=5.0, step=0.1)
        
    f_crit = latest_scores['F Score'] >= f_range
    z_crit = latest_scores['Z Score'] >= z_range
    o_crit = latest_scores['O Score'] <= o_range
    m_crit = latest_scores['M Score'] <= m_range
    
    metadata['Code'] = metadata['Code'].astype(str)
    metadata = metadata.merge(latest_scores, left_on='Code', right_on='Code', how='left')
    latest_scores = latest_scores.loc[f_crit & z_crit & o_crit & m_crit, :]    
        
    with st.expander("List of stocks", expanded=True):
        last_n_periods = st.number_input("Show signal for last # trading days:", key="offset_periods", min_value=1, max_value=40, value=10, step=1)
        sectors = metadata['Sector'].sort_values().unique()
        selected_sectors = st.multiselect("Select sector(s)", options=sectors, default=None)
                                
        if not selected_sectors:
            selected_sectors = sectors
            
        metadata = metadata[metadata['Sector'].isin(selected_sectors)]
        signal_meta = set_signal_statistics(metadata, prices, last_n_periods)
        st.dataframe(signal_meta, use_container_width=True)
        # AgGrid(signal_meta, key="consolidated_signals", selection_mode="single", try_to_convert_back_to_original_types=True)
            
        if not metadata.empty:
            # render_dataframe(signal_meta, key="consolidated_signals", selection_mode="single")            
            st.info(f"""Total records: {signal_meta.shape[0]}""")                                    
        else:            
            st.info("No record found.")
            
        selected_rows = st.session_state['selectedRows']
            
        if selected_rows:
            st.session_state['code'] = selected_rows[0]['Code']
            st.session_state['expand_chart'] = True
    
    with st.expander(label=f"Chart Section", expanded=st.session_state['expand_chart']):
        stocks = metadata['Name'] + " (" + metadata['Code'] + ")"
        stocks = pd.Series("--- Pick a stock ---").append(stocks)
        
        # format_func=lambda x: stocks[x]
        st.selectbox("Select a stock:", options=stocks, key="selected_stock", on_change=update_code)
        row = metadata[metadata['Code'] == st.session_state['code']]
        
        if not len(row):
            st.stop()
        
        row = row.iloc[0, :]
        name = row['Name']
        description = row['Description']
        hyperlink = row['Hyperlink']
        address = row['Address']
        code = st.session_state['code']
        
        price = prices[prices['Code'] == code]                 
        default_index = min(len(price), 60)
        initial_start_date = price.index[-default_index]
        initial_end_date = price.index[-1]                                 
        
        st.write(f"## {name} ({code})")        
        sector_col, _, industry_col, _, shariah_col  = st.columns([4, 1, 4, 1, 4], gap="small")

        with sector_col:
            render_icon("sector", caption = f"{row['Sector']}")
        with industry_col:
            render_icon("industry", caption = f"{row['Industry']}")  
            
        if row['Is Shariah']:                                
            with shariah_col:                
                render_icon("shariah", caption="Shariah Compliance")
                        
        f, z, o, m = None, None, None, None
        score = scores[scores['Code'] == code]
        
        if not score.empty:
            f = score['F Score'].iat[0]
            z = score['Z Score'].iat[0]
            o = score['O Score'].iat[0]
            m = score['M Score'].iat[0]
        
        st.markdown("---")        
        f_col, z_col, o_col, m_col, _ = st.columns([1,1,1,1,2], gap="small")
        st.markdown("---")      
        st.markdown("##### **Summary**")                                  
        st.markdown(description)
                                
        f_col.metric("F Score:", f, help="Piotroski F-score is a ranking between zero and nine that incorporates nine factors that speak to a firm's financial strength. If a company has a score of eight or nine, it is considered a good value. If a company has a score of between zero and two points, it is likely not a good value.")
        z_col.metric("Z Score:", z, help="Altman Z-score predict the likelihood of bankruptcy in next 2 years using 5 financial ratios. A Z-score that is lower than 1.8 means that the company is in financial distress and with a high probability of going bankrupt. On the other hand, a score of 3 and above means that the company is in a safe zone and is unlikely to file for bankruptcy. A score of between 1.8 and 3 means that the company is in a grey area and with a moderate chance of filing for bankruptcy")
        o_col.metric("O Score:", o, help="Ohlson O-score predict the financial distress of a company using 8 factors linear combination of coefficient-weighted accounting ratios. Any results larger than 0.5 suggest that the firm will default within two years")
        m_col.metric("M Score:", m, help="Beneish’s M-Score measures the likelihood of a company has manipulated its profits using eight financial ratios weighted by coefficient. Less than -2.22 means Unlikely profit manipulation. A company with greater than 1.78 is likely profit manipulation. The middle range implies possible only.")
                
        hyperlink_col, address_col, _ = st.columns([1, 2, 1], gap="small")        
        hyperlink_col.markdown("***Link***: " + hyperlink)
        address_col.markdown("***Address***: " + address)
        
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
        
        metric = metrics[metrics['Code'] == code].drop('Code', axis=1)
        st.dataframe(metric.style.format("{:.2f}"), use_container_width =True)
    
if __name__ == '__main__':
    page()
    