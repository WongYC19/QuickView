import streamlit as st
from pages import main, stock, sample

state = st.session_state
state.setdefault("is_authenticated", False)
state.setdefault("auth_name", "")

global_state = {}

pages = {
    "Main Page 🥑": main.page,
    "Stock Page 💧": stock.page,
    "Sample Page": sample.page,
}       

def login():
    username = state.auth_name
    print("username:", username)
    is_valid_username = username in ['ycwong'] 
    state.is_authenticated = is_valid_username       
    state.auth_name = username 
    global_state['auth_name'] = username
    global_state['is_authenticated'] = is_valid_username
    return state.is_authenticated

def logout():
    state.is_authenticated = False    
    state.auth_name = ""

if not state.is_authenticated:            
    with st.form(key="login_form"):        
        st.header("🎈Welcome to QuickView 🎈")
        st.markdown("""---""")
        st.text_input(label="Username", key="auth_name")
        submit = st.form_submit_button("Login", on_click=login)
        if not state.is_authenticated and submit:
            st.error("Invalid username")
        elif state.is_authenticated:
            st.success("Redirecting to home page...")
else:
    global_state['auth_name'] = global_state.get('auth_name')
    st.sidebar.write(f"🎈 Welcome back, {global_state.get('auth_name')} 🎈")
    st.write(state)
    st.sidebar.markdown("""---""")
    st.sidebar.button("Logout", on_click=logout)        
    st.sidebar.markdown("""---""")
    for i, (page_name, page_callback) in enumerate(pages.items(), start=1):
        st.sidebar.button(f"{i}) {page_name}", on_click=page_callback)  
    
    # selected_page = st.sidebar.selectbox("Select a page", page_names_to_funcs.keys())
    # st.markdown(f"# {selected_page}")    
    # st.sidebar.markdown(f"# {selected_page}")            
    # page_names_to_funcs[selected_page]()

    
#%% Collect stock price data via API
# import os
# import pandas as pd
# from quickview.price import SharePrice, International, compile_signals
# from quickview.technical import PriceAction
# from quickview.fundamental import Fundamental
# from quickview.bursamktplc import Bursa
# from datetime import date, timedelta
# from tqdm.auto import tqdm
# from quickview.utils.util import Notification
# from pandas.tseries.offsets import BMonthEnd

# def get_price(code, last_n_days=10):
#     try:
#         return existing_prices[code]['close'].iloc[-last_n_days:].tolist()
#     except KeyError:
#         return None

# today = date.today().strftime("%Y-%m-%d")
# MAX_LAG = 200
# RECIPIENT = "ycfkjc@Hotmail.com"
# SUBJECT = f"Daily KLSE Screener ({today})"

# ROOT_FOLDER = 'C:/Users/ycwong/Desktop/Bursa Quickview'
# DATA_FOLDER = os.path.join(ROOT_FOLDER, "data")
# OUTPUT_FOLDER = os.path.join(ROOT_FOLDER, "output")

# BURSA_METADATA_FILE = os.path.join(DATA_FOLDER, 'Bursa Metadata.xlsx')
# PRICE_DATA_FILE = os.path.join(DATA_FOLDER, "prices.pkl")
# TECHNICAL_FILE = os.path.join(OUTPUT_FOLDER, f"result ({today}).xlsx")
# FUNDAMENTAL_FILE = os.path.join(OUTPUT_FOLDER, "Fundamental Summary.xlsx")
# FUNDAMENTAL_FILE_PICKLE = os.path.join(OUTPUT_FOLDER, "Fundamental Summary.pkl")

# STOCKLIST_FILE  = os.path.join(DATA_FOLDER, 'klse stocklist.xlsx')
# QUARTER_STATEMENT_FILE = os.path.join(DATA_FOLDER, "quarter_statements.pkl")
# ANNUAL_STATEMENT_FILE = os.path.join(DATA_FOLDER, "annual_statements.pkl")

# today = date.today()
# end_of_month = BMonthEnd().rollforward(today) # last business day of current month
# bursa = Bursa(DATA_FOLDER)
# inter = International()
# klse = SharePrice(DATA_FOLDER)

# #%%
# if today == end_of_month:
#     print("1. Collect tickers and its details attributes from Bursa Marketplace")
#     bursa_tickers = bursa.stock_tickers()
#     inputs_df = bursa_tickers['link'].str.extract(r"/mkt/themarket/(\w+)/(\w+)")
#     inputs_df.columns = ['type', 'code']
#     bursa_metadata = bursa.collect_stock_metadata(inputs_df)
#     bursa_df = bursa.write_metadata(bursa_tickers, bursa_metadata, BURSA_METADATA_FILE)

#     print("2. Collect Financial Statements from Bursa Marketplace")
#     stock_list = bursa.load_stocklist(STOCKLIST_FILE)
#     bursa_df = bursa.load_metadata(BURSA_METADATA_FILE)
#     ticker_codes = bursa_metadata.set_index('stockcode')['bursacode'].dropna().to_dict()
#     quarter_statement = bursa.get_statements(ticker_codes, freq='quarter')
#     annual_statement = bursa.get_statements(ticker_codes, freq='annual')

#     print("3. Update Financial Statement to existing data file")
#     old_quarter_statement = bursa.read_pickle(QUARTER_STATEMENT_FILE)
#     updated_qr_stat = bursa.update(old_quarter_statement, quarter_statement, axis=1)
#     bursa.write_pickle(updated_qr_stat, QUARTER_STATEMENT_FILE)

#     old_annual_statement = bursa.read_pickle(ANNUAL_STATEMENT_FILE)
#     updated_an_stat = bursa.update(old_annual_statement, annual_statement, axis=1)
#     bursa.write_pickle(updated_an_stat, ANNUAL_STATEMENT_FILE)


# quarter = bursa.read_pickle(QUARTER_STATEMENT_FILE)
# annual = bursa.read_pickle(ANNUAL_STATEMENT_FILE)

# stock_list = bursa.load_stocklist(STOCKLIST_FILE)
# bursa_metadata = bursa.load_metadata(BURSA_METADATA_FILE)

# gnp = inter.get_gnp()
# market_caps = bursa_metadata.set_index('bursacode')['mktcap'].to_dict()

# fundamental = Fundamental(bursa_metadata)
# metavalue = fundamental.compile_summary(annual, quarter, gnp, market_caps)


# klse_codes = klse.get_stock_codes()
# existing_prices = klse.read_pickle(PRICE_DATA_FILE)

# metavalue = (metavalue.sort_values(by=['Code', 'Date'], ascending=False)
#             .drop_duplicates(['Code'], keep='first')
#             .drop(['Date', 'Year'], axis=1))

# metavalue['Price'] = metavalue['Code'].apply(get_price, last_n_days=30)
# metavalue.dropna(subset=['Price'], inplace=True)

# #%% Collect and update daily share price
# klse_codes = klse.get_stock_codes()
# collected_prices = klse.get_share_price(klse_codes.index)
# existing_prices = klse.read_pickle(PRICE_DATA_FILE)
# # update data based on date information
# updated_prices = klse.update(existing_prices, collected_prices, axis=0, subset=['date'])
# klse.write_pickle(updated_prices, PRICE_DATA_FILE)
# print('Price Data Exported Successfully!')

# signals = compile_signals(updated_prices)
# metavalue = metavalue.merge(signals, left_on='Code', right_index=True, how='left')
# metavalue.to_pickle(FUNDAMENTAL_FILE_PICKLE)
# print("Exported meta value successfully!")

# #%%
# prices = klse.read_pickle(PRICE_DATA_FILE)
# adj_prices = {}
# for code, price in prices.items():    
#     price['date'] = pd.to_datetime(price['date']).dt.date
#     price = price.drop_duplicates(subset=['date'], keep='last')
#     price = price.sort_values(by=['date'], ascending=True).reset_index(drop=True)
#     adj_prices[code] = price

#%% 
# from sklearn.linear_model import LinearRegression
# import numpy as np
# from quickview.price import KLSE
# import pandas as pd

# def get_trends(price, window_size=15):
#     window = price['close'].rolling(window=window_size)
#     window_high = window.max()
#     window_low = window.min()
#     average_gain = window.mean().diff(window_size)
#     rad = np.arctan(average_gain / window_size)
#     degree = rad * 180 / np.pi
#     price['degree'] = degree
#     return price

# ROOT_FOLDER = 'C:/Users/ycwong/Desktop/Bursa Quickview'
# DATA_FOLDER = os.path.join(ROOT_FOLDER, "data")
# PRICE_DATA_FILE = os.path.join(DATA_FOLDER, "prices.pkl")

# klse = SharePrice(DATA_FOLDER)
# prices = klse.read_pickle(PRICE_DATA_FILE)
# price = prices['5279']
# p = get_trends(price)

# pd.set_option("display.max_rows", 100)
# y = price['close'].rolling(window=15).mean().iloc[-100:]
# X = pd.DataFrame(np.arange(1, 1+ len(y)))
# lr = LinearRegression()
# lr.fit(X, y)
# y_pred = lr.predict(X)
# print(lr.coef_, lr.intercept_, lr.score(X, y))
# slope = lr.coef_[0]
# deg = np.arctan(slope) * 180 / np.pi
# print(deg)
# price.tail(100)

