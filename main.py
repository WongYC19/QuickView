#%%
import pandas as pd
from theia.flows import Pickler
from theia.strategies import MA_3, run_all_shares
from theia.companies import SharePrice
from theia.bursamktplc import Bursa

def get_statements():
    bursa = Bursa()
    tickers = bursa.get_stock_tickers()
    metadata = bursa.get_stock_metadata(tickers)
    metadata[bursa.metadata_fields].to_csv("metadata.csv", index=False)
    metadata = pd.read_csv("metadata.csv")

    ticker_codes = metadata.set_index('stockcode')['bursacode'].to_dict()
    quarter_statements = bursa.get_statements(ticker_codes, freq='quarter')
    annual_statements = bursa.get_statements(ticker_codes, freq='annual')

    pickler = Pickler()
    pickler.write(dict_=quarter_statements, file_path="quarter_statements.pkl", is_dataframe=True, merge=True)
    pickler.write(dict_=annual_statements, file_path="annual_statements.pkl", is_dataframe=True, merge=True) 

pickler = Pickler()

sp = SharePrice()
stock_codes = sp.get_stock_codes()
# share_prices = pickler.read("prices.pkl")
share_prices = sp.get_share_price(stock_codes.index)
pickler.write(share_prices, "prices.pkl", merge=False)
signals = run_all_shares(share_prices, MA_3)
signals.to_pickle("MA_3_Signals.pkl")

# %%
from theia.bursamktplc import Bursa
bursa = Bursa()
# tickers = bursa.get_stock_tickers()
# metadata = bursa.get_stock_metadata(tickers)
# tickers.to_csv("tickers.csv", index=False)
# metadata.to_csv("metadata.csv", index=False)
tickers = bursa.get_stock_tickers()
tickers.to_csv("tickers.csv", index=False)
metadata = bursa.get_stock_metadata(tickers)
metadata[bursa.metadata_fields].to_csv("metadata.csv", index=False)
# %%
