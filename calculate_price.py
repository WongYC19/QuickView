# Last modified date: 12 Apr 2021
import os
import sys
import numpy as np
import pandas as pd
sys.path.insert(1, os.getcwd())

from datetime import timedelta
from tqdm.auto import tqdm
from warnings import warn
from datetime import date
from decimal import Decimal
from sklearn.linear_model import LinearRegression

from helper import Notification

class PriceAction(): 
    
    def __init__(self, price, o='open', h='high', l='low', c='close', v='volume', max_periods=None): 
        """
            Derive Price Action Data from raw OHLCV Price Data.
            Parameters:
            -----------
            price (Pandas DataFrame): column: open, high, low, close, volume, index: int range
            o (str): column label for "open" price, default open
            h (str): column label for "high" price, default high
            l (str): column label for "low" price, default low
            c (str): column label for "close" price, default close
            v (str): column label for "volume" price, default volume
            max_period (int): Most recent number of price data for calculation, default to using full data
        """
        
        remaining_cols = list(set([o, h, l, c, v, 'date']) - set(price.columns))
        assert not remaining_cols, f"Missing columns: {remaining_cols}"
        
        self.price = price.copy()        
        self.price = self.price.sort_values(by='date', ascending=True).reset_index(drop=True)
        
        data_size = self.price.shape[0]
        
        if max_periods is not None: 
            if data_size > max_periods:
                self.price = self.price.iloc[-max_periods:]
#             else:
#                 warn(f"The data size {data_size} is less than the specified period {max_periods}. Skip slicing.")
            
        self.o = self.price[o]
        self.h = self.price[h]
        self.l = self.price[l]
        self.c = self.price[c]
        self.v = self.price[v]
                                    
    def get_difference(self, n_period=1, normalize=True, inplace=True):
        """
            To calculate the difference in close price between N days
        """
        close_diff = self.c.diff(n_period)
        if normalize:
            close_diff_pct = close_diff.abs() / (close_diff.abs() + self.c)            
            
        if inplace: 
            self.price[f'close_diff_{n_period}'] = close_diff
            if normalize: 
                self.price[f'close_diff_pct_{n_period}'] = close_diff_pct
                
        return close_diff        
                        
    def get_body_size(self, inplace=True):
        """
            To calculate the size of the candlestick body (without +/- sign)
        """
        body_size = (self.c - self.o).abs()
        if inplace: self.price['body_size'] = body_size
        return body_size
    
    def get_shadow_to_range_ratio(self, inplace=True):
        """
            To calculate the ratio of non-body part in candletick to range of candlestick
        """
        shadow = np.where(self.c > self.o, self.h - self.c, self.c - self.l)        
        uncertainty = shadow / (self.h - self.l)
        uncertainty[uncertainty == np.inf] = 0
        uncertainty = uncertainty.fillna(0)
        if inplace: self.price['uncertainty'] = uncertainty
        return uncertainty
    
    def get_overlapped_size(self, inplace=True):
        """
            To calculate the percentage of intersected price range between candlesticks
        """
        def overlapping(x):    
            a = np.arange(x[0], x[1], 0.005)
            b = np.arange(x[2], x[3], 0.005)
            o = len(set(b) & set(a))    
            if len(a) == 0: return 0
            return o / len(a)
        
        comb = pd.concat([self.c, self.o], axis=1)
        top = comb.max(axis=1)
        bottom = comb.min(axis=1)
        overlapped = (pd.concat([bottom.shift(1), top.shift(1), bottom, top], axis=1)
                     .fillna(0).apply(overlapping, axis=1))
        
        if inplace: self.price['overlapped'] = overlapped
        return overlapped
    
    def get_volume_total_ratio(self, n_period=10, inplace=True):
        """
            To calculate the ratio of specified date volume to the total volume of last N period 
        """
        volume_sum = self.v.rolling(n_period).sum()
        volume_pct = self.v / volume_sum
        volume_pct[volume_pct == np.inf] = 0        
        if inplace: self.price['volume_total_ratio'] = volume_pct
        return volume_pct
    
    def get_close_to_high_ratio(self, n_period=10, inplace=True):
        """        
            To calculate the ratio of closing price to previous high in last N period
        """                      
        close_to_high_ratio = self.c / self.h.rolling(n_period).max().fillna(self.c)
        if inplace: self.price['close_to_high_ratio'] = close_to_high_ratio
        return close_to_high_ratio
                    
    def get_accumulated_gain(self, n_period=10, inplace=True):
        """            
            To calculate the accumulated gain/loss of price between open and close for last N period
        """
        accumulated_gain = (self.c - self.o).rolling(n_period).sum().fillna(0)                
        self.price['accumulated_gain'] = accumulated_gain
        return accumulated_gain
    
    def get_volume_distribution(self, n_period=10, inplace=True):
        """
            To calculate the skewness of volume for last N period
        """
        volume_distribution = self.v.rolling(n_period).skew().fillna(0)
        if inplace: self.price['volume_distribution'] = volume_distribution
        return volume_distribution                          
    
    def get_target(self, n_period=10, inplace=True):
        """
            To calculate the percentage of gain over next N period
        """
        gained_count = ((self.c - self.o) > 0).rolling(n_period).mean()
        if inplace: self.price['target'] = gained_count
        return gained_count
    
    def get_range(self, n_period=10, inplace=True):
        """
            To calculate the position of latest price based on last n period range
        """        
        high = self.c.iloc[-n_period:].max()
        low = self.c.iloc[-n_period:].min()
        current = self.c.iloc[-1]
        if abs(high - low) < 0.005:
            n_range = None
        else:
            n_range = (current - low) / (high - low) 
        
        if inplace: self.price[f'{n_period} days range'] = n_range
        return n_range
            
def get_feature(price, max_periods=None): 
    if price.shape[0] <= 1: return price
    pa = PriceAction(price, max_periods=max_periods)
    
    pa.get_accumulated_gain(5)
    pa.get_body_size()
    try:
        pa.get_overlapped_size()
    except Exception as e:
        print(e)
    pa.get_shadow_to_range_ratio()

    pa.get_close_to_high_ratio(5)
    pa.get_difference(1)
    pa.get_difference(5)
    pa.get_difference(20)
    
    pa.get_volume_distribution(5)
    pa.get_volume_total_ratio(5)
    pa.get_target(5)
    
    pa.get_range(50)
    pa.get_range(100)
    pa.get_range(200)
    return pa.price

FOLDER = "C:/Users/ycwong/Desktop/Bursa Quickview/data"
BURSA_METADATA_FILE = 'Bursa Metadata.xlsx'
PRICE_DATA_FILE = "prices.pkl"
today = date.today().strftime("%Y-%m-%d")
OUTPUT_FOLDER = 'C:/Users/ycwong/Desktop/Bursa Quickview/output/'
OUTPUT_FILE = f"result ({today}).xlsx"
MAX_LAG = 200

bursa = pd.read_excel(os.path.join(FOLDER, BURSA_METADATA_FILE))
prices = pd.read_pickle(os.path.join(FOLDER, PRICE_DATA_FILE))    
feature_list = [pd.Series(dtype=float)] * len(prices.keys())

for i, code in enumerate(tqdm(list(prices.keys()))):
    if len(code) != 4: continue
    price = prices[code]
    if price is None: 
        print(f"No data for {code}, skip process.")
        continue
    feature = get_feature(price, MAX_LAG)    
    feature_list[i] = feature.iloc[-1]    
    
features = pd.concat(feature_list, axis=1) 
features = features.T.reset_index(drop=True)
features.sort_values(by=['close_to_high_ratio', 'volume_total_ratio'], inplace=True)

meta_cols = ["bursacode", "name", "alias", "description", "economicsectorcode", "industrygroupcode"]
bursa_meta = bursa[meta_cols].dropna(subset=['bursacode']).rename(columns={"bursacode": "code"})
results = bursa_meta.merge(features, on='code', how='right')
results['date'] = results['date'].dt.date
results['close'] = results['close'].astype(float)
results = results[results['code'].str.len() <= 4]
results = results.drop(['open', 'high', 'low', 'volume'], axis=1)
results = results.rename(columns={'economicsectorcode': 'Sector', 'industrygroupcode': 'Industry'})
results = results[results['date'] >= results['date'].max() - timedelta(days=3)]
results = results.dropna(subset=['name'])
results.sort_values(by=['Sector', 'Industry', '50 days range', '100 days range', '200 days range'])
os.chdir(OUTPUT_FOLDER)
results.to_excel(OUTPUT_FILE, index=False)

tables = []
noti = Notification(TO="ycfkjc@Hotmail.com")        

for lag in [1, 5, 20]:
    col = f'close_diff_{lag}'
    pct_col = f'close_diff_pct_{lag}'
    table = results[['code', 'alias', 'Sector', 'Industry', 'close', col, pct_col]].copy()
#     table = table.sort_values(by=[pct_col, col], ascending=False)
    table['Industry'] = table['Industry'].fillna('-')
    table = table[table['close'] > 0.2]
    table[col] = table[col].apply(noti.rounding)
    table[pct_col] = table[pct_col].apply(noti.rounding)    
    table = table.dropna(axis=0, subset=[col, pct_col])
    top_gain = noti.create_html(table.sort_values(by=[pct_col, col], ascending=False).head(10), gain=True)
    top_loss = noti.create_html(table.sort_values(by=[pct_col, col], ascending=True).head(10), gain=False)
    tables.append(top_gain)
    tables.append(top_loss)

HTML = noti.get_style() + '<br><br>'.join(tables)
noti.send_email(f"Daily KLSE Screener ({today})", HTML=HTML, FILENAMES=[OUTPUT_FILE])