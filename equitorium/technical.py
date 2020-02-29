import os
from warnings import warn
from getpass import getuser

import numpy as np
import pandas as pd
import cufflinks as cf
import plotly.graph_objects as go
from plotly.subplots import make_subplots

pd.set_option('display.max_rows', 100)
cf.set_config_file(offline=True, world_readable=False, theme='ggplot')
INCREASING_COLOR = 'Green' #'rgb(144,238,144)' # '#17BECF'
DECREASING_COLOR = 'Crimson' #'rgb(255,99,71)' # '#7F7F7F'

class Indicators:
    
    def __init__(self, price):
        price.sort_index(ascending=True, inplace=True)
        self.price = price
        self.open = price['open']
        self.close = price['close']
        self.high = price['high']
        self.low = price['low']
        self.volume = price['volume']
        
    def bollinger(self, period=20, n_std=2):
        """ Signal: W-Bottoms: 
            i) Touch lower bands twice
            ii) Bounce towards middle band
            iii) New Price Low (Above lower band)
            iv) Strong move from second low and break resistance
        """
        rolling = self.close.rolling(window=period)    
        rolling_std  = n_std * rolling.std()
        middle_band = rolling.mean()
        upper_band = middle_band + rolling_std
        lower_band = middle_band - rolling_std
        return middle_band, upper_band, lower_band

    def rsi(self, period=14, lower=30, upper=70, signal_only=True):
        changes = pd.DataFrame(self.close.diff(1)).dropna()
        gain = changes.clip(lower=0)
        loss = changes.clip(upper=0).abs()

        first_gain = gain.iloc[:period].sum()
        first_loss = loss.iloc[:period].sum()

        avg_gain = gain.rolling(period).mean().dropna()
        avg_loss = loss.rolling(period).mean().dropna()

        avg_gain.iloc[0] = first_gain
        avg_loss.iloc[0] = first_loss

        avg_gain['prev_gain'] = avg_gain['close'].shift(-1)
        avg_loss['prev_loss'] = avg_loss['close'].shift(-1)

        avg_gain = ((period-1) * avg_gain['prev_gain'] + avg_gain['close']) / period
        avg_loss = ((period-1) * avg_loss['prev_loss'] + avg_loss['close']) / period

        RS = avg_gain / avg_loss
        RSI = 100 - 100 / (1 + RS)
        index = RSI.index        
        RSI.index = index
        RSI.dropna(inplace=True)    
        
        signal = pd.cut(RSI, include_lowest=False, right=True,
                        bins=[0, lower, upper, 100], 
                        labels=['oversold', 'neutral', 'overbought'])        

        if signal_only:
            return signal
        else:
            return pd.concat((RSI, signal), 1)        

    def obv(self):
        diff = self.close.diff(1).fillna(0)
        diff = diff / diff.abs()        
        return (diff * volume).cumsum()       

    def mov_avg(self, price, period=20, ema=False):
        if ema:
            return price.ewm(span=period).mean()
        else:
            return price.rolling(period).mean()

    def ma_crossover(self, period1=50, period2=20, ema=False, signal_only=True):
        ma1 = mov_avg(self.close, period1, ema)
        ma2 = mov_avg(self.close, period2, ema)        
        signal = pd.DataFrame(ma1 - ma2)        
        signal.columns=[f'ma({period1}-{period2})']
        #signal = signal / signal.abs()
        
        if signal_only:
            return signal
        else:
            signal = pd.concat((ma1, ma2, signal), 1)
            signal.columns = [f'ma({period1})', f'ma({period2})', f'ma({period1}-{period2})']
            return signal

    def obv_crossover(self, period1=20, period2=50, signal_only=True):
        obv1 = mov_avg(obv(self.volume), period=period1)
        obv2 = mov_avg(obv(self.volume), period=period2)
        signal = pd.DataFrame(obv1 - obv2)
        signal.columns=[f'obv({period1}-{period2})']

        if signal_only:        
            return signal
        else:
            signal = pd.concat((obv1, obv2, signal), 1)
            signal.columns = [f'obv({period1})', f'obv({period2})', f'obv({period1}-{period2})']
            return signal

    def macd(self, period1=12, period2=26, signal_only=True):
        ema1 = mov_avg(self.close, period1, True)
        ema2 = mov_avg(self.close, period2, True)    
        signal = pd.DataFrame(ema1 - ema2)
        signal.columns=[f'macd({period1}-{period2})']

        if signal_only:        
            return signal
        else:
            signal =  pd.concat((ema1, ema2, signal), 1)
            signal.columns = [f'ema{period1}', f'ema{period2}', 'macd']
            return signal
    
    def support_resistance(self, period=200):
        price = self.price.iloc[-period:, :].drop('volume', 1)
        latest_price = price.close.iloc[-1]
        
        changes = price.close.diff()                
        turning_date = changes[changes * changes.shift(-1) <= 0].index        
        tp_date = price.loc[turning_date, :]        
        active_price = pd.Series(tp_date.values.ravel()).value_counts()
        
        resistances = active_price[active_price.index > latest_price]                
        resistance = resistances[resistances >= 2].index.max()                        
            
        supports = active_price[active_price.index < latest_price]        
        support = supports[supports > 1].index.max()
                
        return support, resistance    
    
    def moving_range(self, windows=20, method='quantile'):
        assert method in ['quantile', 'average', 'median']
        high_roll = self.high.rolling(windows)
        low_roll = self.low.rolling(windows)
        
        if method == 'quantile':
            top = high_roll.quantile(0.9)
            bottom = low_roll.quantile(0.3)
        elif method == 'average':
            top = high_roll.mean()
            bottom = low_roll.mean()
        elif method == 'median':
            top = high_roll.median()
            bottom = low_roll.median()
        return top, bottom
    
    def volatility(self):
        prev_constant = (1 + self.close.rolling(2).max() - self.open.rolling(2).min())
        norm_constant = 1 + (self.high - self.low).abs()
        changes = self.close - self.open
        volatility = changes * prev_constant / norm_constant
        return volatility
    
class Caster(Indicators):
    
    def __init__(self, price_path=None, n_splits=3):
        if price_path is None:
            os.chdir(f'C:/Users/{getuser()}/Documents/bursa/prices')
        else:
            os.chdir(price_path)
        
        self.prices = pd.read_parquet('prices.parquet')        
        self.prices.code = self.prices.code.replace('[A-Za-z]+', '', regex=True).astype(int)
        self.prices = (self.prices.drop_duplicates(subset=['date', 'code'], keep='last')
                       .sort_values(['code', 'date'], ascending=[True, True])
                       .reset_index(drop=True))
        self.prices['volume'] *= 100
        
        self.codes = self.prices['code'].unique()        
        tp_count = self.prices['code'].value_counts(sort=False)
        tp_count = pd.DataFrame(tp_count[self.codes])
        tp_count.rename({'code': 'start'}, axis=1, inplace=True)
        tp_count['end'] = tp_count['start'].cumsum()
        tp_count['start'] = tp_count['end'] - tp_count['start']        
        location = tp_count.to_dict(orient='index')
                
        self.data = {code: (self.prices
                            .iloc[loc['start']:loc['end'], :]
                            .drop('code', 1)
                            .set_index('date')) for code, loc in location.items()}
        
        self.n_splits = int(n_splits)
        self.forward = None
        self.train_margin = None
        self.plots = []
        self.size = 0
        self.total = 0        
        
        self.meta = os.path.join(f'C:/Users/{getuser()}', 'Documents', 'bursa', 'metadata.xlsx')
        assert os.path.isfile(self.meta)
        metadata = pd.read_excel(self.meta)
        metadata = (metadata.dropna(subset=['bursacode'], axis=0)
                            .applymap(lambda x: x.replace('amp;', '') if type(x)== str else x))        
        self.code_name = metadata.set_index('bursacode')['alias'].dropna().to_dict()
        self.code_name.update({name: code for code, name in self.code_name.items()})
        self.sectors = metadata.set_index('bursacode')['industrygroupcode'].dropna().to_dict()                
        
    def get_price(self, code, max_points=None):
        if isinstance(code, str):
            try:
                code = int(code)
            except ValueError:
                code_text = self.code_name[code]
                code = int(code_text)
            else:
                code_text = str(code).rjust(4, '0')                
        else:
            code_text = str(int(code)).rjust(4, '0')
        
        self.code = code        
        
        try:        
            price = self.data[self.code]
        except Exception as e:            
            warn(f'{code} not found. Please check available stock codes by looking at Price().codes')            
        else:                
            self.name = self.code_name[code_text]
            self.sector = self.sectors[code_text]
            self.title = f"{self.name} ({code_text}) \t {self.sector}"
            if max_points is not None and max_points > price.shape[0]:#if max_points is not None: start = max(end-max_points, start)                     
                price = price.iloc[-max_points, :]
            return price        
        
    def show_chart(self, price):
        if 'date' not in price.columns: price = price.reset_index(drop=False)                                        
        vol_colors = np.where(price.close.diff(1) > 0, INCREASING_COLOR, DECREASING_COLOR)                     
        fig = go.Figure(make_subplots(shared_xaxes=True, vertical_spacing= 0.02,
                                  specs=[
                                      [{'type': 'candlestick'}], 
                                      [{'type': 'bar'}]
                                   ], rows=2, cols=1))            

        fig.add_trace(go.Candlestick(x= price.date, name= 'Price (MYR)', 
                   open = price.open, 
                   high = price.high, 
                   low = price.low, 
                   close = price.close, yaxis='y'), row=1, col=1)

        fig.add_trace(go.Bar(x= price.date, 
                         y= price.volume, 
                         name='Volume', 
                         orientation = "v",
                         marker_color=vol_colors, yaxis='y2'), row=2, col=1)

        rangeselector= dict(x = 0, y = 0.96,  font = dict(size=13),
                       bgcolor = 'rgba(150, 200, 250, 0.4)',
                       buttons=[dict(count=1, label='1m', step='month', stepmode='backward'),
                                dict(count=3, label='3m', step='month', stepmode='backward'),
                                dict(count=6, label='6m', step='month', stepmode='backward'),
                                dict(count=1, label='1y', step='year', stepmode='backward'),
                                dict(count=5, label='5y', step='year', stepmode='backward'),
                                dict(label='All', step='all')])

        fig.update_layout(title= dict(text=self.title, x=0.5, y=0.97, 
                                  xanchor='center', yanchor='top'),
                      plot_bgcolor= 'rgba(0, 0, 0, 0)', 
                      margin={'t': 0.1, 'b': 0, 'l': 0, 'r': 0},
                      showlegend= False)    
    
        fig.update_layout(xaxis=dict(autorange=True, type="date", 
                                 rangeslider=dict(autorange=True, visible=True), 
                                 rangeselector=rangeselector),                                                       
                      yaxis1= dict(domain=[0.25, .97]),
                      yaxis2= dict(domain=[0, 0.25]))
                      
        fig.update_xaxes(showgrid=True, gridwidth=2, gridcolor='rgb(234,234,234)', row=1)
        fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgb(234,234,234)', row=1)                
        return fig  
                
    def plot_prediction(self, y, yhat):
        plt.style.use('fivethirtyeight')
        plt.figure(figsize=(15, 4))        
        
        y = np.array(y).ravel()
        yhat = np.array(yhat).ravel()
        index = len(y)-len(yhat)
        test_rng = range(index, len(y))        
        plt.plot(range(1, len(y)+1), y, color='b', linewidth=2, label='Original')            
        plt.plot(test_rng, yhat, color='r', linewidth=2, alpha=0.8, label='Predicted')
        plt.axvline(x=index, linestyle='--', color='grey', linewidth=1.5)    
        plt.axvspan(xmin=index, xmax=len(y)+1, facecolor='c', alpha=0.1)

        y_test = y[test_rng]
        mse = mean_squared_error(y_test, yhat)
        mae = mean_absolute_error(y_test, yhat)
        plt.title('Mean Square Error: %.2f \nMean Absolute Error: %.2f'%(mse, mae), fontsize=12)    
        plt.legend(fontsize=14)
        
    def forecast(self, code, n_steps=3, train_size=0.8, max_size=None, plot=True):        
        price = self.get_price(code)
        self.price = price
        data = price.close
        if max_size is not None: data = data[-max_size:]
        
        X, y = self.supervised_ts(data, n_steps=n_steps)
        idx = int(train_size*X.shape[0])
        X_train, X_valid = X[:idx], X[idx:]    
        y_train, y_valid = y[:idx], y[idx:]    

        lr = LinearRegression()
        lr.fit(X_train, y_train)
        train_pred = lr.predict(X_train)
        valid_pred = lr.predict(X_valid)
        
        train_mse = mean_squared_error(y_train, train_pred)
        train_mae = mean_absolute_error(y_train, train_pred)
        train_r2 = r2_score(y_train, train_pred)
        
        valid_mse = mean_squared_error(y_valid, valid_pred)
        valid_mae = mean_absolute_error(y_valid, valid_pred)
        valid_r2 = r2_score(y_valid, valid_pred)
        
        print('Result of training set')
        print('---------------------------')
        print('Mean square error: {:.2f}'.format(train_mse))
        print('Mean absolute error: {:.2f}'.format(train_mae))
        print('R square: {:.2f}'.format(train_r2))
        
        print('\nResult of training set')
        print('---------------------------')
        print('Mean square error: {:.2f}'.format(valid_mse))
        print('Mean absolute error: {:.2f}'.format(valid_mae))
        print('R square: {:.2f}'.format(valid_r2))
                
        if plot: self.plot_prediction(y, valid_pred)
        predictions = lr.predict(X)        
        return predictions