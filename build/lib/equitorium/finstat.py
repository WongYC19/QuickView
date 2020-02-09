import os
import re
import json
import swifter
import logging
import requests
import numpy as np
import pandas as pd
from time import sleep

import ctypes
from ctypes.wintypes import MAX_PATH    
from functools import wraps
from getpass import getuser
from logging import warning, info

from ast import literal_eval    
from datetime import datetime
from bs4 import BeautifulSoup
from tqdm.notebook import tqdm    
from scipy.special import expit
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed

pd.set_option('display.max_columns', None)

def multithreads(n_threads=2):  
    def inner_function(function):        
        @wraps(function)
        def wrapper(*args, n_threads=n_threads):            
            if n_threads is None: n_threads = 2 * os.cpu_count()    
            n_threads = n_threads * os.cpu_count()
            n_threads = min(n_threads, len(args))
            with ThreadPoolExecutor(max_workers=n_threads) as executor:
                results = list(tqdm(executor.map(function, *args), total=len(list(args))))
            return results
        return wrapper
    return inner_function

class Score:
    """
    Input: df, concatenated df by rows - Income Statement, Balance Sheet and Cash Flow statement
    
    Sample Code:
    bursa = BursaScraper()
    results = bursa.collect_statements(['ARNK.KL', 'ADVA.KL'], 'annual')
    score = Score(results['ARNK'])        
    score.get_score()
    """            
    
    @classmethod
    def get_gnp(cls):            
        url = 'https://www.macrotrends.net/countries/MYS/malaysia/gnp-gross-national-product'
        response = requests.get(url)
        soup = BeautifulSoup(response.text, features='lxml')
        table = soup.find_all('table', attrs={'class': 'historical_data_table table table-striped table-bordered'})
        table = table[1].find_all('tr')
        gnp = pd.DataFrame([row.text.split('\n') for row in table[3:]])
        gnp.columns = table[1].text.split('\n')
        gnp.drop('', axis=1, inplace=True)
        gnp.GNP = gnp.GNP.str.replace('\$|B', '').astype(float)
        gnp['Per Capita'] = gnp['Per Capita'].str.replace('\$|,', '').astype(float)
        gnp['Growth Rate'] = gnp['Growth Rate'].str.replace('\%', '').astype(float)
        gnp.set_index('Year', drop=True, inplace=True)
        gnp.index = gnp.index.astype('int')    
        #return gnp
        cls.gnp = gnp

    #gnp = get_gnp()
    def __init__(self, ticker):         
        self.ticker = ticker
        #self.fail_scores = []
        try:
            xlsx = pd.ExcelFile(f'{ticker}.xlsx')
        except Exception as e:            
            #self.fail_scores.append(f'\nAnnual report for {ticker} file not found')                        
            warning(f'Annual report for {ticker} file not found')
            return None
        else:
            df = pd.concat((xlsx.parse(sheet, index_col=0) for sheet in xlsx.sheet_names), axis=0, sort=False)
            if df.shape[0] == 0:
                #self.fail_scores.append(f'\nThe financial statement of {ticker} is empty.')
                warning(f'\nThe financial statement of {ticker} is empty.')
                return None
                
        self.index = pd.to_datetime(df.columns, format='%d %b %Y')
        
        try:
            self.revenue = df.loc['Revenue', :]
            self.net_sales = df.loc['Net Sales', :]
            self.cogs = df.loc['Cost of Revenue, Total', :]
            self.sga = df.loc['Selling/General/Admin. Expenses, Total', :]
            self.gross_profit = df.loc['Gross Profit', :]               
            self.ebit = df.loc['Net Income Before Taxes', :]
            self.net_profit = df.loc['Net Income', :]

            self.total_asset = df.loc['Total Assets', :]            
            self.current_asset = df.loc['Total Current Assets', :]
            self.ppe = df.loc['Property/Plant/Equipment, Total - Net', :]
            self.retain_earning = df.loc['Retained Earnings (Accumulated Deficit)', :]        
            self.total_liability = df.loc['Total Liabilities', :]
            self.current_liability = df.loc['Total Current Liabilities', :]
            self.long_debt = df.loc['Total Long Term Debt', :]
            self.total_equity = df.loc['Total Equity', :]

            self.cash = df.loc['Cash and Short Term Investments', :]
            self.operating_cf = df.loc['Cash from Operating Activities', :]
            self.net_receivable = df.loc['Total Receivables, Net', :]
            self.security = df.loc['Security Deposits', :]
            self.depreciation = df.loc['Depreciation/Depletion', :]
            self.share = df.loc['Total Common Shares Outstanding', :]  
        
        except Exception as e:            
            
            warning(f'Ticker: {self.ticker}, not applicable to financial industry.')
            return None
        
        else:
            self.working_capital = self.current_asset - self.current_liability        
            self.f = None
            self.m = None
            self.z = None
            self.o = None    
        finally:
            logging.info('Initialization complete')
        
    def f_score(self):
        revenue = self.revenue
        net_sales = self.net_sales
        gross_profit = self.gross_profit
        net_profit = self.net_profit
        total_asset = self.total_asset
        current_asset = self.current_asset
        long_debt = self.long_debt
        current_liability = self.current_liability
        operating_cf = self.operating_cf
        share = self.share
        
        average_asset = total_asset.rolling(window=2).mean().shift(-1).ffill()
        roa = net_profit / total_asset
        current_ratio = current_asset / current_liability
        gross_margin = gross_profit / net_sales
        asset_turnover = revenue / average_asset

        headers = ['Positive Income', 'Positive ROA', 'Positive OCF', 'OCF > Income', 
                   'Lower Long term debt ratio', 'Higher current ratio', 'No New Share',
                   'Higher Gross Margin', 'Higher asset turnover']

        f_score = pd.concat([net_profit > 0,
                            (roa > 0),
                            (operating_cf > 0),
                            (operating_cf > net_profit),
                            ((long_debt / total_asset).diff(-1).fillna(0) <= 0),
                            (current_ratio.diff(-1).fillna(0) > 0),
                            (share.diff(-1).fillna(0) >= 0),
                            (gross_margin.diff(-1).fillna(0) > 0),
                            (asset_turnover.diff(-1).fillna(0) > 0)], axis=1)
        f_score.columns = headers
        f_score = f_score.applymap(int)    
        f_score.index = self.index.year
        f_score = f_score.sum(axis=1)
        f_score = pd.DataFrame(f_score.values, index=self.index.year, columns=['f_score'])    
        self.f = f_score 
        return f_score

    def m_score(self):       
        net_sales = self.net_sales
        cogs = self.cogs        
        sga = self.sga
        ppe = self.ppe        
        total_asset = self.total_asset
        current_asset = self.current_asset        
        long_debt = self.long_debt
        current_liability = self.current_liability        
        cash = self.cash
        net_receivable = self.net_receivable                
        depreciation = self.depreciation
        security = self.security
        working_capital = self.working_capital
        
        DSR = net_receivable / net_sales
        DSRI = (DSR / DSR.shift(-1)).fillna(0)

        GM = (net_sales - cogs) / net_sales
        GMI = (GM.shift(-1) / GM).fillna(0)

        AQ = 1 - (current_asset + ppe + security) / total_asset
        AQI = (AQ / AQ.shift(-1)).fillna(0)

        SGI = (net_sales / net_sales.shift(-1)).fillna(0)

        DEP = depreciation / (depreciation + ppe)
        DEPI = (DEP / DEP.shift(-1)).fillna(0)

        SGA = sga / net_sales
        SGAI = (SGA / SGA.shift(-1)).fillna(0)

        LVG = (current_liability + long_debt) / total_asset
        LVGI = (LVG / LVG.shift(-1)).fillna(0)

        TATA = (working_capital - cash - depreciation).diff(-1).fillna(0)

        m = -4.84 + 0.92 * DSRI + 0.528 * GMI + 0.404 * AQI + 0.892 * SGI + 0.115 * DEPI - 0.172 * SGAI + 4.679 * TATA - 0.327 * LVGI
        m = pd.DataFrame(m.values, index=self.index.year, columns=['m_score'])
        self.m = m
        return m

    def z_score(self):
        """
            A = working capital / total assets
            B = retained earnings / total assets
            C = earnings before interest and tax / total assets
            D = market value of equity / total liabilities
            E = sales / total assets
            Z-Score = 1.2A + 1.4B + 3.3C + 0.6D + 1.0E
            A score below 1.8 means it's likely the company is headed for bankruptcy, 
            while companies with scores above 3 are not likely to go bankrupt
        """
        working_capital = self.working_capital
        total_asset = self.total_asset
        retain_earning = self.retain_earning
        ebit = self.ebit
        total_equity = self.total_equity
        total_liability = self.total_liability
        net_sales = self.net_sales        
        A = working_capital / total_asset
        B = retain_earning / total_asset
        C = ebit / total_asset
        D = total_equity / (total_liability + total_equity)
        E = net_sales / total_asset
        Z = 1.2 * A + 1.4 * B + 3.3 * C + 0.6 * D + 1.0 * E             
        Z = pd.DataFrame(Z.values, index=self.index.year, columns=['z_score'])
        self.z = Z
        return Z
    
    def o_score(self):
        """
        GNP (float): Gross National Product Index (Growth Rate, %)
        Ohlson O-Score is the result of a 9-factor linear combination of coefficient-weighted business ratios         
        """        
        total_asset = self.total_asset
        total_liability = self.total_liability
        current_asset = self.current_asset
        current_liability = self.current_liability
        net_profit = self.net_profit
        operating_cf = self.operating_cf
        working_capital = self.working_capital
        #if gnp is None: gnp = get_gnp()        
        GNP = self.gnp.loc[list(self.index.year - 1), 'Growth Rate']
        GNP.index = total_asset.index
        X = (total_liability > total_asset).astype('int')
        Y = ((net_profit.shift(1) < 0) | (net_profit < 0)).astype('int')
        Z = (net_profit.diff(-1) / (net_profit.abs() + net_profit.shift(-1).abs())).fillna(0)        
        T = -1.32 - 0.407 * np.log(total_asset/GNP) + 6.03 * total_liability / total_asset - 1.43 * working_capital / total_asset + 0.0757 * current_liability / current_asset - 1.72 *  X - 2.37 * net_profit / total_asset - 1.83 * operating_cf  / total_liability + 0.285 * Y - 0.521 * Z        
        T = T.apply(expit)                
        O = pd.DataFrame(T.values, index= self.index.year, columns=['o_score'])
        self.o = O
        return O
    
    def get_score(self):
        try:
            return pd.concat([self.f_score(), self.z_score(), self.o_score(), self.m_score()], axis=1)
        except Exception as e:                        
            #self.fail_scores.append(f'\nTicker: {self.ticker}, {e}')            
            #warning(f'Ticker: {self.ticker}, {e}')
            return None 