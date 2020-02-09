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
        
class BursaScraper:    
    """
    path (str): Absolute path, default to "Documents", it is the location to store scraped information as well    
    
    Sample Code:
    
    bursa = BursaScraper()            
    tickers = bursa.collect_tickers() # collect stock tickers for statement use
    statements = bursa.collect_statements(codes='all', frequency='annual') # get financial statements
    scores = bursa.calculate_scores() # an overview of financial health for a company
    share_prices = bursa.get_price() # get share prices     
    commodity_prices = bursa.collect_commodities()        
    prices = bursa.read_price(111)
    """    
    
    code_file = 'codes.xlsx'
    meta_file= 'metadata.xlsx'
    ticker_file = 'tickers.xlsx'
    price_file= 'prices.parquet'    
    
    def __init__(self, path=None):                
        if path is None:                                     
            dll = ctypes.windll.shell32
            buf = ctypes.create_unicode_buffer(MAX_PATH + 1)
            if dll.SHGetSpecialFolderPathW(None, buf, 0x0005, False):
                path = buf.value
            else:
                path = f'C:/Users/{getuser()}/Documents'
                
        folder = os.path.join(path, 'bursa')
        if not os.path.exists(folder): os.mkdir('bursa')        
        self.path = folder
        os.chdir(folder)         
        self.meta = None
        self.response = None
        self.text = None
        self.total_page = None
        self.records = None
        self.prices = None
        self.stock_infos = []
        self.fail_tickers = []
        self.fail_codes = []
        self.fail_statements = []        
        self.tickers = pd.DataFrame()
    
    def log_error(self, file_name='logger.txt'):              
        os.chdir(self.path)
        tickers = self.fail_tickers
        statements = self.fail_statements
        codes = self.fail_codes        
        with open(file_name, 'a+') as log:
            for ticker in tickers: log.write(ticker)
            for statement in statements: log.write(statement)
            for code in codes: log.write(code)            
                                                 
    def sample(self, n_ticker=10, seed=None):
        """ Randomly picked n sample from all tickers """
        tickers = self.collect_ticker().stockcode                    
        if seed is not None: np.random.seed(seed)                     
        idx = np.arange(tickers.shape[0])
        np.random.shuffle(tickers)             
        return tickers[:n_ticker].tolist()
                                
    def collect_ticker(self, local=True, replace=True):
        """
            Faster to collect available tags and stock name from bursa marketplace than collect_metadata, but less details
            Collect tickers (tag and company name in excel file name: tickers.xlsx) from BursaMarketplace
            local (bool): True will read the excel file instead of collecting from website
            replace (bool): True will collect tickers from website and replace excel file, 
                            ignore this argument if local=True
        """        
        
        if local: 
            try:                
                ticker_file = os.path.join(self.path, self.ticker_file)
                return pd.read_excel(ticker_file)
            except Exception as e:
                logging.warning(f'{e}\nFail to read file from local path, extracting from website...')                
                pass            
                
        page = 1                
        ticker_url =  f'http://www.bursamarketplace.com/index.php?screentype=stocks&board=all&tpl=screener_ajax&type=getResult&action=listing&pagenum={page}&sfield=name&stype=desc'                                
        self.response = requests.get(ticker_url)
        self.text = json.loads(self.response.text)
        self.total_page = int(self.text['totalpage'])        
        
        for i in tqdm(range(1, self.total_page+1)):        
            for attempt in range(3):
                try:                    
                    ticker_url = f'http://www.bursamarketplace.com/index.php?screentype=stocks&board=all&tpl=screener_ajax&type=getResult&action=listing&pagenum={page}&sfield=name&stype=desc'                                                    
                    page += 1
                    response = requests.get(ticker_url)
                except:
                    fail_msg = f'\nFail at page {i}, attempt {attempt}'
                    logging.warning(fail_msg)    
                    self.fail_tickers.append(f'{datetime.now()}, {fail_msg}, attempt {attempt}')                    
                    sleep(0.5)
                else:
                    break
        
            self.text = json.loads(response.text)
            records = self.text['records']            
            if isinstance(records, dict): records = list(records.values())
            self.records = records            
            self.stock_infos.append([{k:v for k, v in record.items()} for record in records])
                
        self.tickers = pd.DataFrame.from_dict(sum(self.stock_infos, []))                
                
        try:
            numeric_cols = ['avgrating', 'price', 'mktcap', 'pev', 'pbv', 'estpeg', 'divyld']
            self.tickers[numeric_cols] = self.tickers[numeric_cols].astype(float)
        except:
            pass
        
        if replace:
            output_path = os.path.join(self.path, self.ticker_file)
            self.tickers.to_excel(output_path, sheet_name='stock_info', index=False)
            logging.info(f'Tickers output to {output_path}')            
            
        return self.tickers
    
    def collect_metadata(self, tickers=None, local=True, replace=True, stock_only=True):
        """ 
            ticker (str): Example: ADVN, the cash tag from collect_ticker method without '$' sign
            replace (str): True, write the output file to local drive
            stock_only=True, select equities only (Exclude ETF and etc)                
            Output: Metadata (Include stock code)                         
        """
        cols = ['stockcode', 'bursacode', 'bloomberg', 'alias', 'name', 'economicsectorcode', 'industrygroupcode', 'businesssummary', 
                'srscoreanalyst', 'srscorefund', 'srscorerv', 'srscorerisk', 'srscoretech', 
                'srscoreavg', 'srscorerectxt', 'sdate', 'exdivdate', 'divpaydate', 'mktvalue']        
        metadata = []        
        os.chdir(self.path)
        
        if os.path.exists(self.meta_file) and local: 
            meta = pd.read_excel(self.meta_file)
            meta = meta[cols]
            if stock_only: 
                meta = meta[(meta.stockcode != '$') & (meta.bursacode.str.contains('-') == False) & (meta.bursacode.str.contains(r'\d{5}')==False)]                        
            self.meta = meta
            return meta                    
                    
        if tickers is None: 
            company_list = self.collect_ticker(local=True, replace=False)            
            tickers = company_list.stockcode.tolist()
                
        tickers = pd.Series(tickers).astype('str').str.replace(r'\.KL', '').str.replace(r'^\$', '')
                       
        for ticker in tqdm(tickers, total=len(tickers)):                     
            for types in ['stock', 'reit']:
                url = f'http://www.bursamarketplace.com/index.php?tpl={types}_ajax&type=gettixdetail&code={ticker}'                
                response = requests.get(url)
                #return response
                if response.status_code == 200:
                    json = response.json()                        
                    if json['stockcode'] != '$': break                    
                else:                
                    fail_msg = f'\nFail to connect to {url}.'
                    warning(fail_msg)    
                    self.fail_tickers.append(f'{datetime.now()}, {fail_msg}')                    
                                            
            dict_ = {k:v for k,v in json.items() if not isinstance(v, dict)}            
            metadata.append(pd.DataFrame.from_dict(dict_, orient='index').T)
            
        meta_df = pd.concat(metadata, axis=0, sort=False)                
        meta_df = meta_df[cols]
        meta_df.drop_duplicates(subset=['stockcode'], inplace=True, keep='last')
        
        if replace: meta_df.to_excel(self.meta_file, index=False)
        if stock_only: 
            meta_df = meta_df[(meta_df.stockcode != '$') & (meta_df.bursacode.str.contains('-') == False) & (meta_df.bursacode.str.contains(r'\d{5}')==False)]
        
        self.meta = meta_df
        return meta_df
    
    def collect_statement(self, code=None, frequency='quarter', verbose=True):
        fq = {'quarter': 'qr', 'annual': 'yr'}[frequency]
        st_code = {'INC': 'is', 'BAL': 'bs', 'CAS': 'cf'}        
        file = code.split('.')[0] + '.xlsx'    
        results = dict()        
        missing_data = []
        
        if os.path.exists(file):        
            xlsx = pd.ExcelFile(file)
            for st in st_code.keys(): results[st] = xlsx.parse(sheet_name=st, index_col=0)    
        else:
            for st in st_code.keys(): results[st] = pd.DataFrame()
                
        for st in st_code.keys():
            url = f'http://www.bursamarketplace.com/index.php?tpl=financial_ajax&type=stockfin{st_code[st]+frequency}&code={code}&fintype={st}'                        
            try:
                response = requests.get(url)                
            except:
                self.fail_tickers.append(f'{datetime.now()}, Fail to collect ticker: {code}, frequency: {st}')
                continue
                
            soup = BeautifulSoup(response.text, features="lxml")
            get_index = soup.find_all('div', attrs={'class': 'tb_cell tb_metr'})
            index = pd.Series([cell.text for cell in get_index if cell.text not in ['', '\xa0', 'MYR (MILLION)']])        
            
            if index.shape[0] == 0:                                                 
                missing_data.append(st)
                continue
            
            periods = pd.Series(re.findall(f'tb_cell tb_{fq}\d\d', str(soup)))            
            periods = periods.str.replace(f'tb_cell tb_{fq}', '').astype('str').unique()
            year, headers = '', []            
            
            for idx, i in enumerate(periods):            
                div = f'tb_cell tb_{fq}{i}'

                if fq == 'qr':                                                                     
                    get_column = soup.find_all('div', attrs={'class': div})                         
                    if len(get_column) == 0: get_column = soup.find_all('div', attrs={'class': div + " tb_div"})      
                    col_value = pd.Series([cell.text for cell in get_column])                          
                    if col_value[0] != '\xa0': year = col_value[0]                                            
                    header, col_value = col_value[1] + ' ' + year, col_value[2:]                            
                    headers.append(header)

                elif fq == 'yr':
                    get_column = soup.find_all('div', attrs={'class': div})                    
                    col_value = pd.Series([cell.text for cell in get_column])
                    if i == '01': headers, col_value = col_value.iloc[:len(periods)], col_value.iloc[len(periods):]

                try:
                    col_value = col_value.str.replace(',', '') # remove ',' from thousand separator
                except: 
                    pass
                                                
                col_value.index = index
                results[st][headers[idx]] = col_value.apply(lambda x: re.sub(r'-', str(0), x)).astype(np.float)
            
            cols = results[st].columns            
            cols = pd.Series(index=pd.to_datetime(cols, format='%d %b %Y'), data=cols)
            cols.sort_index(ascending=False, inplace=True)
            results[st] = results[st][cols.values] 
            
        if len(missing_data) > 0: 
            warn_msg = f"\n Could not find data from {code}-{frequency} {', '.join(missing_data)}. \n Please verify at http://www.bursamarketplace.com/mkt/themarket/stock/{code.split('.')[0]}/financials"
            warning(warn_msg)                            
            self.fail_statements.append(f'{datetime.now()}, {warn_msg}')            
                     
        if verbose:            
            logging.info(f'Collection complete: {code}, frequency: {frequency.capitalize()}')                            
        
        code = code.split('.')[0]                  
        return code, results
    
    def collect_statements(self, codes=None, frequency='quarter', verbose=True, replace=True, n_jobs=None):     
        """ Input: 
                codes: (list of str) XXXX.KL or 'all' to run all available tickers   
                frequency (str): quarter or annual (default: quarter)
                verbose (bool): True, notify the progress of extraction (default: True)
            Output: (dict of pandas dataframe) key: INC, BAL, CAS
        """                
        if str(codes).lower() == 'all':
            #ticker = self.collect_ticker(local=True, replace=True)
            ticker = self.collect_metadata(local=True, stock_only=True, replace=False)
            codes = ticker.stockcode.str.replace(r'^\$', '')
        else:
            codes = list(set(codes))
        
        codes = [code if re.match(r'(\.KL$)', code) is not None else code + '.KL' for code in codes]
        
        storage = 'statements/' # folder for financial statement
        if os.path.isdir(storage) == False: 
            logging.info(f'Folder {storage} not found. Created a new one.')
            os.mkdir(storage)        
        os.chdir(storage)
                 
        if os.path.isdir(frequency) == False:                
            logging.info(f'Folder {storage}/{frequency} not found. Created a new one.')
            os.mkdir(frequency)                 
        os.chdir(frequency)
        
        if n_jobs is None or n_jobs < 1: n_jobs = os.cpu_count()        
        
        with ThreadPoolExecutor(max_workers=n_jobs) as executor:
            futures = {executor.submit(self.collect_statement, code=code, 
                       frequency=frequency, verbose=verbose): code for code in codes}
            
            results = list((future.result() for future in tqdm(as_completed(futures), total=len(codes))))
        #results = {code.split('.')[0]: self.collect_statement(code, frequency, verbose) for code in tqdm(codes)}        
        self.fail_statements.append(f'\n{datetime.now()}, --- Extraction Done.')
                
        if replace:                        
            for ticker, statements in results:#results.items():
                if sum(statements.values()).shape[0] > 0:
                    with pd.ExcelWriter(f'{ticker}.xlsx', engine='xlsxwriter') as writer:                
                        for sheet_name, statement in statements.items():                                    
                            statement.to_excel(writer, sheet_name=sheet_name, index=True)                            
            self.fail_statements.append(f'\n{datetime.now()}, --- Process Done.')
        self.log_error()
        return results
        
    @multithreads(3)
    def collect_price(code):
        url = f'https://www.klsescreener.com/v2/stocks/chart/{code}'        
        for _ in range(3):
            response = requests.get(url)            
            if response.status_code == 200: break

        if response.status_code != 200:
            warning(f'Could not retrieve share price for {code}. Kindly verify at {url}')
            return None

        try:
            soup = BeautifulSoup(response.content, features="lxml")
            js = soup.findAll('script', attrs={'type': 'text/javascript'})[16].text
            start = js.find('data')
            js = js[start:]
            end = js.find(';')
            text = js[len('data = '):end].replace('\n', '').replace('\t', '')                                

        except Exception as e:
            #self.fail_codes.append(f'\n{datetime.now()} --- Unable to extract historical price of {code}. {e}')     
            warning(f'\n{datetime.now()} --- Unable to extract historical price of {code}. {e}')

        else:
            data = pd.DataFrame(literal_eval(text), columns=['date', 'open', 'high', 'low', 'close', 'volume'])
            data['date'] = pd.to_datetime(data['date'], unit='ms')
            data['code'] = code                         

            null = data.isna().sum(axis=1)
            if null.sum() > 0:                 
                dates = list(null[null > 0].index)
                warning(f'Null value detected in {code}, date: {dates}')
            return data
        
    def get_price(self, codes=None, get_all=True):
        """ Input: 
                codes (list): list of 4 digits (str/int) stock codes 
                get_all (bool): extract all available share price if True
            Output: A DataFrame            
        """                
        if codes is None and get_all == False: return None
        if codes is not None: get_all = False
        prices = []
        os.chdir(self.path)        
        
        storage = 'prices/' # folder for prices data
        if os.path.isdir(storage) == False:              
            logging.info(f'Folder {storage} not found. Created a new one.')
            os.mkdir(storage)        
        os.chdir(storage)
        
        file = self.price_file
        
        if get_all:
            company_list = self.collect_metadata(tickers=None, local=True, replace=True, stock_only=True)
            codes = company_list.bursacode.tolist()
        else:
            codes = list(map(str,codes))        
                
        prices = BursaScraper.collect_price(codes)
        
        if os.path.exists(file):        
            local_prices = pd.read_parquet(file)                
            prices.append(local_prices)
        else:
            warning(f'Storage of share price {file} not found. New one will be created.')            
        
        prices_df = pd.concat(prices, axis=0, sort=False)        
        prices_df.drop_duplicates(subset=['date', 'code'], inplace=True, keep='first')        
        prices_df['code'] = prices_df['code'].replace('[A-Za-z]+', '', regex=True).astype(int)                        
        prices_df.drop_duplicates(subset=['date', 'code'], keep='last', inplace=True)
        prices_df.sort_values(by=['code', 'date'], ascending=[True, True], inplace=True)
        prices_df.to_parquet(file)            
        
        if len(self.fail_codes) > 0: self.log_error()            
            
        return prices_df
        
    def investing(self, ascending=False, frequency='Daily', **kwargs):        
        """
            ascending (bool): sort the order by Date, default = True
            frequency (str): the interval of date index, possible value = 'Daily', 'Weekly', 'Monthly', default = 'Daily'
            **kwargs (header, smlID, curr_id) for commodity selection
        """

        url = 'https://www.investing.com/instruments/HistoricalDataAjax'
        payload = {param: arg for param, arg in kwargs.items()}        

        date_format = {'Daily': '%b %d, %Y', 'Weekly': '%b %d, %Y', 'Monthly': '%b %y'}    
        if frequency not in date_format.keys(): 
            raise AssertionError("Only 'Daily', 'Weekly', or 'Monthly' allowed for frequency argument")

        if 'st_date' not in payload.keys(): 
            st_date = datetime.strptime("01/01/1950", '%m/%d/%Y').strftime('%m/%d/%Y')
            payload['st_date'] = st_date

        if 'end_date' not in payload.keys():         
            payload['end_date'] = datetime.today().strftime('%m/%d/%Y')    

        payload['sort_col'] =  'date'
        payload['action'] = 'historical_data'
        payload['sort_ord'] = 'ASC' if ascending else 'DESC'
        payload['interval_sec'] = frequency    

        header = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 6.3; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/78.0.3904.108 Safari/537.36',
            'X-Requested-With': 'XMLHttpRequest' 
        }    

        today = pd.to_datetime('now')        
        response = requests.post(url, data=payload, headers=header)
        soup = BeautifulSoup(response.content, features="lxml")

        table = soup.find(attrs={'class': "genTbl closedTbl historicalTbl"})

        cells = table.findAll({'td': 'data-real-value'})
        cells = [cell.text for cell in cells]
        
        columns = table.find(name='tr').findAll('th')
        columns = [c.text for c in columns]
        
        starts = range(0, len(cells), len(columns))
        ends = range(len(columns), len(cells)+1, len(columns))
        
        columns = list(map(lambda x: x.lower(), columns))
        df = pd.DataFrame((cells[s:e] for s, e in zip(starts,ends)), columns=columns)
        df = df.iloc[1:, :]
        df['date'] = pd.to_datetime(df['date'], format=date_format[frequency])
        df = df[df['date'] != pd.Timestamp(today.year, today.month, today.day)]
        
        if 'vol.' in df.columns:
            df.rename(columns={'vol.': 'volume'}, inplace=True)
            multiplier = df['volume'].str.extract(r'([A-Za-z]$)').replace({'K':10e2, 'M': 10e5, 'B':10e8}).fillna(1)
            df['volume'] = (df['volume']
                            .replace(r'K|M|B', '', regex=True)
                            .replace(r'-$', '0', regex=True)
                            .astype(float) * multiplier.T).T
        df['change %'] = df['change %'].str.replace('%', '')        
        df.loc[:, df.columns != 'date'] = df.loc[:, df.columns != 'date'].replace(',', '', regex=True)
        df.loc[:, df.columns != 'date'] = df.loc[:, df.columns != 'date'].applymap(float)
        return df

    def index_mundi(self, commodity='chicken'):
        """ Accepted input for commodity (str): chicken, plywood and etc""" 
        url = f'https://www.indexmundi.com/commodities/?commodity={commodity}&months=240&currency=myr'    
        response = requests.get(url)
        soup = BeautifulSoup(response.text, features="lxml")
        table = soup.find(attrs={'class': 'tblData'}).findAll('tr')
        df = pd.DataFrame((row.findAll('td') for row in table[1:]), columns=['date', 'price', 'change(%)'])
        df = df.applymap(lambda x: x.text)
        df = df.replace(r' %', '', regex=True).replace(r'-$', '0.00', regex=True)
        df['date'] = pd.to_datetime(df['date'], format='%b %Y')
        df.loc[:, df.columns != 'date'] = df.loc[:, df.columns != 'date'].applymap(float)
        return df

    def treasury_yield(self, full=True):        
        if full:
            url = 'https://www.treasury.gov/resource-center/data-chart-center/interest-rates/pages/TextView.aspx?data=yieldAll'
        else:
            url = f'https://www.treasury.gov/resource-center/data-chart-center/interest-rates/pages/TextView.aspx?data=yieldYear&year={datetime.today().year}'

        response = requests.get(url)
        soup = BeautifulSoup(response.text, features='lxml')

        dataset = soup.find(attrs={'class': 't-chart'})
        headers = dataset.findAll(attrs={'scope': "col"})
        headers = [header.text.lower() for header in headers]

        indices = dataset.findAll(attrs={'scope': 'row'})
        indices =[index.text for index in indices]

        table = dataset.findAll(attrs={'class': "text_view_data"})

        n_cells = len(table)
        n_rows = len(indices)    
        n_cols = int(n_cells/n_rows)

        df = pd.DataFrame([table[i*n_cols:(i+1)*n_cols] for i in range(n_rows)], columns=headers)             
        df = df.applymap(lambda x: x.text.strip()).replace('N/A', '0', regex=True)
        df['date'] = pd.to_datetime(df['date'], format='%m/%d/%y')            
        df.loc[:, df.columns != 'date'] = df.loc[:, df.columns != 'date'].applymap(float)
        return df
        
    def calculate_scores(self, annual_path=None, update=False):
        if update: self.collect_statements(codes='all', frequency='annual', replace=True)        
        if annual_path is None: annual_path = os.path.join(os.getcwd(),'statements/annual')
        output_path = os.path.join(self.path, "Scores.xlsx")                
        
        codes = self.collect_metadata(local=True, stock_only=True)
        codes = codes.stockcode.replace(r'\$', '', regex=True)                
        codes.index= codes.values        
        codes.drop_duplicates(inplace=True)        
        logging.info("Codes Loaded.")
        
        os.chdir(annual_path)        
        warnings.filterwarnings('ignore', message=r'^WARNING:root:Annual report for')
        d = codes.swifter.apply(lambda code: Score(code).get_score()).to_dict()        
        scores = pd.concat(d.values(), keys=d.keys())        
        scores.to_excel(output_path)        
        print(f'The summary scores files has been exported to {output_path}')
        os.chdir(self.path)        
        return scores
        
    def read_price(self, code, plot=True):        
        try:
            if isinstance(self.prices, pd.DataFrame) == False:
                price_file = os.path.join(os.getcwd(), 'prices', self.price_file)
                prices = pd.read_parquet(price_file)                            
                self.prices = prices
        except Exception as e:
            warning(e)
        else:                                
            prices.code = prices.code.replace('[A-Za-z]+', '', regex=True).astype(int)
            if int(code) in prices.code.unique():
                prices = prices[prices.code == int(code)].drop('code', axis=1)        
                if plot:
                    st, ed = str(prices.date.min()), str(prices.date.max())
                                        
                    fig = make_subplots(specs=[[{"secondary_y": True}]])                                    
                    fig.add_trace(go.Candlestick(x=prices.date, open=prices.open, high=prices.high, 
                                  low=prices.low, close=prices.close, name = 'Price'), secondary_y=False)                                
                    fig.add_trace(go.Bar(x= prices.date, y= prices.volume, name='Volume', orientation = "v", 
                                  marker=dict(color="Blue", opacity=1)), secondary_y=True)                    
                    
                    title= {'text': f"Code: {str(code).rjust(4,'0')}", 'y':0.9, 'x':0.5, 'xanchor': 'center', 'yanchor': 'top'}
                    shapes = [dict(x0=st, x1=ed, y0=0, y1=1, xref='x', yref='paper', line_width=2)]
                    fig.update_layout(title=title, showlegend=False, shapes=shapes)                                        
                    fig.update_yaxes(title_text="<b>Price(RM)</b>", secondary_y=False)
                    fig.update_yaxes(title_text="<b>Volume('00)'</b>", secondary_y=True)
                    fig.show()
                    
                return prices
            else:
                print(f'Invalid input: {code}. \nPlease select from {list(prices.code.unique())}')
                            
    def collect_commodities(self):        
        prices = {}
        self.commodities = {
         'copper': {'curr_id': 959211, 'smlID': 300012,  'header': 'Copper Futures Historical Data'},
         'lumber': {'curr_id': 959198,  'smlID': 300958,  'header': 'Lumber Futures Historical Data'},
         'gold': {'curr_id': 8830,  'smlID': 300004,  'header': 'Gold Futures Historical Data'},
         'steel': {'curr_id': 961730,  'smlID': 301021,  'header': 'US Midwest Domestic Hot-Rolled Coil Steel Futures Historical Data'},
         'crude oil wti': {'curr_id': 8849,  'smlID': 300060,  'header': 'Crude Oil WTI Futures Historical Data'},
         'crude oil brent': {'curr_id': 8833,  'smlID': 300028,  'header': 'Brent Oil Futures Historical Data'},
         'palm oil': {'curr_id': 49775,  'smlID': 300630,  'header': 'Crude Palm Oil Futures Historical Data'},
         'corn': {'curr_id': 8918,  'smlID': 300196,  'header': 'US Corn Futures Historical Data'},
         'sugar': {'curr_id': 8869,  'smlID': 300100,  'header': 'US Sugar #11 Futures Historical Data'},
         'lldpe': {'curr_id': 961744,  'smlID': 301189,  'header': 'Linear Low Density Polyethylene Futures Historical Data'},
         'usdmyr': {'curr_id': 2168,  'smlID': 107307,  'header': 'USD/MYR Historical Data'},
         'iron ore': {'curr_id': 961741,  'smlID': 301009,  'header': 'Iron ore fines 62% Fe CFR Futures Historical Data'},
         'coal': {'curr_id': 961733,  'smlID': 301057,  'header': 'Coal Futures Historical Data'},
         'coke': {'curr_id': 961742,  'smlID': 301165,  'header': 'Metallurgical Coke Futures Historical Data'}}
                
        for commodity, ids in tqdm(self.commodities.items()): prices[commodity] = self.investing(**ids)
        is_first = os.path.isfile('fearix.xlsx')                            
        prices['fearix'] = self.treasury_yield(is_first)
        
        for commodity in ['chicken', 'plywood']: prices[commodity] = self.index_mundi(commodity)
        
        os.chdir(self.path)
        storage = 'prices/' # folder for financial statement
        if os.path.isdir(storage) == False:  
            logging.info(f'Folder {storage} not found. Created a new one.')
            os.mkdir(storage)        
        os.chdir(storage)
                            
        for commodity, price_df in tqdm(prices.items()):            
            commodity = commodity.replace('/', '')
            price_df['date'] = price_df['date'].dt.date
            file = commodity + '.xlsx'            
            dfs = [price_df]
            if os.path.exists(file):                                        
                dfs.append(pd.read_excel(file))
            dfs = pd.concat(dfs, axis=0)
            dfs.drop_duplicates('date', inplace=True)
            dfs.to_excel(file, index=False)                      
                            
    def setup(self, ticker=True, metadata=True, statement_q=True, statement_a=True, share_price=True, commodity=True):
        if ticker:
            print('Collecting Tickers...')
            self.collect_ticker(local=False)
        if metadata:
            print('Collecting Metadata...')
            self.collect_metadata(local=False)
        if statement_q:
            print('Collecting Financial Statements(Quarter)...')
            self.collect_statements(codes='all', frequency='quarter')        
        if statement_a:
            print('Collecting Financial Statements(Annual)...')
            self.collect_statements(codes='all', frequency='annual')        
        if share_price:
            print('Collecting Historical Share Prices...')                            
            self.get_price()
        if commodity:
            print('Collecting Prices of commodities and Indicators...')
            self.collect_commodities()
            
        print('Setup Done.')                     
        
    def update(self):
        pass        
    
bursa = BursaScraper()    