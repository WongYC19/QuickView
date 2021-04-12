# Last Modified Date: 8 Mar 2021
from scipy.special import expit
import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from itertools import product
from datetime import datetime
from bs4 import BeautifulSoup
import requests
from logging import warning, info
from functools import wraps
from helper import iterate, DataFile
import os
import sys
import warnings
import logging
from tqdm.auto import tqdm
sys.path.insert(1, os.getcwd())
sys.path.insert(1, "C:/Users/ycwong/Desktop/Bursa Quickview/code")

#from collections.abc import Mapping


pd.set_option('display.max_columns', 150)


class Bursa(DataFile):

    root_url = "https://www.bursamarketplace.com/index.php"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 6.3; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/83.0.4103.116 Safari/537.36"}
    ignore_rows = 0
    fintypes = ["INC", "BAL", "CAS"]
    freqs = ["quarter", "annual"]

    def __init__(self, folder=None):
        if folder is None:
            folder = os.getcwd()

        super().__init__(folder)
        print(f"Bursa current working directory: {folder}")
        self.folder = folder
        self.iterate = iterate

    def get_headers(self, soup):
        headers_div = soup.find_all("div", attrs={"class": "tb_row tb_label"})
        years_div = headers_div[0].find_all("div", attrs={"class": "tb_cell"})
        years = [div.text for div in years_div]
        clean_years = [""]
        self.ignore_rows = len(headers_div)
        if len(headers_div) == 1:  # if year is detected
            return years

        dates_div = headers_div[1].find_all("div", attrs={"class": "tb_cell"})
        dates = [div.text for div in dates_div]
        ref = years[0]

        for year in years[1:]:
            if year == u"\xa0":
                year = ref
            ref = year
            clean_years.append(" " + year)

        headers = [date + year for date, year in zip(dates, clean_years)]
        return headers

    def cleaner(self, element):
        text = element.text.replace(u"\xa0", u"").replace(
            "amp;", "").replace(",", "")
        return None if text == "-" else text

    def get_statements(self, ticker_codes, fintype=None, freq="quarter", workers=5):
        """
            Parameters:
            -----------
            ticker_codes (dict): ticker (str): codes (str) ("SEVE": "5250")
            fintype (list of str): Accepted value: INC, BAL, CAS, default: None (All of them)
            freq (str): Accepted value = "quarter" or "annual"
            worker (int): Maximum number of requests made concurrently

            Return dictionary of dataframe (columns=Period, index=Components, values=data)
        """

        freq = freq.strip().lower()

        if fintype is None:
            fintype = self.fintypes
        else:
            fintype = self.iterate(fintype)
            fintype = [str(ftype).strip().upper() for ftype in fintype]

        for ftype in fintype:
            assert ftype in self.fintypes, f"Only accepts {', '.join(self.fintypes)} as argument for fintype"

        assert freq in self.freqs, "Only accepts quarter or annual as argument for freq"

        def add_kl(code): return code.replace("$", "") + \
            ".KL" if not code.endswith(".KL") else code
        clean_ticker_codes = {add_kl(ticker.strip().upper()): str(code).strip()
                              for ticker, code in ticker_codes.items()
                              if str(code).strip().isdigit()}

        def parse_statement(code, fintype, freq):
            types = {"INC": "is", "BAL": "bs", "CAS": "cf"}[fintype] + freq
            params = {
                "tpl": "financial_ajax",
                "fintype": fintype,
                "type": "stockfin" + types,
                "code": code
            }

            self.resp = requests.get(
                self.root_url, headers=self.headers, params=params)
            self.request_url = self.resp.url
            self.soup = BeautifulSoup(self.resp.text)
            self.columns = self.get_headers(self.soup)
            self.num_col = len(self.columns)

            # Reformat data from single list to tabular format (row-wise)
            cells_div = self.soup.find_all(
                "div", attrs={"class": "tb_cell"})  # collect all cells data
            cells = [self.cleaner(div) for div in cells_div]

            data_lists = [cells[i*self.num_col: (i+1)*self.num_col]
                          for i in range(0, len(cells) // self.num_col)]

            # Parse data (remove non-data rows, set component as index, convert data to numeric, convert column to date)
            statement = pd.DataFrame(data_lists, columns=self.columns)
            statement = (statement.iloc[self.ignore_rows:]
                         .set_index(statement.columns[0])
                         .astype(float))
            statement.columns = pd.to_datetime(
                statement.columns, format="%d %b %Y")
            self.statement = statement
            return statement

        self.statements = dict()

        with ThreadPoolExecutor(max_workers=workers) as executor:
            inputs = tqdm(list(product(clean_ticker_codes.keys(),
                                       fintype)), desc=f"Extract {freq} statement")
            for ticker, fintype in inputs:
                try:
                    stock_digit = clean_ticker_codes[ticker]
                    self.statements[(stock_digit, fintype)] = executor.submit(
                        parse_statement, ticker, fintype, freq).result()
                except Exception as e:
                    print(
                        f"Code: {ticker}, Fintype: {fintype}, freq: {freq} Exception {e}")

        return self.statements

    def stock_tickers(self, workers=5):

        def parse_ticker(page_num):
            params = {
                "screentype": "stocks",
                "board": "all",
                "tpl": "screener_ajax",
                "type": "getResult",
                "action": "listing",
                "sfield": "name",
                "stype": "desc",
                "pagenum": page_num,
            }

            resp = requests.get(
                self.root_url, params=params, headers=self.headers)

            if resp.status_code != 200:
                warning(
                    f'Request failure, url: {resp.url}, status code: {resp.status_code}')

            json_resp = resp.json()
            parse_ticker.total_page = int(json_resp['totalpage'])

            try:
                df = pd.DataFrame.from_dict(
                    json_resp['records'], orient='index')
            except:
                df = pd.DataFrame(json_resp['records'])
            return df

        df_list = []
        parse_ticker(1)
        self.total_page = parse_ticker.total_page  # get total pages

        with ThreadPoolExecutor(max_workers=workers) as executor:
            page_nums = tqdm(range(1, self.total_page+1),
                             desc="Extract tickers")
            future_to_url = [executor.submit(
                parse_ticker, page_num) for page_num in page_nums]
            df_list = [f.result() for f in future_to_url]

        self.df_list = df_list
        self.tickers = pd.concat(
            df_list, axis=0, ignore_index=True, sort=False)

        try:
            numeric_cols = ['avgrating', 'price',
                            'mktcap', 'pev', 'pbv', 'estpeg', 'divyld']
            self.tickers[numeric_cols] = self.tickers[numeric_cols].astype(
                float)
        except:
            pass

        return self.tickers

    def stock_metadata(self, inputs_df):
        """
            Parameters:
            -----------
            inputs_df (Pandas DataFrame): 
                Columns: code, type
                Index: int range
                Values: code - Company ticker, 4 letter ended with .KL (e.g: SEVE.KL)                                    
               c         type - Company type, accepted values: stock, reit, etf 

            Return: Pandas DataFrame, Metadata of each company codes with extra details
        """
        def parse_metadata(type_, code):
            self.params = {"type": "gettixdetail",
                           "code": code, "tpl": type_ + "_ajax"}
            meta_df = None

            resp = requests.get(
                self.root_url, headers=self.headers, params=self.params)
            resp_json = resp.json()

            if 'stockcode' in resp_json:
                meta_df = pd.json_normalize(resp_json, sep="_")

            if meta_df is None:
                print(f"Fail to extract detailed metadata for ticker:{code}")
#             https://www.bursamarketplace.com/index.php?tpl=company_ajax&type=profile_desc&code=ARNK.KL

            self.params = {"tpl": "company_ajax",
                           "type": "profile_address", "code": code + ".KL"}
            resp = requests.get(
                self.root_url, headers=self.headers, params=self.params)
            text = resp.text

            info = text.split('<br>', maxsplit=1)
            address = info[1].replace('<br>', '') if len(info) > 1 else "NaN"
            a_tag = BeautifulSoup(text).find("a")
            hyperlink = a_tag.text if a_tag is not None else "NaN"

            self.params["type"] = "profile_desc"
            resp = requests.get(
                self.root_url, headers=self.headers, params=self.params)
            description = resp.text

            meta_df = meta_df.assign(
                **{'address': address, 'hyperlink': hyperlink, 'description': description})
            return meta_df

        inputs_df['code'] = inputs_df['code'].str.replace(
            r"\.KL$", "", regex=True)
        inputs_df = inputs_df.apply(lambda x: x.str.strip())
        inputs = inputs_df.itertuples(index=False, name=None)
        meta_list = []

        for type_, code in tqdm(inputs, desc="Extract Company Metadata", total=inputs_df.shape[0]):
            try:
                meta = parse_metadata(type_, code)
            except Exception as e:
                warning(f"Exception {e} at code {code}")
            else:
                meta_list.append(meta)

        if len(meta_list):
            metadata = pd.concat(meta_list, sort=False,
                                 axis=0, ignore_index=True)
            return metadata

    def write_metadata(self, tickers, metadata, filename):
        metadata['stockcode'] = tickers['cashtag']
        metadata = metadata.drop('name', axis=1)
        tickers['name'] = tickers['name'].replace(r'\\', '', regex=True)
        full_metadata = tickers.merge(
            metadata, how='left', left_on='cashtag', right_on='stockcode')
        full_metadata.to_excel(filename, index=False)
        print(f"Successfully exported {filename}")
        return full_metadata


if __name__ == '__main__':
    FOLDER = "C:/Users/ycwong/Desktop/Bursa Quickview/data"
    BURSA_METADATA_FILE = os.path.join(FOLDER, 'Bursa Metadata.xlsx')
    QUARTER_STATEMENT_FILE = "quarter_statements.pkl"
    ANNUAL_STATEMENT_FILE = "annual_statements.pkl"

    print("1. Collect tickers and its details attributes from Bursa Marketplace")
    bursa = Bursa(FOLDER)
    bursa_tickers = bursa.stock_tickers()
    inputs_df = bursa_tickers['link'].str.extract(
        r"/mkt/themarket/(\w+)/(\w+)")
    inputs_df.columns = ['type', 'code']
    bursa_metadata = bursa.stock_metadata(inputs_df)
    bursa_df = bursa.write_metadata(
        bursa_tickers, bursa_metadata, BURSA_METADATA_FILE)

    print("2. Collect Financial Statements from Bursa Marketplace")
    bursa_df = pd.read_excel(BURSA_METADATA_FILE)
    ticker_codes = bursa_metadata.set_index(
        'stockcode')['bursacode'].dropna().to_dict()
    quarter_statement = bursa.get_statements(ticker_codes, freq='quarter')
    annual_statement = bursa.get_statements(ticker_codes, freq='annual')

    print("3. Update Financial Statement to existing data file")
    old_quarter_statement = bursa.read_pickle(QUARTER_STATEMENT_FILE)
    updated_qr_stat = bursa.update(
        old_quarter_statement, quarter_statement, axis=1)
    bursa.write_pickle(updated_qr_stat, QUARTER_STATEMENT_FILE)

    old_annual_statement = bursa.read_pickle(ANNUAL_STATEMENT_FILE)
    updated_an_stat = bursa.update(
        old_annual_statement, annual_statement, axis=1)
    bursa.write_pickle(updated_an_stat, ANNUAL_STATEMENT_FILE)
