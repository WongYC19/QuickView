# Last Modified Date: 20 Feb 2021
from helper import iterate, DataFile
import os
import sys
import pickle
import warnings
import requests
from bs4 import BeautifulSoup
import pandas as pd
from tqdm.auto import tqdm
from datetime import datetime

sys.path.insert(1, os.getcwd())


def check_adjusted_price(old_price, prices):
    """
        Detect the stock with adjusted share price (bonus issue, dividend and etc)
    """
    adjusted_price = {}
    codes = set(old_price.keys()) & set(prices.keys())
    for code in codes:
        merged = prices[code].merge(old_price[code], how='inner', on='date')
        diff = (merged['close_x'] - merged['close_y']).mean()
        if diff > 0:
            adjusted_price[code] = merged

    if adjusted_price:
        print(f"Alert: Adjusted share price: {adjusted_price.keys()}")

    # Filter out
    if adjusted_price.keys():
        new_prices = {c: p for c, p in prices.items(
        ) if code not in adjusted_price.keys()}
    else:
        new_prices = prices

    return new_prices


class KLSE(DataFile):

    root_url = 'https://www.klsescreener.com/v2/'

    def __init__(self, folder=None):
        super().__init__(folder)
        self.iterate = iterate

    @staticmethod
    def clean_code(codes):
        return list(dict.fromkeys(str(code).rjust(4, "0").strip() for code in codes))

    def stock_codes(self, stock_only=True):
        """
            Retrieve full list of stock codes available from KLSE screener

            Parameters:
            -----------
            stock_only (bool): Remove warrants if True, default=True

            Return: Pandas Series (index=Bursa Stock Codes in digit, data=Stock Name)
        """
        header = {
            'user-agent': 'Mozilla/5.0 (Linux; Android 6.0; Nexus 5 Build/MRA58N) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/88.0.4324.150 Mobile Safari/537.36'}
        response = requests.get(
            self.root_url + 'screener/quote_results', headers=header)
        soup = BeautifulSoup(response.content, features='html.parser')
        codes = soup.findAll('td', attrs={'title': 'Code'})
        codes = [code.text for code in codes]
        names = soup.findAll('a', attrs={'target': '_blank'})
        names = [name.text for name in names if name.text != '>>']
        series = pd.Series(data=names, index=codes)
        if stock_only:
            series = series[~series.str.contains("-")]
        return series

    def share_price(self, codes):
        """
            Retrieve historical share price (up to 10 Years) for list of stock codes 

            Parameters:
            ----------
            codes (list of str): Bursa Stock Code 

            Return dictionary: Bursa Stock Code: Pandas DataFrame (stock price)
        """

        def parse_price(url, code):
            response = requests.get(url + code)
            if response.status_code != 200:
                print(
                    f'Could not retrieve share price for {code}. Kindly verify at {response.url}')
                return None

            try:
                text = response.text
                start_mark = 'data = ['
                start = text.find(start_mark)
                text = text[start+len(start_mark)-1:]
                end = text.find(';')
                text = text[:end].replace('\n', '').replace('\t', '')

            except Exception as e:
                print(
                    f'\n{datetime.now()} --- Unable to extract historical price of {code}. Message: {e}')

            else:
                try:
                    data = pd.DataFrame(eval(text), columns=[
                                        'date', 'open', 'high', 'low', 'close', 'volume'])
                except Exception as e:
                    print(text)
                else:
                    data['date'] = pd.to_datetime(data['date'], unit='ms')
                    data['code'] = code

                    null = data.isna().sum(axis=1)

                    if null.sum() > 0:
                        dates = list(null[null > 0].index)
                        warning(
                            f'Null value detected in {code}, date: {dates}')

                    return data

        url = self.root_url + "stocks/chart/"
        codes = self.iterate(codes)
        codes = self.clean_code(codes)
        self.prices = dict()

        for code in tqdm(codes, desc="Extract share price"):
            price = parse_price(url, code)
            if price is not None:
                self.prices[code] = price

        return self.prices

    def share_price_new(self, codes, start_date=None, end_date=None):
        """
            Retrieve historical share price from trading view API via KLSE screener
            based on list of stock code for list of stock codes 

            Parameters:
            ----------
            codes (list of str): 4-digits Bursa Stock Code (Example: 1295)
            start_date (str): Earliest date in extraction result (YYYY-MM-DD)
            end_date (str): Latest date in extraction result (YYYY-MM-DD)

            Return dictionary: Bursa Stock Code: Pandas DataFrame (stock price)            
        """
        def parse_price(code, start_date, end_date):
            url = f"{self.root_url}trading_view/history"
            renamer = {'o': 'open', 'h': 'high', 'l': 'low',
                       'c': 'close', 'v': 'volume', 't': 'date'}
            code = str(code).strip()
            params = {"symbol": code, "resolution": "1D",
                      "from": start_date, "to": end_date}
            header = {
                'user-agent': 'Mozilla/5.0 (Linux; Android 6.0; Nexus 5 Build/MRA58N) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/88.0.4324.150 Mobile Safari/537.36'}
            resp = requests.get(url, params=params, headers=header)

            try:
                price_df = pd.DataFrame(resp.json())
                price_df['t'] = pd.to_datetime(price_df['t'], unit='s')
                price_df[['o', 'h', 'l', 'c']] = price_df[[
                    'o', 'h', 'l', 'c']].astype(float)
                price_df = price_df.rename(columns=renamer)[renamer.values()]
                price_df['code'] = code

            except Exception as e:
                print(f"Exception {e} occurred at code {code}")

            else:
                return price_df

        initial_date = datetime(year=1970, month=1, day=1).date()

        if start_date is None:
            start_date = 0
        if end_date is None:
            end_date = int((datetime.today().date() -
                            initial_date).total_seconds())
        else:
            end_date = datetime.strptime(end_date, "%Y-%m-%d")
            end_date = int((end_date - initial_date).total_seconds())

        codes = [str(code) for code in codes]
        price_dict = {}

        for code in tqdm(codes, desc="Collecting daily share price"):
            price = parse_price(code, start_date, end_date)
            if price is not None:
                price_dict[code] = price

        return price_dict

    def stock_summary(self, codes, stock_only=True, workers=6):
        def parse_summary(code):
            resp = requests.get(
                self.root_url + "stocks/view/" + code + ".json")
            js = resp.json()
            basic = pd.json_normalize(js, sep="_")
            return basic

        codes = self.iterate(codes)
        codes = self.clean_code(codes)
        summaries = []

        with ThreadPoolExecutor(max_workers=workers) as executor:
            future_summary = [executor.submit(
                parse_summary, code) for code in codes]
            for f_sum in tqdm(future_summary, desc="Extract stock details"):
                try:
                    summary = f_sum.result()
                except Exception as e:
                    warning(f"Exception occurred at code {code}: {e}")
                else:
                    summaries.append(summary)

        if summaries:
            summary_df = pd.concat(
                summaries, axis=0, sort=False, ignore_index=True)
            summary_df.columns = summary_df.columns.str.lower()
            self.summary = summary_df

            if stock_only:
                summary_df = summary_df[~summary_df['sector_name'].isin(
                    ['BOND ISLAMIC', None])]

            return summary_df


FOLDER = "C:/Users/ycwong/Desktop/Bursa Quickview/data"
PRICE_DATA_FILE = "prices.pkl"

klse = KLSE(FOLDER)
klse_codes = klse.stock_codes()
prices = klse.share_price_new(klse_codes.index)
old_prices = klse.read_pickle(PRICE_DATA_FILE)
# new_prices = check_adjusted_price(old_prices, prices)
updated_prices = klse.update(old_prices, prices, axis=0, subset=['date'])
klse.write_pickle(updated_prices, PRICE_DATA_FILE)
print('Done')
