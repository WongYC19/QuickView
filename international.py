# Last modified date: 12 Feb 2021
from helper import iterate
import os
import sys
import requests
import pandas as pd
from bs4 import BeautifulSoup

sys.path.insert(1, os.getcwd())


class International:

    inv_url = 'https://www.investing.com/instruments/HistoricalDataAjax'
    index_mundi_url = 'https://www.indexmundi.com/commodities'
    treasury_url = 'https://www.treasury.gov/resource-center/data-chart-center/interest-rates/pages/TextView.aspx'
    macro_url = 'https://www.macrotrends.net/'

    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 6.3; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/78.0.3904.108 Safari/537.36',
        'X-Requested-With': 'XMLHttpRequest',
    }

    inv_commodities = {
        'copper': {'curr_id': 8831, 'smlID': 300012,  'header': 'Copper Futures Historical Data'},
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
        'coke': {'curr_id': 961742,  'smlID': 301165,  'header': 'Metallurgical Coke Futures Historical Data'},
        'wheat': {"curr_id": 8917, "smlID": 300188, "header": "US Wheat Futures Historical Data"},
    }

    inv_date_format = {'Daily': '%b %d, %Y',
                       'Weekly': '%b %d, %Y', 'Monthly': '%b %y'}

    def __init__(self):
        self.iterate = iterate

    def investing(self, commodities=None, ascending=False, frequency='Daily', start_date=None, end_date=None):
        """
            commodity (str): refer to the key in inv_commodities
            ascending (bool): sort the order by Date, default = True
            frequency (str): the interval of date index, accepted value = 'Daily', 'Weekly', 'Monthly'
            start_date (str): "mm/dd/yyyy" format
            end_date (str): "mm/dd/yyyy" format
        """

        def parse_investing(payload, date_format):
            response = requests.post(
                self.inv_url, data=payload, headers=self.headers)
            soup = BeautifulSoup(response.text, features="html.parser")

            table = soup.find(
                attrs={'class': "genTbl closedTbl historicalTbl"})
            cells = table.find_all({'td': 'data-real-value'})
            cells = [cell.text for cell in cells]
            columns = table.find(name='tr').findAll('th')
            columns = [c.text.lower() for c in columns]

            starts = range(0, len(cells), len(columns))
            ends = range(len(columns), len(cells)+1, len(columns))

            df = pd.DataFrame(
                (cells[s:e] for s, e in zip(starts, ends)), columns=columns)
            df['date'] = pd.to_datetime(df['date'], format=date_format)
            today = pd.to_datetime('now')
            df = df[df['date'] != pd.Timestamp(
                today.year, today.month, today.day)]

            if 'vol.' in df.columns:
                df.rename(columns={'vol.': 'volume'}, inplace=True)
                multiplier = (df['volume'].str.extract(r'([A-Za-z]$)')
                              .replace({'K': 10e2, 'M': 10e5, 'B': 10e8})
                              .fillna(1)
                              )
                df['volume'] = (df['volume']
                                .replace(r'K|M|B', '', regex=True)
                                .replace(r'-$', 'NaN', regex=True)
                                .astype(float) * multiplier.T).T

            df['change %'] = df['change %'].str.replace('%', '')
            df.loc[:, df.columns != 'date'] = df.loc[:,
                                                     df.columns != 'date'].replace(',', '', regex=True)
            df.loc[:, df.columns != 'date'] = df.loc[:,
                                                     df.columns != 'date'].applymap(float)
            return df

        if commodities is None:
            commodities = self.inv_commodities.keys()
        commodities = self.iterate(commodities)
        commodities = [
            c for c in commodities if c in self.inv_commodities.keys()]
        frequency = str(frequency).capitalize()

        if frequency not in self.inv_date_format.keys():
            raise AssertionError(
                "Only 'Daily', 'Weekly', 'Monthly' allowed for frequency argument")

        date_format = self.inv_date_format[frequency]

        if start_date is None:
            start_date = datetime.strptime(
                "01/01/1950", '%m/%d/%Y').strftime('%m/%d/%Y')
        else:
            start_date = datetime.strptime(
                start_date, "%m/%d/%Y").strftime('%m/%d/%Y')

        if end_date is None:
            end_date = datetime.today().strftime('%m/%d/%Y')
        else:
            end_date = datetime.strptime(
                end_date, "%m/%d/%Y").strftime('%m/%d/%Y')

        payload = {
            "sort_col": "date",
            "action": "historical_data",
            "sort_ord": 'ASC' if ascending else 'DESC',
            "interval_sec": frequency,
            "st_date": start_date,
            "ed_date": end_date,
        }

        prices = dict()

        for commodity in tqdm(commodities, desc="Commodity price (Investing)"):
            commodity = commodity.lower().strip()
            payload.update(self.inv_commodities[commodity])
            prices[commodity] = parse_investing(payload, date_format)

        return prices

    def index_mundi(self, commodities):
        """ Accepted input for commodity (str, iterable): chicken, plywood and etc"""

        def parse_mundi(params):
            response = requests.get(self.index_mundi_url, params=params)
            soup = BeautifulSoup(response.text, features="html.parser")
            table = soup.find(attrs={'class': 'tblData'}).findAll('tr')
            data = pd.DataFrame((row.findAll('td') for row in table[1:]), columns=[
                                'date', 'price', 'change(%)'])
            data = data.applymap(lambda x: x.text)
            data = data.replace(r' %', '', regex=True).replace(
                r'-$', '0.00', regex=True)
            data['date'] = pd.to_datetime(data['date'], format='%b %Y')
            data.loc[:, data.columns != 'date'] = data.loc[:,
                                                           data.columns != 'date'].applymap(float)
            return data

        commodities = self.iterate(commodities)
        prices = dict()

        for commodity in tqdm(commodities, "Commodity Price (Index Mundi)"):
            params = {"commodity": commodity, "months": 240, "currency": "myr"}
            prices[commodity] = parse_mundi(params)

        return prices

    def treasury_yield(self, year='all'):
        """
            year = all -> Get yield for full range
            year = None ->  Get yield for recent year
            year = "dddd" -> Get the yield for specified year
        """

        if str(year).lower() == 'all':
            params = {"data": "yieldAll"}
        else:
            if year is None:
                datetime.today().year
            params = {"data": "yieldYear", "year": year}

        response = requests.get(self.treasury_url, params=params)
        soup = BeautifulSoup(response.text, features='html.parser')

        dataset = soup.find(attrs={'class': 't-chart'})
        headers = dataset.findAll(attrs={'scope': "col"})
        headers = [header.text.lower() for header in headers]

        indices = dataset.findAll(attrs={'scope': 'row'})
        indices = [index.text for index in indices]

        table = dataset.findAll(attrs={'class': "text_view_data"})

        n_cells = len(table)
        n_rows = len(indices)
        n_cols = int(n_cells/n_rows)

        df = pd.DataFrame([table[i*n_cols:(i+1)*n_cols]
                           for i in range(n_rows)], columns=headers)
        df = df.applymap(lambda x: x.text.strip()).replace(
            'N/A', 'NaN', regex=True)
        df['date'] = pd.to_datetime(df['date'], format='%m/%d/%y')
        df.loc[:, df.columns != 'date'] = df.loc[:,
                                                 df.columns != 'date'].applymap(float)
        return df

    def get_gnp(self):
        url = self.macro_url + "countries/MYS/malaysia/gnp-gross-national-product"
        response = requests.get(url)
        soup = BeautifulSoup(response.text, features='lxml')
        table = soup.find_all('table', attrs={
                              'class': 'historical_data_table table table-striped table-bordered'})
        table = table[1].find_all('tr')
        gnp = pd.DataFrame([row.text.split('\n') for row in table[3:]])
        gnp.columns = table[1].text.split('\n')
        gnp.drop('', axis=1, inplace=True)
        gnp.GNP = gnp.GNP.str.replace('\$|B', '').astype(float)
        gnp['Per Capita'] = gnp['Per Capita'].str.replace(
            '\$|,', '').astype(float)
        gnp['Growth Rate'] = gnp['Growth Rate'].str.replace(
            '\%', '').astype(float)
        gnp.set_index('Year', drop=True, inplace=True)
        gnp.index = gnp.index.astype('int')
        return gnp


if __name__ == '__main__':
    FOLDER = "C:/Users/ycwong/Desktop/Bursa Quickview/code/"
    # codes = pd.read_excel(FOLDER + "commo.xlsx")

    proxy = International()
#     investing = proxy.investing()
#     index_mundi = proxy.index_mundi()
#     treasure = proxy.treasury_yield()

    gnp = proxy.get_gnp()
    print(gnp)
