import os
import re
import pickle
import pandas as pd

import smtplib
from email.mime.base import MIMEBase
from email.mime.message import MIMEMessage
from email.mime.text import MIMEText
from email import encoders
from email.mime.multipart import MIMEMultipart
from decimal import Decimal


def iterate(inputs):
    if isinstance(inputs, str):
        if ';' in inputs:
            inputs = inputs.split(';')
        elif ',' in inputs:
            inputs = inputs.split(',')
        else:
            inputs = [inputs]
    else:
        try:
            inputs = iter(inputs)
        except:
            inputs = [str(inputs)]
        else:
            inputs = list(map(str, inputs))

    inputs = list(map(lambda x: str(x).strip(), inputs))
    inputs = list(dict.fromkeys(inputs))
    return inputs


class DataFile:

    def __init__(self, working_dir=None):
        if os.path.isdir(working_dir) == False:
            print(
                f"Invalid path provided. Change to current working directory: {os.getcwd()}")
            working_dir = None

        if working_dir is None:
            working_dir = os.getcwd()

        print("Detected working directory:", working_dir)
        self.working_dir = working_dir

    def read_pickle(self, file_path):
        data = None
        file_path = os.path.join(self.working_dir, file_path)
        if os.path.exists(file_path) and file_path.endswith(".pkl"):
            with open(file_path, "rb") as file:
                data = pickle.load(file)
        else:
            print(f"{file_path} is not a pickle file or doesn't exist.")
        return data

    def write_pickle(self, obj, file_path):
        file_path = os.path.join(self.working_dir, file_path)
        if file_path.endswith(".pkl"):
            with open(file_path, "wb") as file:
                pickle.dump(obj, file)
            print(
                f"The python object is exported to {file_path} successfully.")
        else:
            warning(
                "Invalid file format. Please provide a pickle file with .pkl extension")

    def update(self, old_data, new_data, axis=1, subset=None, verbose=1):
        """
            Update (overwrite and append) data in dictionary.
            old_data and new_data (dict): {"key": Pandas DataFrame Object}             
            axis (bool): 0 or 1, whether to update the data horizontally/vertically
            subset: to remove duplicate when updated horizontally, ignored when axis=1
        """
        if isinstance(old_data, dict) == False:
            raise NotImplementedError(
                "The old_data is not in dictionary type.")
        if isinstance(new_data, dict) == False:
            raise NotImplementedError(
                "The new_data is not in dictionary type.")
        old_data = old_data.copy()
        new_data = new_data.copy()
        self.new_keys = []

        for key, data in new_data.items():
            if key in old_data.keys():
                if axis == 1:
                    df = pd.concat([data, old_data[key]], axis=axis)
                    df = df.loc[:, ~df.columns.duplicated()]

                    if verbose:
                        new_col = set(df.columns) - set(old_data[key].columns)
                        if new_col:
                            print("New period update:", key, ":", new_col)

                else:
                    df = pd.concat([data, old_data[key]], axis=axis)
                    df = df.drop_duplicates(subset=subset, keep='first')

                old_data[key] = df

            else:
                old_data[key] = data
                self.new_keys.append(key)

        if len(self.new_keys):
            print(f"Newly added keys: {self.new_keys}")
        self.updated_data = old_data
        return old_data

    def finscore(self):
        pass


class Notification:

    def __init__(self, TO=None, CC=None, BCC=None):

        self.FROM = "bigquery369@gmail.com"

        if TO is None:
            TO = "ycfkjc@hotmail.com"

        self.TO = TO
        self.message = MIMEMultipart()
        self.message['From'] = self.FROM
        self.message['To'] = TO
        self.message['Cc'] = CC
        self.message["Bcc"] = BCC

    @staticmethod
    def rounding(x):
        rounder = '1.11' if float(abs(x)) > 1 else '1.111'
        return float(Decimal(x).quantize(Decimal(rounder)))

    def get_style(self):
        return """<style> table {
            font-family: 'Poppins', sans-serif;
            border-collapse: collapse;
            border: 1px solid white;
            margin: 1em 0;
            width: 100%;
            overflow: hidden;
            background-color: #FFF;
            color: $text-color;
            border-radius: $border-radius;
            border: $outer-border;
        }

        table tr {
            padding: 0.2rem;
            text-align: center;
            font-size: 12px;
            width: 100%;
        }

        table tr:nth-child(even) {
            background-color: #F5F5F5;
        }

        table tr:nth-child(odd) {
            background-color: #EAF3F3;
        }

        table tr th {
            border: 1px solid black;
            font-weight: bold;
            font-size: 14px;
            color: #FFF;            
            padding: 1em;
        }

        .gain {
            background-color: #167F92;
        }

        .loss {
            background-color: lightpink;
        }
        
        </style>
        """.replace("\n", "")

    def create_html(self, results_df, gain=True):
        columns = results_df.columns
        rows = results_df.itertuples(index=False, name=None)

        header_html = ''
        row_html = ''
        for col in columns:
            class_name = "gain" if gain else "loss"
            header_html += f"<th class='{class_name}'>{col}</th>"
        html = f"<tr>{header_html}</tr>"

        for row in rows:
            row_html = ''
            for cell in row:
                row_html += f"<td>{cell}</td>"
            row_html = f"<tr>{row_html}</tr>"
            html += row_html

        html = f"<table>{html}</table>"
        return html

    def send_email(self, SUBJECT, HTML=None, TEXT=None, FILENAMES=None):
        if HTML is not None:
            self.message.attach(MIMEText(HTML, "html"))
        if TEXT is not None:
            self.message.attach(MIMEText(TEXT, "plain"))

        if FILENAMES is None:
            FILENAMES = []

        self.message["Subject"] = SUBJECT
        part = None

        for filename in FILENAMES:
            with open(filename, "rb") as attachment:
                # Add file as application/octet-stream
                # Email client can usually download this automatically as attachment
                part = MIMEBase("application", "octet-stream")
                part.set_payload(attachment.read())

            # Encode file in ASCII characters to send by email
            encoders.encode_base64(part)

            # Add header as key/value pair to attachment part
            part.add_header(
                "Content-Disposition",
                f"attachment; filename= {filename}",
            )

        # Add attachment to message and convert message to string
        if part is not None:
            self.message.attach(part)

        self.TEXT = self.message.as_string()
        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as server:
            try:
                server.login("bigquery369@gmail.com", "abcd!1234")
                server.sendmail(self.FROM, self.TO, self.TEXT)
            except Exception as e:
                print(f"Error {e}: Unable to send email.")


noti = Notification()
