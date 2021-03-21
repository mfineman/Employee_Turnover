import pandas as pd
import numpy as np
from config import password
from bs4 import BeautifulSoup as bs
from splinter import Browser
from sqlalchemy import create_engine
import psycopg2


# import os
# from selenium import webdriver
# from selenium.webdriver.chrome.options import Options

url = 'https://www.bls.gov/news.release/archives/jolts_03172020.htm' 


def scrape_bls():
    executable_path = {'executable_path': 'chromedriver.exe'}
    browser = Browser('chrome', **executable_path, headless=True)

    bls_table = pd.read_html(url)
    bls_df=bls_table[15]

    bls_df = bls_df.replace(np.nan, '', regex=True)

    Engine = create_engine(f"postgresql://postgres:{password}@localhost:5432/Employee_Turnover");
    connection = Engine.connect();
    postgreSQLTable = "blsdata"

    try:
        frame = bls_df.to_sql(postgreSQLTable, connection, if_exists='fail');

    except ValueError as vx:
        print(vx)

    except Exception as ex:  
        print(ex)

    else:
        print("PostgreSQL Table %s has been created successfully."%postgreSQLTable);

    finally:
        connection.close();

