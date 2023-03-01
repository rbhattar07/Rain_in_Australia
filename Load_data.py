import opendatasets as od
import pandas as pd
import numpy as np
import os
import kaggle
from kaggle.api.kaggle_api_extended import KaggleApi
from urllib.request import urlretrieve
import zipfile

dataset_url = 'jsphyg/weather-dataset-rattle-package'

api = KaggleApi()
api.authenticate()

api.dataset_download_file(dataset_url,file_name='weatherAUS.csv')

with zipfile.ZipFile('weatherAUS.csv.zip', 'r') as zipref:
    zipref.extractall('./')