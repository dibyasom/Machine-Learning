# Import dependcies
import numpy as np
import pandas as pd
import os

# Fetch a file


class genDataset:
    FILE_LOC=''
    file_name=''
    dataset=[]

    def __init__(self, fileName):
        self.FILE_LOC = "../data/"
        self.file_name = fileName

    def fetch_file(self):
        try:
            if not os.path.exists('../data'):
                os.makedirs('../data')
        except OSError:
            print('Plis, place me in the right directory idiot.')
            exit()
        self.dataset = pd.dataframe()
            
    def gen_add_dataset(self):
        self.dataset['X1'] = np.random.randint(low=0, high=999999999, size=100)
        self.dataset['X2'] = np.random.randint(low=0, size=100, dtype='np.int')
        self.dataset['Y']  = self.dataset['X1']+self.dataset['X2']
        self.dataset.to_csv("../data/add-dataset.csv")

#Driver
data = genDataset('addition.csv')
data.fetch_file()
data.gen_add_dataset()