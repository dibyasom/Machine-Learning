#Importing dependencies
import pandas as pd
import os

class linear_regression:

    def __init__(self, path):
        self.__path = path
        self.__dataset = None

    def fetch_dataset(self):
        try:
            if os.path.exists(self.__path):
                self.__dataset = pd.read_csv(self.__path)
        except OSError:
            print("Dataset not found :(")
            exit()
    
    def dataset_info(self):
        print(self.__dataset)
        print(self.__dataset.info())
        print(self.__dataset.isnull().sum())
    
    def preprocess_dataset(self):
        

    def engage_engine(self):
        self.fetch_dataset()
        self.dataset_info()

if __name__=='__main__':
    path = './Resources/titanic_data.csv'
    linear_regression(path).engage_engine()