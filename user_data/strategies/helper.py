import pandas as pd 

class helper(object):
    @staticmethod
    def save_dataframe(dataframe, file_name:str='dataframe.csv'):
        #print(f'filename = {file_name}')
        dataframe.to_csv(f'/freqtrade/user_data/dataframe_files/{file_name}')