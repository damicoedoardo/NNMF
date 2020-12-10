'''
Created on Thu Sep 26 2019

@author XXX
'''

import datetime
import os

class WriteTextualFile:

    def __init__(self, folder_path, name, append_datetime=False, ):
        complete_path = os.path.join(folder_path, name)
        if append_datetime:
            complete_path += '_{}'.format(datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S"))
        self.complete_path = complete_path + '.txt'
        os.makedirs(folder_path, exist_ok=True)

    def write_line(self, text):
        with open(self.complete_path, 'a') as _file:
            print(text, file=_file)
