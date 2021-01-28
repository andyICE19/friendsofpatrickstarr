# -*- coding: utf-8 -*-
"""
Text preprocessing utilities

Created on Fri Oct  9 10:37:59 2020

@author: Tapesh
"""

import re
import os
import pandas as pd

def preprocess ( text ):
    '''
    performs the following processing steps
    
    1. Removes non alpha numeric characeres using regular expressions
    2. Converts to lower case
    3. split the string into a list of strings

    Parameters
    ----------
    text : string
        string that needs to be processes

    Returns
    -------
    string
        processed string.

    '''
    
    #return re.sub(r'\W+', ' ', str(text)).lower().split()
    return re.sub('[^a-zA-Z0-9\n\.]', ' ', str(text)).lower().split()


def gold_standard_test_data(data_dir):
    
    files = os.listdir(data_dir)
    
    all_headers = {}
    
    for f in files:        
        
        df = pd.read_csv(data_dir + f , header = 0 , sep = '\t')
        
        all_headers.update(df.iloc [0,:].apply(lambda x: x.replace('CDE_ID:','')).to_dict())
    
    return all_headers
        
        