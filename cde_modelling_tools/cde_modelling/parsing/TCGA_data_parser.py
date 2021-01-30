# -*- coding: utf-8 -*-
"""
Imports TCGA data, creates header list, value list and value type
Created on Tue Oct 13 18:15:53 2020

@author: Tapesh
"""

import pandas as pd
from os import listdir, path
import json
import time
from tqdm import tqdm
from sklearn.neighbors import NearestNeighbors
import random
import mlflow
import json
import sys
import multiprocessing
from multiprocessing import Pool
from collections import defaultdict
import re
from os import listdir
from cde_modelling.utils import Preprocessing as prs
import numpy as np





class TCGA_data_processor:
    
    
    '''
    This class processess TCGA data, specifically, reads the data and creates header,value lists and value types.
    !!! Warning: There are hard coded elements in this script, which could not be removed due to lack of time for code cleaning. This is not best practise and should be avoided.
    '''
    
    
    def __init__(self, dirname, gold_standard_flag):
        '''
        Parses TCGA data and create three dictionaries, one containing bag of words in each header, the second containing bag of words in the columns associated with each header
        , the third containing the data type of each column.
        

        Parameters
        ----------
        dirname : string
            Path to the directory where TCGA data is stored
        gold_standard_flag : boolean
            The training data has the gold standard information. When parsing training data set it to true and the parse will create a gold standard dataset.

        Returns
        -------
        None.

        '''
        self.gold_standard_flag = gold_standard_flag
        print(' Processing clinical metadata.. please wait..')
        #Step 1: collect all files from the training file dump
        
        tcga_files_dirs = [path.join( dirname , f) for f in listdir( dirname )]
        
        #index = 0
        
        
        self.all_headers = {}   
        self.headers_list = {}
        
        self.clinical_value_info = pd.DataFrame(columns = ['public_id', 'data_type'] )
        
        self.clinical_value_list = defaultdict(set)
        
        
        # for each clinical data file
        
        for i in tqdm(range(len( tcga_files_dirs ))):
            
            try :
                
                #files=listdir( tcga_files_dirs[i] )
                
                #for f in files:
                file = tcga_files_dirs[i]
                
                # ! warning hard coded element below
                
                if 'nationwide' in file:
                    
                    # read the file into a dataframe
                    
                    #df = pd.read_csv( file , index_col = 0 , header = 0, sep='\t')
                
                    # if parsing training data we need to extract the CDE_IDs for each header, this is done in the following code
                    if gold_standard_flag:
                        df = pd.read_csv( file , index_col = 0 , header = 0, sep='\t')
                        self.all_headers.update( df.loc ['CDE_ID:',:].apply(lambda x: x.replace('CDE_ID:','')).to_dict())
                        
                        # drop the following two rows they are not needed for the rest of the parsing.
                        
                        # drop the rows regarding the CDE_ID of a header
                        df.drop('CDE_ID:',axis = 0, inplace = True)
                        
                        # duplicate row
                        df.drop('bcr_patient_uuid',axis = 0, inplace = True)

                    else:
                        df = pd.read_csv( file , index_col = 0 , header = 0, sep='\t')
                        df = df[~df.iloc[:,0].str.contains('_',case=False)] #drop CDE_ID and CDE header name
                    
                    #print(df.head())
                    
                    # create dict containg headers as key and lower cased, splitted headers as values
                    self.headers_list.update ({c: prs.preprocess(c) for c in df.columns})
                    
                   # for each column header create a list of values in the corresponding column
                    for c in df.columns:
                        vals = df[c].unique()
                        
                        # don't create value list for bcr_patient_barcode column and continuous variable. 
                        # bcr_patient_barcode column has unique value in each row and are barcode numbers which cant be 
                        # semantically defined. Similarly, continuous variables are numbers and can't be semantically defined

                        #---- added by Sher Lynn
                        #noise_values = ['[Not Available]','[Not Applicable]','[Not Evaluated]','[Discrepancy]','[Unknown]']
                        #----------
                        if (c!= 'bcr_patient_barcode') and (len(vals) < 0.9*df.shape[0]): 
                            
                            for v in vals:
                                #if v not in noise_values: #line added by Sher Lynn
                                self.clinical_value_list[c].update(prs.preprocess(v))
                    
                    # identify value type (string or number)
                    
                    value_type = self.get_value_info(df)             
                    
                    #print(value_type)
                    
                    self.clinical_value_info = pd.concat( [self.clinical_value_info , value_type] )
                        
            except :
                
                pass
        
        
        # convert to Dummy variables
        self.clinical_value_info.drop_duplicates(inplace = True) # drop true duplicates
        #drop when data type is unknown
        # section added by Sher Lynn
        self.clinical_value_info['data_type'].replace('', np.nan, inplace = True)
        self.clinical_value_info.dropna(subset = ['data_type'], inplace = True)
        #print(self.clinical_value_info['data_type'])
        #print(self.clinical_value_info['data_type'].value_counts())
        # section end
        #pivot table
        self.clinical_value_info = pd.get_dummies ( self.clinical_value_info, columns = ['data_type']).set_index('public_id')
        self.clinical_value_info = self.clinical_value_info.sort_values(by=['public_id','data_type_boolean','data_type_number','data_type_string'], 
                                                                        ascending=[True,False,False,False])

        self.clinical_value_info = self.clinical_value_info.groupby(self.clinical_value_info.index).first()
        # if self.clinical_value_info.shape[1]==1:
        #     col = self.clinical_value_info.columns[0]
        #     # ---- section modified by Sher Lynn
        #     if 'string' in col:
        #         self.clinical_value_info['data_type_number'] = 0
        #         self.clinical_value_info['data_type_string'] = 1
        #         self.clinical_value_info['data_type_boolean'] = 0
            
        #     #else:
        #     elif 'boolean' in col:
        #         self.clinical_value_info['data_type_string'] = 0
        #         self.clinical_value_info['data_type_number'] = 0
        #         self.clinical_value_info['data_type_boolean'] = 1
            
        #     else:
        #         self.clinical_value_info['data_type_string'] = 0
        #         self.clinical_value_info['data_type_number'] = 1
        #         self.clinical_value_info['data_type_boolean'] = 0

        #     ## -- Section end
            
        
        #self.clinical_value_list= { item[0] : [ prs.preprocess(v) for v in item[1] ] for item in self.clinical_value_list.items() }
              
        # remove empty header if any
        self.all_headers = {key:val for key, val in self.all_headers.items() if val != ''}
        #self.headers_list = {key: prs.preprocess(key) for key, val in self.all_headers.items()}
        
        
    def get_header_list(self):
        
        '''
        Return the header list
    
        Returns
        -------
        TYPE
            DESCRIPTION.
    
        '''
        return self.headers_list
    
    def get_gold_standard(self):
        
        '''
        Returns gold standard
    
        Returns
        -------
        TYPE
            DESCRIPTION.
    
        '''
        return self.all_headers
   
    def get_value_list(self):
        '''
        Returns the value list

        Returns
        -------
        TYPE
            DESCRIPTION.

        '''
        return self.clinical_value_list
    
    def get_value_type(self):
        '''
        Returns the value type (i.e. column data types)

        Returns
        -------
        None.

        '''
        return self.clinical_value_info
   
    
    def get_parsed_data(self):
        '''
        

        Returns
        -------
        dict
            DESCRIPTION.

        '''
        parsed_data ={'headers': self.get_header_list(), 'values': self.get_value_list(), 'value_type': self.get_value_type()}
        
        if self.gold_standard_flag:
            parsed_data['gold_standard'] = self.get_gold_standard()
    
        return parsed_data
        
    def get_value_info( self, df ):
        '''
        checks if the columns of a dataframe are numeric or string

        Parameters
        ----------
        df : DataFrame
           dataframe to be checked for data types.

        Returns
        -------
        headers_data_type: DataFrame
            data type of each column
        '''
        
    
        headers_data_type = pd.DataFrame( columns = ['public_id', 'data_type'] )
        
        headers_data_type['public_id'] = df.columns
        
        headers_data_type.set_index(['public_id'], inplace = True) 
    
        for c in df.columns:
            values = list(df[c].unique())

            ## Section added by Sher Lynn

            noise_values = ['[Not Available]','[Not Applicable]','[Not Evaluated]','[Discrepancy]','[Unknown]','[Completed]','|',',']
            
            #Remove values such as '[Not Available]|[Not Available]|[Not Available]|9.5'
            for value in values[:]:
                if any(noise_value in value for noise_value in noise_values):
                    values.remove(value)
            

            values = [v for v in values if v not in noise_values]

            if values == []: #If no value available just skip
                continue

            # Section end -----------------

            if self.check_if_numeric(values):
                    
                data_type = 'number'

            #--- Section added by Sher Lynn-----

            elif self.check_if_yesno(values):

                data_type = 'boolean'

            #Section end-----------------------

            else:
                    
                data_type = 'string'
            
            headers_data_type.loc[c] = data_type
    
        return headers_data_type.reset_index()

    #--- Section added by Sher Lynn-----
    def check_if_yesno(self,values):
        '''
        Checks if a column contains yes no values
        
        Parameters
        ---------
        values: Series
            Data column that needs to be checked

        Returns
        --------
        bool
            if yes and no exists, return True
        '''
        if "YES" in values: #convert series into an array of lower cased string
            return True
        elif "NO" in values:
            return True
        else:
            return False

    #---- Section end-----------------------

    def check_if_numeric(self,values):
        '''
        Checks if a column is numeric

        Parameters
        ----------
        values : Series
            Data column that needs to be checked

        Returns
        -------
        bool
            DESCRIPTION.

        '''

        for v in values:
            try:
                float(v)
            except:
                return False
        return True


