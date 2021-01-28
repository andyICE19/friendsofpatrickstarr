# -*- coding: utf-8 -*-
"""
Created on Fri Oct  9 10:07:14 2020

The class in this file will ingest CDE data, process the data create a fasttext word embedding model using the data. 
!! WARNING: There are some hard coded elements (due to lack of time for a proper cleanup) in this file. Having hard coded elements in module code is agaist best practise.

@author: Tapesh


"""

######## Imports ######################

import pandas as pd
import os
import json
import numpy as np
from gensim.models import FastText
import time
from tqdm import tqdm
import multiprocessing
from cde_modelling.utils import Preprocessing as prs
from scipy.spatial.distance import cdist



################# Class definition #######################

class CDE_data_modeller:
    
    def __init__(self, cde_database_file, params):
        '''
        Constructor of the class CDE_data_modeller

        Parameters
        ----------
        cde_data_file : string
            path to the json file which stores the CDE data elements
        params: dict
        containing parameters like which field of the CDE element should we model
        e.g. , parameters for word embedding models etc.
        params = { 'descriptor': 'long_name'
        'fasttext' : {
        'epochs': 100
        'vecsize': 25 
        etc.
        }
        }
        Returns
        -------
        None.

        '''
        self.cde_database_file = cde_database_file
        self.params = params
        
        # utility class
        self.utilities = self.utils()
        
        #empty corpus
        self.corpus  = []
        
        # Step1: Load the CDE database        
        self.load_cde_data()
        
        # internal name for data vectors identifiers
        self.vector_name = 'headers'
        
    
    def load_cde_data(self) :
        '''
        Loads the json file containing CDE elements

        Returns
        -------
        None.

        '''
        
        print('Loading CDE database... please wait')
        
        start_time = time.time()
        
        try:
            
            with open(self.cde_database_file,'r') as file:
                self.cde_database_json = json.load(file)
        except:
            
            print('*******Could not load json data ********')
            
        end_time = time.time()
        
        print('Took %f  minutes to load CDE database..'%((end_time - start_time)/60) )
        
        
        
    def create_training_corpus (self):
        '''
        Creates a corpus for the CDE descriptors such as long names etc. and permissible values. Each descriptor/value is 
        converted to lower case, special characeter removed, then trnsformed into list of words
    
        Returns
        -------
        None.
     
        '''
        cde_descriptors = self.params['descriptor']
        
        #####self.descriptor_corpus = [ prs.preprocess(item[1][cde_descriptor]) for cde_descriptor in cde_descriptors for item in cde_database_json.items()]
        
        # add description corpus, basically means take the long name or any other descriptor of the CDE element and split into list of words after lowercasing
        
        for cde_descriptor in cde_descriptors:
            
            for item in self.cde_database_json.items():
                
                               
                self.corpus.append(prs.preprocess(item[1][cde_descriptor]))
                
        
        # add value corpus , basically means take the permissible values of the CDE element and split into list of words after lowercasing
        
        flatten = lambda l: [item for sublist in l for item in sublist]
        
        # all values appended next to each other are considered as one sentence
        value_list = lambda x : flatten([prs.preprocess(d ['valid_value']) for d in x])
        
        for item in self.cde_database_json.items():
            
            valid_values = value_list(item[1]['value']['permissible_values'])
            
            if len(valid_values):
                
                self.corpus.append(valid_values)
        
        
    def create_model_and_cde_indexes(self):
        '''
        Loads CDE data, creates model and indexes CDE data using the model

        Returns
        -------
        None.

        '''
        
        # Step2: Create the training corpus
        self.create_training_corpus()
        
        # Step3: Train word embedding model
        self.train_word_embedding_model()
        
        # Step 4: Vectorize all CDE elements
        
        # Step 4a: Create list of CDE descriptors       
        self.create_descriptor_list()
        
        # Step 4b: vectorize the descriptors
        self.descriptor_vectors = self.get_vectors( self.cde_list_of_descriptions )
        
        # Step 4c: Create a list of CDE vakues
        self.create_value_list()
        
        # Step 4d: self.get_vectors
        self.value_vectors = self.get_vectors(self.cde_value_list)
        
        
    def load_model_and_create_cde_indexes(self, model_filepath):
        '''
        Load a pretrained model and index CDE elements

        Parameters
        ----------
        model_filepath : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        '''
        
        #Step1 : Load the model
        
        self.load_model(model_filepath)
        
        # Step 2: Vectorize all CDE elements
        
        # Step 2a: Create list of CDE descriptors       
        self.create_descriptor_list()
        
        # Step 2b: vectorize the descriptors
        self.descriptor_vectors = self.get_vectors( self.cde_list_of_descriptions )
        
        # Step 2c: Create a list of CDE vakues
        self.create_value_list()
        
        # Step 2d: self.get_vectors
        self.value_vectors = self.get_vectors(self.cde_value_list)
        
        
    

    def load_model_and_cde_indexes(self, dirname):
        '''
        Load a pretrained model and indexes

        Parameters
        ----------
       dirnames : string
            path to model files

        Returns
        -------
        None.

        '''
        
        model_filepath = dirname+'//word2vec_model.txt'
               
        descriptor_vector_filepath = dirname+'//descriptor_vec.csv'
        
        value_vector_filepath = dirname+'//value_vec.csv'
       
        #Step1 : Load the model
        
        self.load_model(model_filepath)
        
        # Step 2: Vectorize all CDE elements       
        
        # Step 2a: load the precalculated descriptor vectors
        self.load_cde_descriptor_vectors( descriptor_vector_filepath )
        
        # Step 2b: load the precalculated value vectors
        self.load_cde_value_vectors( value_vector_filepath )
        
        

    
    def load_cde_descriptor_vectors (self, filename):
        '''
        

        Parameters
        ----------
        filename : string
            csv file contaiing the precaculated vectors

        Returns
        -------
        None.

        '''        
        self.descriptor_vectors = pd.read_csv(filename, header = 0, index_col=['public_id'])
    
    
    def load_cde_value_vectors (self, filename):
        '''
        

        Parameters
        ----------
        filename : string
            csv file contaiing the precaculated vectors

        Returns
        -------
        None.

        '''        
        self.value_vectors = pd.read_csv(filename, header = 0, index_col=['public_id'])
    
    def train_word_embedding_model(self):
        '''
        Trains a word embedding model using the CDE descriptors

        Parameters
        ----------
        params : dict
            containing model paramters. fields are             
            vector_size : Dimensionality of the feature vectors.
            window : The maximum distance between the current and predicted word within a sentence.
            min_count : Ignores all words with total frequency lower than this.
            alpha : The initial learning rate.
       

        Returns
        -------
        None.

        '''        
        
        print('Starting model training ... ')
        
        start = time.time()       
        
        self.model = FastText(sentences = self.corpus, window = self.params['fasttext']['window'], min_count = self.params['fasttext']['min_count'], iter = self.params['fasttext']['epochs'], size = self.params['fasttext']['vector_size'], workers = self.utilities.get_cpu_count())   
        
        end = time.time()
        print('Model training took %f minutes' %((end-start)/60))
        
    
    def create_value_list (self):
        '''
        Create a dictionary containing permissible values for the CDEs

        Returns
        -------
        None.

        '''
        flatten = lambda l: [item for sublist in l for item in sublist]
        
        value_list = lambda x : flatten ( [ prs.preprocess ( d ['valid_value'] )   for d in x ] )
                                         
        self.cde_value_list = { item[0] : value_list( item[1]['value']['permissible_values'] )  for item in self.cde_database_json.items()}

    
    def get_differences( self, vectors, vector_type, params ):
        '''
        Returns difference between a given set of vectors and cde  vectors
    

        Parameters
        ----------
        vectors : DataFrame
            given vectors
        vector_type : string
            possible options 'descriptor', 'value'
        difference_type : string
            possible options 'subtraction', 'absolute'
        params : dict
                params = {'type': 'standard' = standard difference, 'absolute' = absolute difference, 'squared' = squared difference}


        Returns
        -------
        TYPE
            DataFrame containing the differences

        '''
        
        if vector_type == 'descriptor':
            
            return self.utilities.pairwise_differences(self.descriptor_vectors , vectors, params)
        
        elif vector_type == 'value':
            return self.utilities.pairwise_differences(self.value_vectors , vectors, params)
        
        
    
    def get_metrics(self, vectors, vector_type, params , col_name = 'metrics'):
        '''
        Return similarities between a given set of vectors and CDE vectors

        Parameters
        ----------
        vectors : DataFrame
            given vectors
        vector_type : string
            possible options 'descriptor', 'value'
        params : dict
            params = {'sim_type': 'correlation', 'kernel', 'distances', 'metric': 'pearson', 'spearman', 'kendall', 'euclidian', etc., 'scaleing' : float for kernel only}
        Returns
        -------
        DataFrame 
            DataFrame containing the metrics of choices
        '''
        
        
        
        if params['sim_type'] == 'correlation':
            
            if vector_type == 'descriptor':
                
                return self.utilities.correlation_based_similarities(self.descriptor_vectors , vectors, params, col_name = col_name)
            
            elif vector_type == 'value':
                return self.utilities.correlation_based_similarities(self.value_vectors , vectors, params, col_name = col_name)
            
        else :  
            if vector_type == 'descriptor':
                
                return self.utilities.distances_or_kernel_similarities(self.descriptor_vectors , vectors, params, col_name = col_name)
            
            elif vector_type == 'value':
                return self.utilities.distances_or_kernel_similarities(self.value_vectors , vectors, params, col_name = col_name)
    

        
        
    def save_model(self, filename):
        '''
        save the model to a file, the model can then be loaded for nlp analytics

        Parameters
        ----------
        filename : string
            full path to the model file

        Returns
        -------
        None.

        '''
        self.model.save(filename)#.wv.save_facebook_format( filename )
        
        
    def load_model(self, filename):
        '''
        Load a fasttext model from file

        Parameters
        ----------
        filename : string
            path to a saved model

        Returns
        -------
        None.

        '''
        self.model = FastText.load(filename)#gensim.models.KeyedVectors.load_facebook_format( filename , binary=False, encoding='utf8')
    
    
    
    def save_model_and_indexes(self, dirname):
        '''
        saves model and indexes of data

        Parameters
        ----------
        dirname : string
            Path to the directory where model will be saved

        Returns
        -------
        None.

        '''
        try:
            os.makedirs(dirname, mode=0o777, exist_ok=False)
        except:
            pass
        
        self.save_model(dirname+'//word2vec_model.txt')
        self.descriptor_vectors.to_csv(dirname+'//descriptor_vec.csv')
        self.value_vectors.to_csv(dirname+'//value_vec.csv')
    
    
    
    def create_descriptor_list (self):
        '''
        Creates dictionary containing CDE descriptors for each items

        Returns
        -------
        None.

        '''
        self.cde_list_of_descriptions ={item[0]: prs.preprocess(    ' '. join( [ item[1][ descriptor] for descriptor in self.params['descriptor'] ] )   )  for item in self.cde_database_json.items()}

    
    
    
    def get_vector(self, cde_item):
        
        '''
        returns the vector corresponding to a sentence
    
        Parameters
        ----------
        cde_item : tuple
            a tuple containing a CDE element
    
        Returns
        -------
        vector = sentence vector
    
        '''
        
        sentence  =  cde_item[1]
        
        vector = np.zeros(self.params['fasttext']['vector_size'])
        for word in sentence:
            
            # standardize word vector
            v =self.model.wv[word]
            #v = (v-np.mean(v))/np.std(v)
            v = v / np.linalg.norm(v)
            
            vector += v
        
        vector = vector/len(sentence)
        
        return vector
    
    
    
    
    def get_vectors( self, list_of_sentences ):        
        '''
        Vectorizes a list of sentences
        
        inputs:
            list_of_sentences = a dict id : sentences that need to vectorized
            model = trained model
            combine = boolean flag indicating if we should combine word vectors into sentence vector or not
            
        outputs:
            sentences_to_vec = dataframe containing vectorized sentences
        '''
        
        # Vectorize the entire CDE database
        
        feature_columns = ['feature_vec_'+ str(i) for i in range(self.params['fasttext']['vector_size'])]
        
        # define column names for dataframe where the data will be stored
        
        all_columns = ['public_id']
        
        all_columns.extend(feature_columns)
        
        # create dataframe where the vectors will be stored
        
        sentences_to_vec = pd.DataFrame(columns = all_columns)
        
        #sentences_to_vec.set_index( ['public_id'], inplace = True)
        
        cde_items = list(list_of_sentences.items())       
        
        #Record time
        print('Start converting descriptors to vectors' )
        start_time = time.time()
        
        
        
        #pool = Pool(cpu_count)
        #vectors = pool.map(get_vector,cde_items)
        
        # The following code is meant to avoid dynamic allocation into a dataframe
        # which is very slow
        vectors = np.zeros ( (   len ( list_of_sentences ) , self.params [ 'fasttext' ][ 'vector_size' ] ) ) 
        public_ids = []
        
                                                              
        for i in tqdm( range ( len ( cde_items ) ) ):
            
            vec = self.get_vector(cde_items[i])
            #print(vec)
            #sentences_to_vec.loc[cde_items[i][0]] = vec
            public_ids.append( cde_items[i][0] )
            vectors[i,:] = vec
        
        sentences_to_vec['public_id'] = public_ids
        sentences_to_vec[feature_columns] = vectors
        
        end_time = time.time()
        
        print('Took %f minutes to vectorize the dataset'%((end_time-start_time)/60))
        #print(sentences_to_vec)
        
        sentences_to_vec.fillna(0, inplace=True)
        
        sentences_to_vec.set_index('public_id', inplace = True)
        
        return sentences_to_vec
    
    def get_data_types(self):
        '''
        Get data types

        Returns
        -------
        cde_value_types : DataFrame
            Simplified data types of the cde elements

        '''
        # try:
        #     len(self.cde_database_json)
        # except:
        #     self.load_cde_data()
        
        try:
            # if it's already created return
            return self.cde_value_types
        except:
            
            #else return the following                
            self.cde_value_types = {item[0] : self.utilities.data_type_simplifier(item[1])  for item in self.cde_database_json.items()}
    
            self.cde_value_types = pd.DataFrame.from_dict (self.cde_value_types, orient = 'index').reset_index().rename({'index': 'public_id', 0:'data_type'}, axis=1)
            self.cde_value_types = pd.get_dummies(self.cde_value_types, columns = ['data_type']).set_index('public_id')
            
            self.cde_value_types.index = self.cde_value_types.index.astype(np.int64)

            return self.cde_value_types

    
    def create_abt(self, dataset, params = None):
        '''
        

        Parameters
        ----------
        dataset : dictionary
           dataset = {self.vector_name: dictionary containing headers, 'values': dictionary containing values, data_types= DataFrame containing datatypes}
        params : dictionary
            parameters for the similarity matrices
        gold_standard : dictionaty
            containes the ground truth. The default is None.

        Returns
        -------
        DataFrame
            joined model or score ABT

        '''
        if params is None:
            params = self.params
        
        headers_embedding_vectors = self.get_vectors( dataset[self.vector_name] )
        value_embedding_vectors = self.get_vectors ( dataset['values'] )
        data_types = dataset['value_type']
        
        #change the index names to differentiate from the cde vectors
        headers_embedding_vectors.index.name = self.vector_name
        value_embedding_vectors.index.name = self.vector_name
        data_types.index.name = self.vector_name
        
        
        
        headers_embedding_vectors.rename(columns = {'public_id' : self.vector_name}, inplace = True) 
        value_embedding_vectors.rename(columns = {'public_id' : self.vector_name}, inplace = True) 
        
        # descriptor features
        header_differences = self.get_differences( headers_embedding_vectors, 'descriptor',  params["features"]["differences"] )
        
        # value features
        value_differences = self.get_differences( value_embedding_vectors, 'value',  params["features"]["differences"] )
        
        # header similarities
        header_similarities = self.get_metrics(headers_embedding_vectors, 'descriptor', params["features"]["metrics"], col_name = 'header_metrics' )
        
        # value similarities
        value_similarities = self.get_metrics(value_embedding_vectors, 'descriptor',  params["features"]["metrics"], col_name = 'value_metrics' )
        
        # get data types
        cde_data_types = self.get_data_types()        
        data_type_sim = self.utilities.distances_or_kernel_similarities(cde_data_types, data_types, {"metric":"cosine", "sim_type":"normal"})
        
        if 'gold_standard' in dataset.keys() :
            # creating model abt
            
            target = pd.DataFrame.from_dict(dataset['gold_standard'], orient = 'index').reset_index().rename(columns = {'index':self.vector_name, 0:'public_id'})
            target['public_id'] = target['public_id'].astype(str).astype(int)
            target['target'] =1
        
            # join features together
            joined = header_differences.merge(value_differences, left_on = ['public_id',self.vector_name], right_on = ['public_id', self.vector_name], how = 'left')\
                              .merge(header_similarities, left_on = ['public_id',self.vector_name], right_on = ['public_id', self.vector_name], how = 'left' )\
                              .merge(value_similarities, left_on = ['public_id',self.vector_name], right_on = ['public_id', self.vector_name], how = 'left' )\
                              .merge(data_type_sim, left_on = ['public_id',self.vector_name], right_on = ['public_id', self.vector_name], how = 'left' )\
                              .merge(target, left_on = ['public_id',self.vector_name], right_on = ['public_id', self.vector_name], how = 'left' )
                              
                              
                
            joined.fillna(0 , inplace = True)
            
            # now undersample
            
         
            
            positive_class_total = np.sum(joined['target'])
            
            negative_class_total = positive_class_total * params["features"]['sampling_ratio']
            
          
            
            pos_class = joined.loc[joined ['target']==1 ]
            neg_class = joined.loc[joined ['target']==0 ].sample( n = int(negative_class_total))
            
            dataset = pd.concat([pos_class,neg_class])
            return dataset
        
        else:
            joined = header_differences.merge(value_differences, left_on = ['public_id',self.vector_name], right_on = ['public_id', self.vector_name], how = 'left')\
                              .merge(header_similarities, left_on = ['public_id',self.vector_name], right_on = ['public_id', self.vector_name], how = 'left' )\
                              .merge(value_similarities, left_on = ['public_id',self.vector_name], right_on = ['public_id', self.vector_name], how = 'left' )\
                              .merge(data_type_sim, left_on = ['public_id',self.vector_name], right_on = ['public_id', self.vector_name], how = 'left' )\
                              
            return joined
        
            
         
        
    def create_base_for_unsupervised_learning(self, dataset = None):
        '''
        Creates headers values and value type vectors for the CDE elements

        Parameters
        

        
        -------
        DataFrame
            joined CDE vectors

        '''
        if dataset is None:            
            self.get_data_types()
            dataset = pd.merge (self.descriptor_vectors, self.value_vectors, left_on = ['public_id'], right_on = ['public_id'], how = 'left', suffixes = ('_desc','_val'))\
                      .merge(self.cde_value_types, left_on = ['public_id'], right_on = ['public_id'], how = 'left')\
                      .fillna(0)
            
            return dataset
        else:
            headers_embedding_vectors = self.get_vectors( dataset[self.vector_name] )
            value_embedding_vectors = self.get_vectors ( dataset['values'] )
            data_types = dataset['value_type']
            
            dataset1 = pd.merge (headers_embedding_vectors, value_embedding_vectors, left_on = ['public_id'], right_on = ['public_id'], how = 'left', suffixes = ('_desc','_val'))\
                      .merge(data_types, left_on = ['public_id'], right_on = ['public_id'], how = 'left')\
                      .fillna(0)
            return dataset1
           
            
        
        

        
                                     
                
    
        
    
    class utils:
        ''' 
        A calss containing the utility functions and variables for CDE data modeller
        
        '''
        
        def __init__(self):
            '''
            Class contructor

            Returns
            -------
            None.

            '''
            # definition of numbers
            self.numbers = ['number', 'float', 'integer', 'double', 'long', 'short']

            
        def data_type_simplifier (self, cde_element):
            '''      
            CDE elements can have various data types ranging from character, string, float, integer, double
            boolean etc. Here we are simplifying the data type to string or number.
            
            The logic of the conversion is as follows:
                1. if the CDE  element contains a list of permissible values, we check if the values are number or strings
                   if, the values are all numbers then we say data type is a number. if permissible values are there we don't look at the 
                   designated data_type of the CDE elements, because the entries in that field is quite messy and noisy.
                   
                2. if there are no permissible values, then we look at the data_types of the CDE element. All numeric data types e.g. 
                integer, long, double, float etc. a grouped together as number. everything else is string
                
                ***** This function can be improved to have a more fine grained data_type definition *****
            
            
            Parameters
            ----------
            cde_element : dict
                Dictionary containing the CDE element.
    
            Returns
            -------
            simple_type: string                    
    
            '''
        
            # get the data types and permissible values
            data_type = cde_element['value']['data_type'].lower()
            
            permissible_values = self.value_list ( cde_element['value']['permissible_values'] )
            
            check_number = lambda x : x in self.numbers if 'java' not in x else x.split(sep = '.')[2] in self.numbers
            
            #we are not using the following at the moment, the definition of boolean is not clear to me
            #check_boolean = lambda x : x in boolean if 'java' not in x else x.split(sep = '.')[2] in boolean
            

            simple_type = 'string'
            
             # 1. if the CDE  element contains a list of permissible values, we check if the values are number or strings
             #       if, the values are all numbers then we say data type is a number. if permissible values are there we don't look at the 
             #       designated data_type of the CDE elements, because the entries in that field is quite messy and noisy.
                   
            
            if len(permissible_values)>0:
                if self.check_if_numeric (permissible_values):
                    return 'number'

                #--- section added by Sher Lynn
                if self.check_yes_no(permissible_values):
                    return 'boolean'
                #java.lang.Boolean to be marked as string.
                #--- section end

            # 2. if there are no permissible values, then we look at the data_types of the CDE element. All numeric data types e.g. 
            #     integer, long, double, float etc. a grouped together as number. everything else is string    
            if check_number ( data_type ):
                simple_type = 'number'
                
            # we are not using the following at the moment
            # elif check_boolean(data_type):
            #     simple_type = 'boolean'
            
            return simple_type

        # --- Section added by Sher Lynn
        def check_yes_no(self, values):
            '''
            In the event that the CDE elements have a list of values, we check if the values are "Yes", "No"
            
            Parameters
            --------------
            values: iterable

            Returns
            --------------
            bool
                True if and only if valid value list has both "Yes" and "No" Options. Examples of acceptable ones:
                    - "Yes", "No", "NA"
                    - "Yes", "No", "Unknown" 
                    - "yes", "no", "not done"
                    - "yes", "no", "unknown", "inadequate sample"
                False otherwise, for example
                    if options include "Yes, <explanation>" or "No, <explanation>"

            '''
            if "yes" in values and "no" in values:
                return True
            
            elif "Yes" in values and "No" in values:
                return True
            else:
                return False

        # --- Section end
        
        def check_if_numeric(self, values):
            '''
            In the event that, the CDE elements have a list of values, here we check if the values are numeric

            Parameters
            ----------
            values : iterable
                DESCRIPTION.

            Returns
            -------
            bool
                True if all values are numeric
                True if all values except one are numeric
                Else False


            '''
            #--- Section modified by Sher Lynn ---
             # if all is number, except one field, it should be numeric
            
            ind = {}

            for v in values:
                try:
                    float(v)
                    ind[v] = True
                except:
                    ind[v] = False

            if all(i == 1 for i in ind.values()): # If all True
                return True
            elif sum(i == 0 for i in ind.values()) == 1: # If there is only one False
                return True
            else: #If there is more than one False.
                return False
            #---- Section end
        
        def value_list(self, values ):    
                
            '''
             
             
             Parameters
             ----------
             values : list of dicts
                 list of dictionary containing valid value and id.
             
             Returns
             -------
             list of valid values
             
             ''' 
            value_list =[]
            for d in values:
                value_list.extend(d['valid_value'].lower().split())
            
            return value_list 
        
        
        def get_cpu_count(self):
            '''
            return the number of CPUS to use for modelling

            Returns
            -------
            integer
                number of CPUs to use for modelling.

            '''
            return multiprocessing.cpu_count()-1
        
        
        # def cosine_similarities(self, df1 , df2):
        #     '''
        #     Calculates cosine similarities between 2 data frames

        #     Parameters
        #     ----------
        #     df1 : Dataframe
        #         DESCRIPTION.
        #     df2 : Dataframe
        #         DESCRIPTION.

        #     Returns
        #     -------
        #     cosine_similarities : DataFrame
        #         DESCRIPTION.

        #     '''
            
            
        #     delta = 0.0001
            
        #     dot_product = df1.dot(df2.transpose())
            
            
        #     norm1 = pd.DataFrame( np.sqrt ( np.square ( df1 ).sum( axis = 1 )))            
        #     # to avoid erros zero norms are replaced by a small number
        #     norm1 [ norm1 == 0] = delta
            
        #     norm2 = pd.DataFrame( np.sqrt ( np.square ( df2 ).sum( axis = 1 )))
        #     # to avoid erros zero norms are replaced by a small number
        #     norm2 [ norm2 == 0] = delta
            
        #     product_of_norms = norm1.dot( norm2.transpose() )
            
            
            
        #     cosine_similarities = dot_product/product_of_norms
            
        #     cosine_similarities.fillna( 0, inplace =True)
            
        #     return  cosine_similarities
        
        def correlation_based_similarities(self, df1, df2, params, col_name = 'metric'):
            '''
            Computes correlation between two sets of embedding vectors. 
            Warning: Vectorized, but not space optimized. For  large matrices use a space optimized version instead.

         
            Parameters
            ----------
            df1 : DataFrame
                first set of embedding vectors
            df2 : DataFrame
                second set of embedding vectors
            params :dictionary
               field= metric:  possible values = 'pearson', 'kendall', 'pearson'
            col_name: string
              name of the metric column in the resulting data

            Returns
            -------
            correlation : DataFrame
                correlation between two dataframes

            '''
            # to avoid duplicate columns names
            #df2.rename(columns = {'public_id' : self.vector_name}, inplace = True)
            
            # The function below is not a space and time optimized, use numpy instead
            correlation = pd.concat([df1, df2], axis=1, keys=['df1', 'df2']).corr(method = params['metric']).loc['df2', 'df1']
            
            correlation = correlation.stack().reset_index().rename(columns = {0:col_name})
            
            return correlation 
        
        def distances_or_kernel_similarities(self, df1 , df2 , params, col_name = 'metric'):
            '''
            Computes distances or kernel similarities between the rows of two dataframes. The dataframe need to have the same number of columns. 
            Warning: Vectorized, but not space optimized. For  large matrices use a space optimized version instead.

            Parameters
            ----------
            df1 : Pandas DataFrame
                DataFrame containing embedding vectors.
            df2 : Pandas DataFrame
                Containing embedding vectors.
            params : dictionary
                defition: {'sim_type' : 'kernel' computes a kernel function), 'dist' computes a distance function
                           'metric' : 'euclidian', 'minkowski', 'cityblock' etc. see https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.cdist.html for more options
                           'scaling' : float (only used for kernels, not distances)
                          }
            col_name: string
              name of the metric column in the resulting data

            Returns
            -------
            df : DataFrame
                Distance of Kernel matrx

            '''
            #df2.rename(columns = {'public_id' : self.vector_name}, inplace = True)
            
            #print(df1.columns, df2.columns)
            
            distances = cdist(df1, df2, params['metric'])
            
            if params['sim_type'] == 'kernel'   :
                distances = np.exp(-distances/params['scaling'])
                
            df =  pd.DataFrame(distances, index = df1.index, columns = df2.index)       
           
            return df.stack().reset_index().rename(columns = {0:col_name})
        
        
        def pairwise_differences(self,df1, df2, params):
            '''
            Pairwise differences between word embedding vectors.

            Parameters
            ----------
            df1 : DataFrame
                word embedding 1
            df2 : DataFrame
                word embedding 2
            params : dict
                params = {'type': 'standard' = standard difference, 'absolute' = absolute difference, 'squared' = squared difference}

            Returns
            -------
            differences : DataFrame 
                DESCRIPTION.

            '''
            
            
            nrows_df1 = df1.shape[0]
            
            nrows_df2= df2.shape[0]
            
            # repeat each row of df2 nrows_df1 times
            expanded_df2 = df2.loc[df2.index.repeat(nrows_df1)]#.reset_index(drop=True)
            
            # repeat the entire df1 nrows_df2 times
            expanded_df1 = pd.concat([df1]*nrows_df2)
            
            differences = pd.DataFrame( expanded_df1.values - expanded_df2.values , columns = df1.columns)
            
            if params['type'] == 'standard':
                pass
            
            elif params['type'] == 'absolute':
                differences = differences.abs()
                
            elif params['type'] == 'squared':
                differences = differences.pow(2)
            
            differences[ expanded_df1.index.name ] = expanded_df1.index
            
            differences[ expanded_df2.index.name ] = expanded_df2.index
            
            return differences
            
            
            
            
            
        


        
        
        