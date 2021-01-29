# -*- coding: utf-8 -*-
"""
Created on Thu Nov  5 11:34:36 2020

This file creates classification models depending on parameters sent to the models.

@author: Tapesh
"""


from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors, KNeighborsClassifier
from sklearn import metrics

class Model:
    ''' 
    Wrapper class for creating ml-models
    '''
    
    def __init__(self,params):
        '''
        Class constructor
        Parameters
        ----------
        params : dictionary
            type and parameters of the model
        Returns
        -------
        None.
        '''
        self.params = params
        self.model = []
        
        # Create classification models depending on parameters
        if params['model']['name'] =='gradient boost':

            self.model = GradientBoostingClassifier(**params['model']['model_params'])
            
        elif params['model']['name'] =='random forest':
            self.model = RandomForestClassifier(**params['model']['model_params'])
            
        elif params['model']['name'] =='logistic regression':
            self.model = LogisticRegression(**params['model']['model_params'])

        elif params['model']['name'] == 'XGBoost':
            print("XGBoost")
            self.model = xgb.XGBClassifier(**params['model']['model_params'])

        elif params['model']['name'] =='knn classifier':
            self.model = KNeighborsClassifier(**params['model']['model_params'])
        
        # The following is is unspuervised model, and not a classification model
        elif params['model']['name'] =='unsupervised knn':
            self.model = NearestNeighbors(**params['model']['model_params'])
        else:
            print('Invalid model type')
        
        
    def fit(self, X, y=None):
        '''
        Fits the model to data.

        Parameters
        ----------
        X : DataFrame or numpy array
            training features
        y : Series or numpy array, optional
            training target. not needed for unsupervised approaches. The default is None. 

        Returns
        -------
        None.

        '''
        self.X = X
 
        if self.params['model']['name'] =='unsupervised knn':
            self.model.fit(X)
            
        else:
            self.model.fit(X,y)
            
    
    def predict(self, X, probability=True):
        
        if self.params['model']['name'] =='unsupervised knn':
            return self.model.kneighbors(X)
            
        else:
            if probability == True:
                return self.model.predict_proba(X)
            else:
                return self.model.predict(X)
    
    
    def accuracy(self, X, y_true):
        
        accuracy_vals = {}
        
        if self.params['model']['name'] =='unsupervised knn':
            print('Not applicable for unsupervised algos')
        else:
            y_pred = self.predict(X, probability = False)
            y_pred_proba = self.predict(X)
            
            accuracy_vals ['accuracy'] = metrics.accuracy_score(y_true, y_pred)
            accuracy_vals ['balanced_accuracy'] = metrics.balanced_accuracy_score(y_true, y_pred)
            accuracy_vals ['f1'] = metrics.f1_score(y_true, y_pred)
            accuracy_vals ['precision'] = metrics.precision_score(y_true, y_pred)
            accuracy_vals ['recall'] = metrics.recall_score(y_true, y_pred)
            accuracy_vals ['auroc'] = metrics.roc_auc_score(y_true, y_pred_proba[:,1])
            #accuracy_vals ['confusion_matrix'] = metrics.confusion_matrix(y_true, y_pred)
            #accuracy_vals ['matthews'] = metrics.matthews_corrcoef(y_true, y_pred)
            
            return accuracy_vals
            
            
    def predict_and_convert_to_json(self, X, K=None, index_cols=None, header_col=None, id_col=None):
        '''
        Make prediction and convert the predictions into json format that will be scored using accuracy models

        Parameters
        ----------
        X : DataFrame
            test data
        K : int
            top K predictions for each CDE value
        index_cols: list of index columns in X
        header_col: string
                    column in X that contains the data headers
        id_col: string
            column in X that contains the cde_ids
        Returns
        -------
        predictions : dictionary containing top K predictions 

        '''
        
        # get only feature colums which will be sent for prediction
        
        if self.params['model']['name'] != 'unsupervised knn':
            
            feature_cols = [c for c in X.columns if c not in index_cols]
            
            data = X[feature_cols].values
            
            # make prediction
            y_pred = self.predict(data, probability = True)
            
            # create DataFrame containing headers, cde_ids, and the probability that the header correspond to the CDE_ID
            
            df = X[index_cols].copy()
            df['probability'] = y_pred[:,1]
            
            # for each header fet the top K predictions
            res = df.groupby( by =index_cols).sum()['probability'].groupby(header_col, group_keys=False).nlargest(K)
            
            res= res.reset_index()
    
            # create the dictionary containing predictions
            headers = res[header_col].unique()
            
            results_dict = {}
            
            for th in headers:
                pid = list(res.loc [ res [header_col] == th, id_col].values)
                results_dict[th] = pid
            
            results_dict =  {k: [str(v) for v in results_dict[k] ] for k in results_dict.keys()}
                
            return results_dict
        
        else:
            distances, indices = self.predict(X.values)
            tags = self.X.index
    
            results_dict = { X.index [ i ] : tags [ indices[i] ]  for i in range(len(indices))}
            results_dict =  {k: [str(v) for v in results_dict[k] ] for k in results_dict.keys()}
            
            return results_dict
            

            
        
        
            
            
            
    
        
        
    
        
        
        
        
