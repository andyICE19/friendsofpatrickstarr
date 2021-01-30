# -*- coding: utf-8 -*-
"""
Created on Thu Nov  5 11:21:26 2020

@author: Tapesh
"""


def calculate_accuracy(gold_standard, predicted):
    '''
    Function to calculate accuracy score
    
    inputs:
        
        gold_standard = dicttionary containing gold standard        
        predicted = dict containing predicted values
    
    outputs:
        avs_score = float, average accuracy score (between 0 and 1)
    '''

       
    total_score = 0
    #count = 0
    
    for item in predicted.items():
        
        
        # gold standard lable
        if item[0] in gold_standard.keys():
            
            assigned_cde_id = gold_standard[item[0]]
            
            # predicted label
            inferred_cde_ids = list(item[1])
            
            #print(inferred_cde_ids.index( str( assigned_cde_id ) ) )
            
            # number of neighbours
            K = len(inferred_cde_ids)
            
            inference_score = 0
            try:
                inference_score = ( K - inferred_cde_ids.index( str( assigned_cde_id ) ) )/K 
            except:
                pass
            
            total_score += inference_score
        
    average_score = total_score / len(predicted)
    
    return average_score
        