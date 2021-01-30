# -*- coding: utf-8 -*-
"""
Created on Wed Jan 27

To Include Cross Validation

@author: Andy
"""

# import the Model class
from cde_modelling.modelling.create_models import Model

# stratified k-fold cross validation evaluation of xgboost model
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_validate
from sklearn.model_selection import GridSearchCV
import mlflow
import pickle
import glob
import json
import pandas as pd
from cde_modelling.utils import Accuracy_calculations as ac

def log_result_for_gridsearch(params, gs_clf):
    
    results = gs_clf.cv_results_
    
    for i, cvparams in enumerate(results['params']):
        accuracy = {}
        accuracy['mean'] = {}
        accuracy['std'] = {}

        #print(cvparams)
        #print("Accuracy: %.2f%% (%.2f%%)" % (results["mean_test_acc"][i]*100, results["std_test_acc"][i]*100))
        #print("Balanced Accuracy: %.2f%% (%.2f%%)" % (results["mean_test_b_acc"][i]*100, results["std_test_b_acc"][i]*100))
        #print("F1: %.2f%% (%.2f%%)" % (results["mean_test_f1"][i]*100, results["std_test_f1"][i]*100))
        #print("Precision: %.2f%% (%.2f%%)" % (results["mean_test_prec"][i]*100, results["std_test_prec"][i]*100))
        #print("Recall: %.2f%% (%.2f%%)" % (results["mean_test_rec"][i]*100, results["std_test_rec"][i]*100))
        #print("AUC: %.2f%% (%.2f%%)" % (results["mean_test_auc"][i]*100, results["std_test_auc"][i]*100))
        #print("\n")

        accuracy['mean']["accuracy"] = results["mean_test_acc"][i]
        accuracy['mean']["balanced_accuracy"] = results["mean_test_b_acc"][i]
        accuracy['mean']["f1"] = results["mean_test_f1"][i]
        accuracy['mean']["precision"] = results["mean_test_prec"][i]
        accuracy['mean']["recall"] = results["mean_test_rec"][i]
        accuracy['mean']["auc"] = results["mean_test_auc"][i]

        accuracy['std']["accuracy"] = results["std_test_acc"][i]
        accuracy['std']["balanced_accuracy"] = results["std_test_b_acc"][i]
        accuracy['std']["f1"] = results["std_test_f1"][i]
        accuracy['std']["precision"] = results["std_test_prec"][i]
        accuracy['std']["recall"] = results["std_test_rec"][i]
        accuracy['std']["auc"] = results["std_test_auc"][i]

        mlflow.set_experiment(str(params['model']['name'])+'Tuning Model Parameters')

        with mlflow.start_run():
            # print out current run_uuid
            run_uuid = mlflow.active_run().info.run_uuid
            print("MLflow Run ID: %s" % run_uuid)
            
            # log parameters
            #mlflow.log_param("window_size", params["fasttext"]["window"])
            #mlflow.log_param("min_count", params["fasttext"]["min_count"])
            #mlflow.log_param("epochs", params["fasttext"]["epochs"])
            #mlflow.log_param("vector_size", params["fasttext"]["vector_size"])
            
            #mlflow.log_param("features_diference_types", params["features"]["differences"]["type"])
            #mlflow.log_param("features_metrics", params["features"]["metrics"]["metric"])
            #mlflow.log_param("features_metrics_sim_type", params["features"]["metrics"]["sim_type"])
            #mlflow.log_param("features_metrics_scaling", params["features"]["metrics"]["scaling"])
            #mlflow.log_param("features_sampling_ratio", params["features"]["sampling_ratio"])
            
            #mlflow.log_param("features_samplinf_ratio", params["features"]["sampling_ratio"])
            
            #mlflow.log_param("model_type", params['model']["name"])

            for k in cvparams.keys():
                mlflow.log_param("model_params_"+k, cvparams[k])
            
            # log metrics
                
            for k in accuracy['mean'].keys():
                mlflow.log_metric("mean_val_accuracy_"+k,accuracy['mean'][k])
                mlflow.log_metric("std_val_accuracy_"+k,accuracy['std'][k])
            
            #mlflow.sklearn.logmodel()
            # with open('models/'+run_uuid+'.pkl','wb') as file:
            #     pickle.dump(model, file)
            #
            mlflow.end_run()



def prep_ml(params, abt, clf="gradient boost", model_params={'n_estimators': 125}, test_size=0.2, 
            random_state=42, cv=False, gridsearch=False):
    
    # First create feature vectors
    features = [c for c in abt.columns if ('feature' in c) or ('metric' in c)]

    # create design matrix and target data
    X = abt[features]
    y= abt['target']

    # create training and validation data
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=test_size, random_state=random_state)


    # import the Model class
    from cde_modelling.modelling.create_models import Model
    
    # Manually assign value
    params['model']['name'] = clf
    params['model']['model_params'] = model_params
    
    if not(cv):
        # create model
        model = Model(params)

        #fit the model
        model.fit(X_train,y_train)

        # calculate the accuracy of the model on the validation data
        accuracy = model.accuracy(X_val, y_val)
        
        print(accuracy)
        
        return model, accuracy
        
    elif not(gridsearch):

        # CV model
        model = Model(params).model
        kfold = StratifiedKFold(n_splits=10, random_state=66)
        scoring = {'acc': 'accuracy',
           'b_acc': 'balanced_accuracy',
           'f1': 'f1',
           'prec': 'precision',
           'rec': 'recall',
           'auc': 'roc_auc' 
          }
        print(params)
        scores = cross_validate(model, X, y, scoring=scoring, cv=kfold, return_train_score=False)

        accuracy = {}

        accuracy ['accuracy'] = {'average': scores["test_acc"].mean()*100, 'std': scores["test_acc"].std()*100}
        accuracy ['balanced_accuracy'] = {'average': scores["test_b_acc"].mean()*100, 'std': scores["test_b_acc"].std()*100}
        accuracy ['f1'] = {'average': scores["test_f1"].mean()*100, 'std': scores["test_f1"].std()*100}
        accuracy ['precision'] = {'average': scores["test_prec"].mean()*100, 'std': scores["test_prec"].std()*100}
        accuracy ['recall'] = {'average': scores["test_rec"].mean()*100, 'std': scores["test_rec"].std()*100}
        accuracy ['auc'] = {'average': scores["test_auc"].mean()*100, 'std': scores["test_auc"].std()*100}


        print("Accuracy: %.2f%% (%.2f%%)" % (scores["test_acc"].mean()*100, scores["test_acc"].std()*100))
        print("Balanced Accuracy: %.2f%% (%.2f%%)" % (scores["test_b_acc"].mean()*100, scores["test_b_acc"].std()*100))
        print("F1: %.2f%% (%.2f%%)" % (scores["test_f1"].mean()*100, scores["test_f1"].std()*100))
        print("Precision: %.2f%% (%.2f%%)" % (scores["test_prec"].mean()*100, scores["test_prec"].std()*100))
        print("Recall: %.2f%% (%.2f%%)" % (scores["test_rec"].mean()*100, scores["test_rec"].std()*100))
        print("AUC: %.2f%% (%.2f%%)" % (scores["test_auc"].mean()*100, scores["test_auc"].std()*100))

        model.fit(X, y)

        return model, accuracy
    
    else:

        params2 = params
        params2['model']['model_params'] = {}
        model = Model(params2).model
        kfold = StratifiedKFold(n_splits=10, random_state=66)
        scoring = {'acc': 'accuracy',
           'b_acc': 'balanced_accuracy',
           'f1': 'f1',
           'prec': 'precision',
           'rec': 'recall',
           'auc': 'roc_auc' 
          }
        
        clf = GridSearchCV(model, params["cv"]["model_params"], scoring=scoring, cv=kfold, return_train_score=False,
                           refit='f1', verbose=3, n_jobs=-1)
        
        clf.fit(X,y)

        # return the Model object of model with the best set of parameter
        params["model"]["model_params"] = clf.best_params_
        best_model = Model(params)
        best_model.fit(X, y)
        
        log_result_for_gridsearch(params, clf) #logs results in mlflow under experiment 'Parameter Tuning'

        return best_model, clf.best_score_, clf.best_params_ #returns the best model, the best score for the model (in this case f1 score), and model name


def make_predictions(model, val_accuracy, clf, params, test_abt_no_target, K, index_cols, header_col, id_col, clinical_data_test_dir, json_output=False,accuracy_and_mlflow_output=True):
    '''
    Parameters: 
    model
    val_accuracy from prep_ml output
    clf: "random forest" or "XGBoost"
    params: params output from prep_ml. a dictionary {<param name>: <best value>}
    input abt with no target
    top _ predictions to output
    index_cols
    header_col
    id_col
    json_output: If True, save top K predictions in a json
    accuracy_and_mlflow_output: If True will calculate and log test accuracy in mlflow

    '''
    results = model.predict_and_convert_to_json(test_abt_no_target,20, index_cols, header_col, id_col)

    all_files = glob.glob(clinical_data_test_dir + "*.txt")

    test_gs_dict ={}

    # print the ground truth of the datasets
    for filename in all_files:
        df = pd.read_csv(filename, index_col = 0, header = 0, sep='\t')
        
        #keeping only rows that has CDE number
        df = df[df.iloc[:, 0].str.contains("CDE")]
        df.reset_index(drop=True, inplace = True)
        
        #striping 'CDE_ID' from string, leaving only the ID numbers
        test_gs_dict.update(df.iloc[0,:].apply(lambda x: x.replace('CDE_ID:','')).to_dict())

    # remove empty header if any
    #test_gs_dict = {key:val for key, val in test_dict.items() if val != ''}

    # Output the top K predictions in a json file
    if json_output:
        with open("finalpredictions.json","w") as outfile:
            json.dump(results, outfile)

    if accuracy_and_mlflow_output:
        test_accuracy = ac.calculate_accuracy(test_gs_dict,results)

        mlflow.set_experiment('Compare ML Models')
        with mlflow.start_run():
            # print out current run_uuid
            run_uuid = mlflow.active_run().info.run_uuid
            print("MLflow Run ID: %s" % run_uuid)
            
            # log parameters
            #mlflow.log_param("window_size", params["fasttext"]["window"])
            #mlflow.log_param("min_count", params["fasttext"]["min_count"])
            #mlflow.log_param("epochs", params["fasttext"]["epochs"])
            #mlflow.log_param("vector_size", params["fasttext"]["vector_size"])
            
            
            #mlflow.log_param("features_diference_types", params["features"]["differences"]["type"])
            #mlflow.log_param("features_metrics", params["features"]["metrics"]["metric"])
            #mlflow.log_param("features_metrics_sim_type", params["features"]["metrics"]["sim_type"])
            #mlflow.log_param("features_metrics_scaling", params["features"]["metrics"]["scaling"])
            #mlflow.log_param("features_sampling_ratio", params["features"]["sampling_ratio"])
            
            #mlflow.log_param("features_samplinf_ratio", params["features"]["sampling_ratio"])
            
            mlflow.log_param("model_type", clf)
            
            for k in params.keys():#['model']['model_params'].keys():
                mlflow.log_param("model_params_"+k, params[k])#['model']["model_params"][k])
            
            # log metrics
            #CHECK THIS SECTION
            mlflow.log_metric("test_accuracy",test_accuracy)
            mlflow.log_metric("val_accuracy_f1",val_accuracy) #hardcoded
            
            #for k in val_accuracy.keys():
            #    if 'confusion' not in k:
            #        mlflow.log_metric("val_accuracy_"+k,accuracy['mean'][k])
            #        mlflow.log_metric("val_accuracy_"+k,accuracy['std'][k])
            
            #mlflow.sklearn.logmodel()
            with open('models/'+run_uuid+'.pkl','wb') as file:
                pickle.dump(model.model, file)
            
            mlflow.end_run()

        return test_accuracy