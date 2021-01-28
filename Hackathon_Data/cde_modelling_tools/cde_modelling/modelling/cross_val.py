# -*- coding: utf-8 -*-
"""
Created on Wed Jan 27

To Include Cross Validation

@author: Andy
"""

# import the Model class
from cde_modelling.modelling.create_models import Model

# stratified k-fold cross validation evaluation of xgboost model
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_validate
from sklearn.model_selection import GridSearchCV

def log_result_for_gridsearch(gs_clf):
    
    results = gs_clf.cv_results_
    
    for i, params in enumerate(results['params']):
        print(params)
        print("Accuracy: %.2f%% (%.2f%%)" % (results["mean_test_acc"][i]*100, results["std_test_acc"][i]*100))
        print("Balanced Accuracy: %.2f%% (%.2f%%)" % (results["mean_test_b_acc"][i]*100, results["std_test_b_acc"][i]*100))
        print("F1: %.2f%% (%.2f%%)" % (results["mean_test_f1"][i]*100, results["std_test_f1"][i]*100))
        print("Precision: %.2f%% (%.2f%%)" % (results["mean_test_prec"][i]*100, results["std_test_prec"][i]*100))
        print("Recall: %.2f%% (%.2f%%)" % (results["mean_test_rec"][i]*100, results["std_test_rec"][i]*100))
        print("AUC: %.2f%% (%.2f%%)" % (results["mean_test_auc"][i]*100, results["std_test_auc"][i]*100))
        print("\n")

def prep_ml(params, abt, clf="gradient boost", model_params={'n_estimators': 125}, test_size=0.2, 
            random_state=42, cv=False):
    
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

        accuracy ['accuracy'] = {'average': scores["test_acc"].mean()*100, 'std': scores["test_acc"].std()*100)}
        accuracy ['balanced_accuracy'] = {'average': scores["test_b_acc"].mean()*100, 'std': scores["test_b_acc"].std()*100)}
        accuracy ['f1'] = {'average': scores["test_f1"].mean()*100, 'std': scores["test_f1"].std()*100)}
        accuracy ['precision'] = {'average': scores["test_prec"].mean()*100, 'std': scores["test_prec"].std()*100)}
        accuracy ['recall'] = {'average': scores["test_rec"].mean()*100, 'std': scores["test_rec"].std()*100)}
        accuracy ['auc'] = {'average': scores["test_auc"].mean()*100, 'std': scores["test_auc"].std()*100)}


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
        
        clf = GridSearchCV(model, params["cv"]["model_params"], scoring=scoring, cv=kfold, return_train_score=False,
                           refit='f1', verbose=3, n_jobs=-1)
        
        clf.fit(X,y)
        
        log_result_for_gridsearch(clf)
        
        return clf


