# -*- coding: utf-8 -*-
"""
Created on Fri Jun 14 17:26:27 2024

@author: luis.lins
"""

"""
Set of candidate models used in the estimations
"""

import sklearn.metrics
import numpy as np
from sklearn import svm
from sklearn import linear_model
from sklearn import ensemble
from sklearn.model_selection import cross_val_score, train_test_split, KFold, cross_validate
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score
import xgboost
import catboost
import optuna
import time

# Defining auxiliary functions

def _accuracy(estimator, X, y):
    predictions = estimator.predict(X)
    rounded_predictions = np.where(predictions > 0.5, 1, 0)
    return accuracy_score(y, rounded_predictions)

def _cfs_matrix(estimator, X, y):
    predictions = estimator.predict(X)
    rounded_predictions = np.where(predictions > 0.5, 1, 0)
    cm = confusion_matrix(y, rounded_predictions)
    return {'tn': cm[0, 0]/len(X.index), 'fp': cm[0, 1]/len(X.index),
            'fn': cm[1, 0]/len(X.index), 'tp': cm[1, 1]/len(X.index)}

def _auc_roc(estimator, X, y):
    predictions = estimator.predict(X)
    rounded_predictions = np.where(predictions > 0.5, 1, 0)
    roc_auc = roc_auc_score(y, rounded_predictions)
    return roc_auc


class MLModelsEstimation:
    
    def __init__(self,X,y,test_size,random_seed=87):
        
        X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=test_size,random_state=random_seed)
        
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test

    def OLS(self, params):
    
        mod = linear_model.LinearRegression(**params)
        mod.fit(self.X_train,self.y_train)

        return mod
    
    def Logit(self,params):

        mod = linear_model.LogisticRegression( **params)
        mod.fit(self.X_train,self.y_train)

        return mod
    
    def SVM(self,params):

        mod = svm.SVR(**params)
        mod.fit(self.X_train,self.y_train)

        return mod
    
    def RandomForestRegressor(self,params):

        mod = ensemble.RandomForestRegressor(**params)
        mod.fit(self.X_train,self.y_train)

        return mod
    
    def XGBoost(self, params):
        
        mod = xgboost.XGBRegressor(**params)
        mod.fit(self.X_train,self.y_train)

        return mod
    
    def XGBoostRF(self, params):

        mod = xgboost.XGBRFRegressor(**params)
        mod.fit(self.X_train,self.y_train)

        return mod
    
    def CatBoost(self, params):

        mod = catboost.CatBoostRegressor(**params)
        mod.fit(self.X_train,self.y_train)

        return mod


class MLModelsCV:
    
    def __init__(self, X,y,test_size,random_seed=87):
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=random_seed)
        
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.random_seed = random_seed

    def OLS(self, trial):
    
        params = {'fit_intercept':trial.suggest_categorical('fit_intercept',[True,False]),
                  'random_state':self.random_seed}
        
        mod = linear_model.LinearRegression(**params)
        mod.fit(self.X_train,self.y_train)

        preds = mod.predict(self.X_test)
        auc = roc_auc_score(self.y_test, preds)
        
        return auc
    
    def Logit(self,trial):
        
        params = {'fit_intercept':trial.suggest_categorical('fit_intercept',[True,False]),
                  'penalty':trial.suggest_categorical('penalty',[None,'l2']),
                  'C'      :trial.suggest_float('C',0,5),
                  'random_state':self.random_seed}
        
        mod = linear_model.LogisticRegression( **params)
        mod.fit(self.X_train,self.y_train)

        preds = mod.predict(self.X_test)
        auc = roc_auc_score(self.y_test, preds)
        
        return auc
    
    def SVM(self,trial):
    
        params = {'kernel':'linear',
                  'C':trial.suggest_float('C',0,5),
                  'random_state':self.random_seed}

        mod = svm.SVR(**params)
        mod.fit(self.X_train,self.y_train)

        preds = mod.predict(self.X_test)
        auc = roc_auc_score(self.y_test, preds)
        
        return auc
    
    def RandomForestRegressor(self,trial):
        
        params = {'n_estimators':trial.suggest_int('n_estimators',50,200),
                  'max_depth':trial.suggest_categorical('max_depth',[None,100,200]),
                  'min_samples_leaf':trial.suggest_int('min_samples_leaf',2,5),
                  'max_features':trial.suggest_int('max_features',1,3),
                  'random_state':self.random_seed}

        mod = ensemble.RandomForestRegressor(**params)
        mod.fit(self.X_train,self.y_train)

        preds = mod.predict(self.X_test)
        auc = roc_auc_score(self.y_test, preds)
        
        return auc
    
    def XGBoost(self, trial):
        
        params = {"verbosity": 0,
        "objective": "binary:logistic",
        "tree_method": "exact",
        "booster": trial.suggest_categorical("booster", ["gbtree", "gblinear", "dart"]),
        "lambda": trial.suggest_float("lambda", 1e-8, 1.0, log=True),
        "alpha": trial.suggest_float("alpha", 1e-8, 1.0, log=True),
        "subsample": trial.suggest_float("subsample", 0.2, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.2, 1.0),
        'random_state':self.random_seed}

        if params["booster"] in ["gbtree", "dart"]:
            params["max_depth"] = trial.suggest_int("max_depth", 3, 9, step=2)
            params["min_child_weight"] = trial.suggest_int("min_child_weight", 2, 10)
            params["eta"] = trial.suggest_float("eta", 1e-8, 1.0, log=True)
            params["gamma"] = trial.suggest_float("gamma", 1e-8, 1.0, log=True)
            params["grow_policy"] = trial.suggest_categorical("grow_policy", ["depthwise", "lossguide"])
    
        if params["booster"] == "dart":
            params["sample_type"] = trial.suggest_categorical("sample_type", ["uniform", "weighted"])
            params["normalize_type"] = trial.suggest_categorical("normalize_type", ["tree", "forest"])
            params["rate_drop"] = trial.suggest_float("rate_drop", 1e-8, 1.0, log=True)
            params["skip_drop"] = trial.suggest_float("skip_drop", 1e-8, 1.0, log=True)
            
        mod = xgboost.XGBRegressor(**params)
        mod.fit(self.X_train,self.y_train)

        preds = mod.predict(self.X_test)
        auc = roc_auc_score(self.y_test, preds)
        
        return auc
    
    def XGBoostRF(self, trial):
        
        params = {"colsample_bynode": trial.suggest_float("colsample_bynode", 0.1, 1, log=True),
                  "learning_rate": trial.suggest_float("learning_rate", 0.1, 1, log=True),
                  "max_depth": trial.suggest_int("max_depth", 2, 10),
                  "num_parallel_tree": trial.suggest_int("num_parallel_tree", 50, 150),
                  "objective": "binary:logistic",
                  "device": "cuda",
                  'random_state':self.random_seed}

        mod = xgboost.XGBRFRegressor(**params)
        mod.fit(self.X_train,self.y_train)

        preds = mod.predict(self.X_test)
        auc = roc_auc_score(self.y_test, preds)
        
        return auc
    
    def CatBoost(self, trial):
        
        params = {"learning_rate": trial.suggest_float("learning_rate", 0.1, 1, log=True),
                  "depth": trial.suggest_int("depth", 2, 10),
                  'silent':True,
                  'random_state':self.random_seed}

        mod = catboost.CatBoostRegressor(**params)
        mod.fit(self.X_train,self.y_train)

        preds = mod.predict(self.X_test)
        auc = roc_auc_score(self.y_test, preds)
        
        return auc
    
    def optimizeOptuna(self, method_name, n_trials=100, timeout=600):
        
        method = getattr(self, method_name)

        start = time.time()
        study = optuna.create_study(direction="maximize")
        study.optimize(method, n_trials=100, timeout=600)
        
        print('')
        print('Best parameters: ', study.best_params)
        print('Best AUC: ', study.best_value)
        print(f'Time elapsed: {time.time()-start}s')
        
        best_params = study.best_params
        
        return best_params
    
class MLModelsKFoldCV:
    
    def __init__(self,X,y,n_folds,random_seed=87):
        
        self.X = X
        self.y = y
        self.n_folds = n_folds

    def OLS(self, params):
    
        mod = linear_model.LinearRegression(**params)
        
        accuracy   = cross_validate(mod, self.X, self.y, cv=self.n_folds,scoring=_accuracy)
        cfs_matrix = cross_validate(mod, self.X, self.y, cv=self.n_folds,scoring=_cfs_matrix)
        roc_auc    = cross_validate(mod, self.X, self.y, cv=self.n_folds,scoring=_auc_roc)

        kfold_cv_res = {'accuracy':accuracy,
                        'cfs_matrix':cfs_matrix,
                        'roc_auc':roc_auc}

        return kfold_cv_res
    
    def Logit(self,params):

        mod = linear_model.LogisticRegression( **params)
        
        accuracy   = cross_validate(mod, self.X, self.y, cv=self.n_folds,scoring=_accuracy)
        cfs_matrix = cross_validate(mod, self.X, self.y, cv=self.n_folds,scoring=_cfs_matrix)
        roc_auc    = cross_validate(mod, self.X, self.y, cv=self.n_folds,scoring=_auc_roc)

        kfold_cv_res = {'accuracy':accuracy,
                        'cfs_matrix':cfs_matrix,
                        'roc_auc':roc_auc}

        return kfold_cv_res
    
    def SVM(self,params):

        mod = svm.SVR(**params)
        
        accuracy   = cross_validate(mod, self.X, self.y, cv=self.n_folds,scoring=_accuracy)
        cfs_matrix = cross_validate(mod, self.X, self.y, cv=self.n_folds,scoring=_cfs_matrix)
        roc_auc    = cross_validate(mod, self.X, self.y, cv=self.n_folds,scoring=_auc_roc)

        kfold_cv_res = {'accuracy':accuracy,
                        'cfs_matrix':cfs_matrix,
                        'roc_auc':roc_auc}

        return kfold_cv_res
    
    def RandomForestRegressor(self,params):

        mod = ensemble.RandomForestRegressor(**params)
        
        accuracy   = cross_validate(mod, self.X, self.y, cv=self.n_folds,scoring=_accuracy)
        cfs_matrix = cross_validate(mod, self.X, self.y, cv=self.n_folds,scoring=_cfs_matrix)
        roc_auc    = cross_validate(mod, self.X, self.y, cv=self.n_folds,scoring=_auc_roc)

        kfold_cv_res = {'accuracy':accuracy,
                        'cfs_matrix':cfs_matrix,
                        'roc_auc':roc_auc}

        return kfold_cv_res
    
    def XGBoost(self, params):
        
        mod = xgboost.XGBRegressor(**params)
        
        accuracy   = cross_validate(mod, self.X, self.y, cv=self.n_folds,scoring=_accuracy)
        cfs_matrix = cross_validate(mod, self.X, self.y, cv=self.n_folds,scoring=_cfs_matrix)
        roc_auc    = cross_validate(mod, self.X, self.y, cv=self.n_folds,scoring=_auc_roc)

        kfold_cv_res = {'accuracy':accuracy,
                        'cfs_matrix':cfs_matrix,
                        'roc_auc':roc_auc}

        return kfold_cv_res
    
    def XGBoostRF(self, params):

        mod = xgboost.XGBRFRegressor(**params)
        
        accuracy   = cross_validate(mod, self.X, self.y, cv=self.n_folds,scoring=_accuracy)
        cfs_matrix = cross_validate(mod, self.X, self.y, cv=self.n_folds,scoring=_cfs_matrix)
        roc_auc    = cross_validate(mod, self.X, self.y, cv=self.n_folds,scoring=_auc_roc)

        kfold_cv_res = {'accuracy':accuracy,
                        'cfs_matrix':cfs_matrix,
                        'roc_auc':roc_auc}

        return kfold_cv_res
    
    def CatBoost(self, params):

        mod = catboost.CatBoostRegressor(**params)
        
        accuracy   = cross_validate(mod, self.X, self.y, cv=self.n_folds,scoring=_accuracy)
        cfs_matrix = cross_validate(mod, self.X, self.y, cv=self.n_folds,scoring=_cfs_matrix)
        roc_auc    = cross_validate(mod, self.X, self.y, cv=self.n_folds,scoring=_auc_roc)

        kfold_cv_res = {'accuracy':accuracy,
                        'cfs_matrix':cfs_matrix,
                        'roc_auc':roc_auc}

        return kfold_cv_res