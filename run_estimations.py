# -*- coding: utf-8 -*-
"""
Created on Fri May 24 22:14:40 2024

@author: lfval
"""

import pickle
import pandas as pd
from tqdm import tqdm
from statsmodels.discrete.discrete_model import Probit
from concurrent.futures import ThreadPoolExecutor, as_completed

def importEstimationDataset(path):
    with open(f'{path}/Data/input/data_final.p','rb') as f:
        data_dict = pickle.load(f)
    
    # We will drop the dummies with multiple NAs
    for d in data_dict:
        data_dict[d] = data_dict[d].dropna(axis=1)
        
    # Now dropping unuseful variables such as fipscode & state
    for d in data_dict:
        to_exc_cols = ['fipscode','state','industry']
        data_dict[d] = data_dict[d].drop(to_exc_cols,axis=1,errors='ignore')
        
    return data_dict

def fitSimpleProbit(year, data):
    probit = Probit(data['lf_status'], data.drop('lf_status', axis=1)).fit(maxiter=250)
    return year, probit

def createProbitEstTimeSeries(data_dict,probit_res,var):
    ts = {}
    for y in probit_res:
        if var in data_dict[y].columns:
            res = probit_res[y]
            est = res.params[var]
            lb  = res.conf_int().loc[var][0]
            ub  = res.conf_int().loc[var][1]
            res = {'Estimate':est,
                   'LB':lb,
                   'UB':ub}
            ts.update({y:res})
        else: pass
    ts = pd.DataFrame(ts).T.sort_index()
    ts.index = pd.to_datetime(ts.index,format='%Y')+pd.offsets.YearEnd()
    
    return ts

def retLatexProbitEst(probit_res,year):
    summ = probit_res[year].summary()
    lat_str = summ.as_latex()
    
    return lat_str

if __name__ == '__main__':
    path = r'C:/Users/lfval/OneDrive\Documentos\FGV\Monografia\DeterminantsofParticipation'
    data_dict = importEstimationDataset(path)
    variables = list(data_dict[2023].columns)
    
    # Simple Probit Estimations
    probit_res = {}
    with ThreadPoolExecutor() as executor:
        futures = {executor.submit(fitSimpleProbit, year, data_dict[year]): year for year in data_dict}
        for future in tqdm(as_completed(futures), total=len(futures)):
            year, probit = future.result()
            probit_res[year] = probit
            del probit
   
    lat_str = retLatexProbitEst(probit_res,2023)    
        

    # probit = Probit(data_dict[2023]['lf_status'],data_dict[2023].drop('lf_status',axis=1)).fit(maxiter=200)
    # probit.summary()
