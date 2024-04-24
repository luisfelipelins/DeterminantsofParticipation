# -*- coding: utf-8 -*-
"""
Created on Wed Apr  3 12:17:30 2024

@author: lfval
"""

import os
import time
import pycps
import pickle
import pandas as pd
from tqdm import tqdm
from datetime import datetime as dt

os.environ['CENSUS_API_KEY'] = 'ea5c9a84ce17ee355946962d82be96d2685f7bc8'

def getVariablesList(year_begin):
    dict_asec = {}
    tqdm_obj = tqdm(range(year_begin,dt.now().year))
    for year in tqdm_obj:
        tqdm_obj.set_description(str(year))
        var_list = []
        for group in ['tags','variables','groups']:
            url      = f"http://api.census.gov/data/{year}/cps/asec/mar/{group}.json"
            temp_vars = [i.lower() for i in list(pd.read_json(url).index)]
            var_list = var_list + temp_vars
        dict_asec.update({year:var_list})
        
    return dict_asec

def downloadDataFromASEC(year_begin,tickers,vars_list,flags_all,col_names):
    dict_asec = {}
    tqdm_obj = tqdm(range(year_begin,dt.now().year))
    for year in tqdm_obj:
        tqdm_obj.set_description(str(year))
        
        temp_tickers = list(set([i for i in tickers if i in vars_list[year]]))
        
        data = []
        for t in temp_tickers:
            t_df = pycps.get_asec(year, [t])
            data.append(t_df)
            time.sleep(1)
        data = pd.concat(data,axis=1)
        
        data = data.apply(lambda x: pd.to_numeric(x, errors='coerce'))
       
        # Selecting only individuals older than 16y old
        data = data.loc[data['a_age'] >= 16]
        
        av_flags = [i for i in flags_all.index if i.lower() in temp_tickers]
        flags = flags_all.loc[av_flags]
         
        for i in flags.index:
            data = data.loc[data[i.lower()] <= flags.loc[i][0]]
        flags = [i.lower() for i in flags.index]
        data = data[[i for i in data.columns if i not in flags]]
        data = data.rename(columns = col_names)

        dict_asec.update({year:data})
    
    return dict_asec

if __name__ == '__main__':
    path = r'C:/Users/lfval/OneDrive\Documentos\FGV\Monografia\DeterminantsofParticipation'
    spec = pd.read_excel(f'{path}/config/spec.xlsx','asec')
    tickers = [x.lower() for x in spec['Ticker'].to_list()]
    
    flags_all = spec[['Ticker','Rename']].set_index('Ticker')
    flags_all['Rename'] = pd.to_numeric(flags_all['Rename'], errors='coerce')
    flags_all.dropna(inplace=True)
    
    col_names = spec[['Ticker','Rename']]
    col_names['Ticker'] = col_names['Ticker'].str.lower()
    col_names.set_index('Ticker',inplace=True)
    col_names = col_names.to_dict()['Rename']
    
    vars_list = getVariablesList(year_begin=1992)
    asec_data = downloadDataFromASEC(year_begin=1992, tickers=tickers, vars_list=vars_list, flags_all=flags_all, col_names=col_names)
    
    with open(f'{path}/Data/input/asec_raw.p','wb') as f:
        pickle.dump(asec_data,f)
