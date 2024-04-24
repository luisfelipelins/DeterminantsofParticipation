# -*- coding: utf-8 -*-
"""
Created on Thu Apr 11 11:23:06 2024

@author: lfval
"""

import pickle
import requests
import pandas as pd

def getJOLTSStateData(yearly_agg):
    path  = r'C:/Users/lfval/OneDrive\Documentos\FGV\Monografia\DeterminantsofParticipation'
    jolts = pd.read_csv(f'{path}/Data/input/jolts.txt', sep='\t')
    jolts.columns = [i.strip() for i in jolts.columns]
    jolts = jolts[['series_id','year','value']]
    
    match yearly_agg:
        
        case 'mean':
            jolts = jolts.groupby(['series_id','year']).mean().reset_index()
        case _:
            raise ValueError('This yearly aggregation is not available.')
    
    
    survey   = jolts['series_id'].str[:2]
    seas     = jolts['series_id'].str[2:3]
    industry = jolts['series_id'].str[3:9]
    jolts['series_id'].str[9:11]
    jolts['series_id'].str[11:16]
    jolts['series_id'].str[16:18]
    jolts['series_id'].str[18:20]


if __name__ == '__main__':
    path = r'C:/Users/lfval/OneDrive\Documentos\FGV\Monografia\DeterminantsofParticipation'
    with open(f'{path}/Data/input/asec_raw.p','rb') as f:
        asec_dict = pickle.load(f)
    
    asec_dict
