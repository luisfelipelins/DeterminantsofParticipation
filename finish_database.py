# -*- coding: utf-8 -*-
"""
Created on Thu Apr 11 11:23:06 2024

@author: lfval
"""

import os
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm

def getStateJOLTS(yearly_agg, simplified):
    path  = r'C:/Users/lfval/OneDrive\Documentos\FGV\Monografia\DeterminantsofParticipation'
    jolts = pd.read_csv(f'{path}/Data/input/jolts.txt', sep='\t') # Had to download it since BLS API doesn't allow request
    jolts.columns = [i.strip() for i in jolts.columns]
    jolts = jolts[['series_id','year','value']]
    
    match yearly_agg:
        
        case 'mean':
            jolts = jolts.groupby(['series_id','year']).mean().reset_index()
        case _:
            raise ValueError('This yearly aggregation is not available.')
    
    jolts['seas']     = jolts['series_id'].str[2:3]
    jolts['industry'] = jolts['series_id'].str[3:9]
    jolts['state']    = jolts['series_id'].str[9:11]
    jolts['s_class']  = jolts['series_id'].str[16:18]
    jolts['data_elm'] = jolts['series_id'].str[18:20]
    jolts['rate']     = jolts['series_id'].str[20]
    
    # Using only NSA data since CPS is not SA
    jolts = jolts.loc[jolts['seas'] == 'U'].drop(['seas','series_id'],axis=1)
    
    # Taking only rates since level is not comparable between states
    jolts = jolts.loc[jolts['rate'] == 'R'].drop('rate',axis=1)
    
    # Excluding "Other separations" variable since it's only available for national level
    jolts = jolts.loc[jolts['data_elm'] != 'OS']
    
    # We could match by industry and firm size, but by simplicity we shall work with all industries and sizes
    if simplified:
        jolts = jolts.loc[jolts['industry'] == '000000'].drop('industry',axis=1)
        jolts = jolts.loc[jolts['s_class'] == '00'].drop('s_class',axis=1)
    
    # Keeping only state level JOLTS
    jolts['state'] = pd.to_numeric(jolts['state'],errors='coerce')
    jolts = jolts.loc[jolts['state'] != 0].dropna()
    
    # Pivoting the dataframe to keep variables in the columns
    jolts = pd.pivot_table(jolts, values='value',index=['year','state'],columns='data_elm')
    
    years = list(jolts.index.get_level_values(0).unique())
    
    jolts_data = {}
    for year in years:
        temp = jolts.loc[year].reset_index()
        jolts_data.update({year:temp})
    
    return jolts_data

def getCountyUnemployment():
    path = r'C:\Users\lfval\OneDrive\Documentos\FGV\Monografia\DeterminantsofParticipation\Data\input\County Level Unemployment'
    files = os.listdir(path)
    
    county_unemployment = {}
    for file in tqdm(files):
        temp = pd.read_excel(f'{path}/{file}', header=4)[['Code.1','Code.2','Year','Force','Unemployed']].dropna(how='all')
        temp = temp.apply(lambda x: pd.to_numeric(x,errors='coerce'))
        temp.rename(columns={'Code.1':'State',
                             'Code.2':'County'},inplace=True)
        temp['County_URate'] = temp['Unemployed'].div(temp['Force'])
        temp = temp.drop(['Force','Unemployed'],axis=1)
        temp['fipscode'] = temp['State'].astype(int).astype(str) + temp['County'].astype(int).astype(str)
        county_unemployment.update({temp['Year'].astype(int).iloc[0]:temp[['fipscode','County_URate']]})
    
    return county_unemployment

def getCountyHealthData():
    path  = r'C:\Users\lfval\OneDrive\Documentos\FGV\Monografia\DeterminantsofParticipation\Data\input\County Level Health Stats'
    files = os.listdir(path)
    
    health_data = {}
    for file in tqdm(files):
        temp = pd.read_csv(f'{path}/{file}', header=1)
        cols = ['statecode','countycode','v001_rawvalue','v011_rawvalue','v049_rawvalue','v014_rawvalue','v155_rawvalue','v136_rawvalue','v137_rawvalue','v138_rawvalue','v015_rawvalue']
        cols = [i for i in cols if i in temp.columns]
        temp = temp[cols]
        temp['fipscode'] = temp['statecode'].astype(str) + temp['countycode'].astype(str)
        temp = temp.loc[temp['fipscode'] != '00']
        health_data.update({int(file[-8:-4]):temp[['fipscode']+cols[2:]]})
    
    return health_data

def addDatasettoASEC(asec_dict,dataset_dict,geo_level):
    new_asec_dict = asec_dict
    
    asec_years = set(new_asec_dict.keys())
    data_years = set(dataset_dict.keys())
    comm_years = list(asec_years.intersection(data_years))
    
    for year in comm_years:
        data = dataset_dict[year]
        asec = new_asec_dict[year]
        
        match geo_level:
            case 'state':
                new_asec = pd.merge(data, asec, on='state', how='inner')
            case 'county':
                if year not in [2001,2002,2003,2004]:
                    new_asec = pd.merge(data, asec, on='fipscode', how='inner')
            case _:
                raise ValueError(f'{geo_level} geo level not supported.')
        if year not in [2001,2002,2003,2004]:
            new_asec_dict[year] = new_asec
        
    return new_asec_dict
            
def adjustingASECDict(asec_dict):
    
    # Adjusting asec_dict
    for i in tqdm(asec_dict):
        temp = asec_dict[i].apply(lambda x: pd.to_numeric(x,errors='coerce'))
        cols = temp.columns
        
        # Selecting only individuals with county available & creating FIPS code county
        if 'county' in cols:
            temp = temp.loc[temp['county'] != 0]
            temp['fipscode'] = temp['state'].astype(int).astype(str) + temp['county'].astype(int).astype(str)
        
        # Creating US national/non-US national variable
        if 'citizenship' in cols:
            temp['citizenship'] = np.where(temp['citizenship'] <= 3, 1,0)
        
        # Creating education variables
        if 'educ' in cols:
            temp['educ_hs'] = np.where(temp['educ'] == 39, 1,0)
            temp['educ_college'] = np.where((temp['educ'] >= 40) & (temp['educ'] <= 43), 1,0)
            temp['educ_post_grad']  = np.where(temp['educ'] >= 44, 1,0)
        
        # Creating categorical industry variable
        if 'industry' in cols:
            temp = pd.concat([temp,pd.get_dummies(temp['industry'], prefix='industry').astype(int)],axis=1).drop('industry',axis=1)
    
        # Create born country
        if 'origin_country' in cols:
            temp['origin_country'] = np.where(temp['origin_country'] == 57,1,0)
        if 'origin_country_father' in cols:
            temp['origin_country_father'] = np.where(temp['origin_country_father'] == 57,1,0)
        if 'origin_country_mother' in cols:
            temp['origin_country_mother'] = np.where(temp['origin_country_mother'] == 57,1,0)

        # Create white/non-white variable
        if 'race' in cols:
            temp['race'] = np.where(temp['race'] == 1,1,0)
        
        # Creating served in the military variable
        if 'veteran_status' in cols:
            temp['veteran_status'] = np.where(temp['veteran_status'] == 1,1,0)

        # Create family type categorical variable
        if 'family_type_marriage' in cols:
            temp.dropna(subset='family_type_marriage',inplace=True)
            temp = pd.concat([temp,pd.get_dummies(temp['family_type_marriage'].astype(int), prefix='family_type_marriage').astype(int)],axis=1).drop('family_type_marriage',axis=1)
    
        # Create married/non-married variable
        if 'marriage_status' in cols:
            temp['marriage_status'] = np.where(temp['marriage_status'] <= 3,1,0)
        
        # Adjusting variables with values -1, transforming into zero
        if 'child_care_cost' in cols:
            temp['child_care_cost'] = np.where(temp['child_care_cost'] < 0,0,temp['child_care_cost'])
    
        # Creating labour force status variable
        if 'lf_status' in cols:
            temp['lf_status'] = np.where(temp['lf_status'] != 7,1,0)
        
        asec_dict[i] = temp
    
if __name__ == '__main__':
    path = r'C:/Users/lfval/OneDrive\Documentos\FGV\Monografia\DeterminantsofParticipation'
    with open(f'{path}/Data/input/asec_raw.p','rb') as f:
        asec_dict = pickle.load(f)
        
    adjustingASECDict(asec_dict)
    
    jolts_dict  = getStateJOLTS('mean',True)
    unemp_dict  = getCountyUnemployment()
    health_dict = getCountyHealthData()
    
    asec_dict = addDatasettoASEC(asec_dict,jolts_dict ,'state')
    asec_dict = addDatasettoASEC(asec_dict,unemp_dict ,'county')
    asec_dict = addDatasettoASEC(asec_dict,health_dict,'county')
    
    
