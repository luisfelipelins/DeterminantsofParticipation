# -*- coding: utf-8 -*-
"""
Created on Thu May 16 20:39:26 2024

@author: lfval
"""

import numpy as np
import geopandas
import geodatasets
import pickle
import pandas as pd
import plotly as py
import plotly.figure_factory as ff
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.ticker as mtick
import seaborn as sns
from statsmodels.stats.descriptivestats import describe
from tqdm import tqdm
from fred import Fred
import ast
from scipy.stats import pearsonr
from statsmodels.regression.linear_model import OLS
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz
from sklearn.model_selection import train_test_split

def importFred(fr,ticker,name):
    data = fr.series.observations(ticker)
    data = [pd.Series(i) for i in ast.literal_eval(data)['observations']]
    data = pd.concat(data,axis=1).T
    data['date']  = pd.to_datetime(data['date'])
    data['value'] = pd.to_numeric(data['value'], errors='coerce')
    data = data.set_index('date')['value'].rename(name).dropna()
    
    return data

def calcParticipationbyCounty(data_dict):
    return_df = []
    for df in data_dict:
        if df not in [2001,2002,2003,2004]:
            temp = data_dict[df].copy()
            county_sum = temp.groupby('fipscode')['ind_weight'].sum().to_dict()
            temp['county_sum'] = temp['fipscode'].map(county_sum)
            temp['part_cont'] = (temp['lf_status']*temp['ind_weight']).div(temp['county_sum'])
            part_county = temp.groupby('fipscode')['part_cont'].sum().rename(df)
            return_df.append(part_county)
    return_df = pd.concat(return_df,axis=1)

    return return_df

def calcCorrelationtoParticipation(data_dict,var):
    return_df = {}    
    for df in data_dict:
        if var in data_dict[df].columns:
            corr = pearsonr(data_dict[df]['lf_status'],data_dict[df][var])
            val  = corr.correlation
            lb   = corr.confidence_interval(confidence_level=0.95)[0]
            ub   = corr.confidence_interval(confidence_level=0.95)[1]
            corr = {'Correlation':val,'LB':lb,'UB':ub}
            return_df.update({df:corr})
        else: pass
    return_df = pd.DataFrame.from_dict(return_df)
    return return_df
    
def getCountyPolygons():
    county_polyg = geopandas.read_file(geodatasets.get_path('geoda.us_sdoh'))
    county_polyg['fipscode'] = county_polyg['state_fips'] + county_polyg['cnty_fips']
    
    return county_polyg

def plotCountyMapData(data,year,polyg):
    temp = data[[year]].dropna().reset_index()
    temp = temp.set_index('fipscode')[year].to_dict()
    
    # state = {i[:2]:temp[i] for i in temp if i[-3:] == '000'}
    
    polyg_new = polyg.copy()
    polyg_new['data'] = polyg['fipscode'].map(temp)
    # polyg_new['data'] = polyg_new['data'].fillna(polyg['state_fips'].map(state))
    
    return polyg_new


if __name__ == '__main__':
    path = r'C:/Users/lfval/OneDrive\Documentos\FGV\Monografia\DeterminantsofParticipation'
    fr = Fred(api_key=open(f'{path}/config/fred_api_key.txt','r').read(),response_type='json')
    with open(f'{path}/Data/input/data_final.p','rb') as f:
        data_dict = pickle.load(f)
        
    desc_stats_dict = {}
    for y in tqdm(data_dict):
        desc_stats_dict.update({y:describe(data_dict[y])})
        
    ##################
    ## Correlations ##
    ##################
    job_openings = calcCorrelationtoParticipation(data_dict,'JO').T
    origin_cnty  = calcCorrelationtoParticipation(data_dict,'origin_country').T
    female       = calcCorrelationtoParticipation(data_dict,'female').T
    soc_secrty   = calcCorrelationtoParticipation(data_dict,'d@social_security').T
    overdose     = calcCorrelationtoParticipation(data_dict,'v138_rawvalue').T
    age          = calcCorrelationtoParticipation(data_dict,'age').T
    
    fig,ax=plt.subplots(ncols=3,nrows=2,figsize=(14,5))
    
    i,j=(0,0)
    for df,name in zip([job_openings,origin_cnty,female,soc_secrty,overdose,age],['Job Openings','American National','Female Dummy','Social Security Income','Overdose Deaths','Age']):
        df.index = pd.to_datetime(df.index,format='%Y')+pd.offsets.YearEnd()
        ax[i,j].plot(df['Correlation'],color='darkblue',marker='.',label='Pearson Correlation')
        ax[i,j].fill_between(x=df.index,y1=df['LB'],y2=df['UB'],color='darkblue',alpha=0.2,label='95% C.I.')
        if (i==0) & (j==0):
            ax[i,j].legend(loc=0)
        ax[i,j].yaxis.set_major_formatter(mtick.PercentFormatter(1))
        ax[i,j].grid(linestyle=':')
        ax[i,j].set_title(name,fontweight='bold')
        ax[i,j].axhline(0,color='grey',linestyle='--')
        
        if j>1:
            i+=1
            j=0
        else:
            j+=1
    plt.tight_layout()
    
    ####################
    ## Income Sources ##
    ####################
    pce_prices      = importFred(fr,'PCEPI','PCE Prices')/100
    
    personal_income = importFred(fr,'PI','Personal Income').div(pce_prices)
    comp_employees  = importFred(fr,'W209RC1','Compensation of Employees').div(pce_prices).div(personal_income)
    inc_assets      = importFred(fr,'PIROA','Income from Assets').div(pce_prices).div(personal_income)
    inc_receipts    = importFred(fr,'PCTR','Income from Current Transfer').div(pce_prices).div(personal_income)
   
    fig,ax=plt.subplots(ncols=3,figsize=(14,5))
    ax[0].plot(comp_employees,color='darkblue',label='Compensation of Employees')
    ax[0].set_title('Compensation of Employees',fontweight='bold',loc='left')
    ax[0].yaxis.set_major_formatter(mtick.PercentFormatter(1))
    ax[0].grid(linestyle='--')
    ax[1].plot(inc_assets,color='darkgreen',label='Income from Assets')
    ax[1].set_title('Income from Assets',fontweight='bold',loc='left')
    ax[1].yaxis.set_major_formatter(mtick.PercentFormatter(1))
    ax[1].grid(linestyle='--')
    ax[2].plot(inc_receipts,color='red')
    ax[2].set_title('Income from Current Transfers',fontweight='bold',loc='left')
    ax[2].yaxis.set_major_formatter(mtick.PercentFormatter(1))
    ax[2].grid(linestyle='--')
    
    ###########################
    ## Fit Illustrative Tree ##
    ###########################
    y2023 = data_dict[2023].drop(['fipscode','ind_weight','industry'],axis=1)
    X = y2023.drop('lf_status',axis=1)
    y = y2023['lf_status']
    tree_reg = DecisionTreeClassifier(max_depth=2).fit(X,y)
    tree_reg.score(X,y)
    
    export_graphviz(tree_reg,feature_names=X.columns,class_names=True,out_file='tree.dot')
    # Use dot -Tpng tree.dot -o tree.png in cmd to generate .png
    
    #############################################
    ## Soc Sec & Dividends Income Distribution ##
    #############################################
    y2007 = data_dict[2007][['hh_income_social_security','hh_income_dividends']]
    y2010 = data_dict[2010][['hh_income_social_security','hh_income_dividends']]
    y2019 = data_dict[2019][['hh_income_social_security','hh_income_dividends']]
    y2023 = data_dict[2023][['hh_income_social_security','hh_income_dividends']]
    
    fig,ax=plt.subplots(nrows=2,figsize=(14,7))
    sns.kdeplot(y2007['hh_income_social_security'], fill=True,ax=ax[0], label='2007')
    sns.kdeplot(y2010['hh_income_social_security'], fill=True,ax=ax[0], label='2010')
    sns.kdeplot(y2019['hh_income_social_security'], fill=True,ax=ax[0], label='2019')
    sns.kdeplot(y2023['hh_income_social_security'], fill=True,ax=ax[0], label='2023')
    ax[0].legend(loc=0)
    ax[0].set_xlim([-10000,10000])
    ax[0].set_title('Income from Social Security',fontweight='bold',loc='left')
    ax[0].set_xlabel('')
    
    sns.kdeplot(y2007['hh_income_dividends'], fill=True,ax=ax[1], label='2007')
    sns.kdeplot(y2010['hh_income_dividends'], fill=True,ax=ax[1], label='2010')
    sns.kdeplot(y2019['hh_income_dividends'], fill=True,ax=ax[1], label='2019')
    sns.kdeplot(y2023['hh_income_dividends'], fill=True,ax=ax[1], label='2023')
    ax[1].legend(loc=0)
    ax[1].set_xlim([-10000,10000])
    ax[1].set_title('Income from Dividends',fontweight='bold',loc='left')
    ax[1].set_xlabel('')
    plt.tight_layout()
    
        
    #####################################
    ## US Aggregate Participation Rate ##
    #####################################
    data  = importFred(fr,'CIVPART','US Participation Rate')/100
    men   = importFred(fr,'LNS11300001','Males (All ages)')/100
    women = importFred(fr,'LNS11300002','Females (All ages)')/100
    
    fig,ax=plt.subplots(ncols=2,figsize=(14,5))
    ax[0].plot(data, color = 'black',zorder=10)
    ax[0].grid(linestyle='--')
    ax[0].yaxis.set_major_formatter(mtick.PercentFormatter(1))
    ax[0].set_title(f'US Monthly Agg. Part. Rate',fontweight='bold',loc='left')
    
    ax[1].plot(men, color = 'darkblue',zorder=10, label = 'Males (All ages)')
    ax[1].plot(women, color = 'red',zorder=10, label = 'Females (All ages)')
    ax[1].grid(linestyle='--')
    ax[1].yaxis.set_major_formatter(mtick.PercentFormatter(1))
    ax[1].set_title(f'Part. Rate by Sex',fontweight='bold',loc='left')
    ax[1].legend(loc=0)
    
    #####################################
    ## Participation Rate by Age Group ##
    #####################################
    lfpr1619 = importFred(fr,'LNS11300012','16-19y')/100
    lfpr2024 = importFred(fr,'LNS11300036','20-24y')/100
    lfpr2554 = importFred(fr,'LNS11300060','25-54y')/100
    lfpr55   = importFred(fr,'LNS11324230','55+y')/100
    
    fig,ax=plt.subplots(ncols=1,figsize=(14,5))
    ax.plot(lfpr1619, color = 'darkblue', label = '16-19y')
    ax.plot(lfpr2024, color = 'lightblue', label = '20-24y')
    ax.plot(lfpr2554, color = 'red', label = '25-54y')
    ax.plot(lfpr55, color = 'green', label = '55+y')
    ax.grid(linestyle='--')
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1))
    ax.legend(loc=0)
    ax.set_title(f'Participation Rate by Age Group',fontweight='bold',loc='left')
    
    #############################################
    ## Ploting overall participation evolution ##
    #############################################

    df_dict = {}
    for df in data_dict:
            temp = data_dict[df].copy()
            temp['sumweight'] = temp['ind_weight'].sum()
            temp['part_cont'] = (temp['lf_status']*temp['ind_weight']).div(temp['sumweight'])
            part = temp['part_cont'].sum()
            df_dict.update({df:part})
    df = pd.Series(df_dict).loc['2006':]
    df.index = pd.to_datetime(df.index,format='%Y')+pd.offsets.YearEnd()
    
    fig,ax=plt.subplots(figsize=(10,5))
    ax.plot(df,color='black',marker='D',zorder=10)
    ax.grid(linestyle='--')    
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1))
    ax.set_title(f'US Aggregate Participation Rate',fontweight='bold',loc='left')
    
    ############################################
    ## Plot Participation by County/State Map ##
    ############################################
    part_by_county = calcParticipationbyCounty(data_dict)
    polyg = getCountyPolygons()
    
    # 2023
    year = 2023
    geodata = plotCountyMapData(part_by_county,year,polyg)
    geodata['data'] = np.where(geodata['data']<0.4,'< 40%',
                               np.where(geodata['data']<0.6,'40% < Part. Rate < 60%',
                                        np.where(geodata['data']<0.7,'60% < Part. Rate < 70%',
                                                 np.where(geodata['data']<0.8,'70% < Part. Rate < 80%',
                                                          np.where(geodata['data']<0.9,'80% < Part. Rate < 90%',
                                                                   np.where(geodata['data'].isnull(),'Not Available','> 90%'))))))

    custom_colors = {'< 40%': '#db1616',
    '40% < Part. Rate < 60%': '#de5b5b',
    '60% < Part. Rate < 70%': '#e08b8b',
    '70% < Part. Rate < 80%': '#8fe08b',
    '80% < Part. Rate < 90%': '#69db63',
    '> 90%': '#1fd916',
    'Not Available': '#a7aba8'}
    
    geodata['color'] = geodata['data'].map(custom_colors)

    fig,ax=plt.subplots(figsize=(10,5))
    for county, color in custom_colors.items():
        subset = geodata[geodata['data'] == county]
        subset.plot(color=color, ax=ax, label=county)

    legend_elements = [Line2D([0], [0], marker='o', color='w', label=county, 
                              markerfacecolor=color, markersize=10) for county, color in custom_colors.items()]
    ax.legend(handles=legend_elements, loc='lower left', fontsize=8, title='Part. Rate', title_fontsize=10)
    ax.set_title(f'Participation Rate by County in {year}',fontweight='bold',loc='left')
    plt.tick_params(left = False, right = False , labelleft = False , 
                labelbottom = False, bottom = False) 
    plt.tight_layout()
    
    # 2019
    year = 2019
    geodata = plotCountyMapData(part_by_county,year,polyg)
    geodata['data'] = np.where(geodata['data']<0.4,'< 40%',
                               np.where(geodata['data']<0.6,'40% < Part. Rate < 60%',
                                        np.where(geodata['data']<0.7,'60% < Part. Rate < 70%',
                                                 np.where(geodata['data']<0.8,'70% < Part. Rate < 80%',
                                                          np.where(geodata['data']<0.9,'80% < Part. Rate < 90%',
                                                                   np.where(geodata['data'].isnull(),'Not Available','> 90%'))))))

    custom_colors = {'< 40%': '#db1616',
    '40% < Part. Rate < 60%': '#de5b5b',
    '60% < Part. Rate < 70%': '#e08b8b',
    '70% < Part. Rate < 80%': '#8fe08b',
    '80% < Part. Rate < 90%': '#69db63',
    '> 90%': '#1fd916',
    'Not Available': '#a7aba8'}
    
    geodata['color'] = geodata['data'].map(custom_colors)

    fig,ax=plt.subplots(figsize=(10,5))
    for county, color in custom_colors.items():
        subset = geodata[geodata['data'] == county]
        subset.plot(color=color, ax=ax, label=county)

    legend_elements = [Line2D([0], [0], marker='o', color='w', label=county, 
                              markerfacecolor=color, markersize=10) for county, color in custom_colors.items()]
    ax.legend(handles=legend_elements, loc='lower left', fontsize=8, title='Part. Rate', title_fontsize=10)
    ax.set_title(f'Participation Rate by County in {year}',fontweight='bold',loc='left')
    plt.tick_params(left = False, right = False , labelleft = False , 
                labelbottom = False, bottom = False) 
    plt.tight_layout()
    
    # 2006
    year = 2006
    geodata = plotCountyMapData(part_by_county,year,polyg)
    geodata['data'] = np.where(geodata['data']<0.4,'< 40%',
                               np.where(geodata['data']<0.6,'40% < Part. Rate < 60%',
                                        np.where(geodata['data']<0.7,'60% < Part. Rate < 70%',
                                                 np.where(geodata['data']<0.8,'70% < Part. Rate < 80%',
                                                          np.where(geodata['data']<0.9,'80% < Part. Rate < 90%',
                                                                   np.where(geodata['data'].isnull(),'Not Available','> 90%'))))))

    custom_colors = {'< 40%': '#db1616',
    '40% < Part. Rate < 60%': '#de5b5b',
    '60% < Part. Rate < 70%': '#e08b8b',
    '70% < Part. Rate < 80%': '#8fe08b',
    '80% < Part. Rate < 90%': '#69db63',
    '> 90%': '#1fd916',
    'Not Available': '#a7aba8'}
    
    geodata['color'] = geodata['data'].map(custom_colors)

    fig,ax=plt.subplots(figsize=(10,5))
    for county, color in custom_colors.items():
        subset = geodata[geodata['data'] == county]
        subset.plot(color=color, ax=ax, label=county)

    legend_elements = [Line2D([0], [0], marker='o', color='w', label=county, 
                              markerfacecolor=color, markersize=10) for county, color in custom_colors.items()]
    ax.legend(handles=legend_elements, loc='lower left', fontsize=8, title='Part. Rate', title_fontsize=10)
    ax.set_title(f'Participation Rate by County in {year}',fontweight='bold',loc='left')
    plt.tick_params(left = False, right = False , labelleft = False , 
                labelbottom = False, bottom = False) 
    plt.tight_layout()
    
    #################################
    ## Plot Drugoverdose by County ##
    #################################
    data = []
    for y in data_dict:
        try:
            data.append(data_dict[y].groupby('fipscode').mean()['v138_rawvalue'].rename(y))
        except:
            pass
    data = pd.concat(data,axis=1)
    
    # 2023
    year = 2023
    geodata = plotCountyMapData(data,year,polyg)
    geodata['data'] = np.where(geodata['data'] < 15, '< 15',
                               np.where(geodata['data'] < 30, r'15 < $x$ < 30',
                                        np.where(geodata['data'] < 45, r'30 < $x$ < 45',
                                                 np.where(geodata['data'].isnull(),'Not Available','> 45'))))

    custom_colors = {'> 45': '#db1616',
    r'30 < $x$ < 45': '#de5b5b',
    r'15 < $x$ < 30': '#69db63',
    '< 15': '#1fd916',
    'Not Available': '#a7aba8'}
    
    geodata['color'] = geodata['data'].map(custom_colors)

    fig,ax=plt.subplots(figsize=(10,5))
    for county, color in custom_colors.items():
        subset = geodata[geodata['data'] == county]
        subset.plot(color=color, ax=ax, label=county)

    legend_elements = [Line2D([0], [0], marker='o', color='w', label=county, 
                              markerfacecolor=color, markersize=10) for county, color in custom_colors.items()]
    ax.legend(handles=legend_elements, loc='lower left', fontsize=8, title='Part. Rate', title_fontsize=10)
    ax.set_title(f'Drug Overdose Deaths by 100,000 inhabitants by County in {year}',fontweight='bold',loc='left')
    plt.tick_params(left = False, right = False , labelleft = False , 
                labelbottom = False, bottom = False) 
    plt.tight_layout()
    
    # 2019 
    year = 2019
    geodata = plotCountyMapData(data,year,polyg)
    geodata['data'] = np.where(geodata['data'] < 15, '< 15',
                               np.where(geodata['data'] < 30, r'15 < $x$ < 30',
                                        np.where(geodata['data'] < 45, r'30 < $x$ < 45',
                                                 np.where(geodata['data'].isnull(),'Not Available','> 45'))))

    custom_colors = {'> 45': '#db1616',
    r'30 < $x$ < 45': '#de5b5b',
    r'15 < $x$ < 30': '#69db63',
    '< 15': '#1fd916',
    'Not Available': '#a7aba8'}
    
    geodata['color'] = geodata['data'].map(custom_colors)

    fig,ax=plt.subplots(figsize=(10,5))
    for county, color in custom_colors.items():
        subset = geodata[geodata['data'] == county]
        subset.plot(color=color, ax=ax, label=county)

    legend_elements = [Line2D([0], [0], marker='o', color='w', label=county, 
                              markerfacecolor=color, markersize=10) for county, color in custom_colors.items()]
    ax.legend(handles=legend_elements, loc='lower left', fontsize=8, title='Part. Rate', title_fontsize=10)
    ax.set_title(f'Drug Overdose Deaths by 100,000 inhabitants by County in {year}',fontweight='bold',loc='left')
    plt.tick_params(left = False, right = False , labelleft = False , 
                labelbottom = False, bottom = False) 
    plt.tight_layout()
    
    ################################################
    ## Plot Median County Participation and Range ##
    ################################################
    part_county = part_by_county.T
    part_county.index = pd.to_datetime(part_county.index,format='%Y')+pd.offsets.YearEnd()
    part_county = part_county.loc['2006':]
    
    median = part_county.median(axis=1)
    per95  = part_county.quantile(0.95,axis=1)
    per75  = part_county.quantile(0.75,axis=1)
    per65  = part_county.quantile(0.65,axis=1)
    per35  = part_county.quantile(0.35,axis=1)
    per25  = part_county.quantile(0.25,axis=1)
    per5   = part_county.quantile(0.05,axis=1)
    
    fig,ax=plt.subplots(figsize=(10,5))
    ax.fill_between(median.index,per5 ,per95,color='lightblue',zorder=3,alpha=0.4,label='5-95% Range')
    ax.fill_between(median.index,per25,per75,color='lightblue',zorder=3,alpha=0.6,label='25-75% Range')
    ax.fill_between(median.index,per35,per65,color='lightblue',zorder=3,alpha=0.8,label='35-65% Range')
    ax.plot(median,marker='D',zorder=10,color='darkblue',label='Median')
    ax.grid(linestyle=':')
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1))
    ax.legend(loc=0)
    ax.set_title('County Participation Rate Distribution',fontweight='bold',loc='left')
    
    #####################################
    ## Participation Rate by Age Group ##
    #####################################
    data = []
    for df in data_dict:
            year = data_dict[df].copy()
            year['age'] = np.where(year['age'] < 23, 'Below 23 years old',
                                   np.where(year['age'] < 36, '23-35 years old',
                                            np.where(year['age']<55, '36-54 years old', '55+ years old')))
            year_df = {}
            for group in ['Below 23 years old','23-35 years old','36-54 years old','55+ years old']:
                for sex in [1,2]:
                    temp = year.loc[(year['age'] == group) & (year['sex'] == sex)]
                    temp['sumweight'] = temp['ind_weight'].sum()
                    temp['part_cont'] = (temp['lf_status']*temp['ind_weight']).div(temp['sumweight'])
                    part = temp['part_cont'].sum()
                    name = group + [', Men' if sex == 1 else ', Women'][0]
                    year_df.update({name:part})
            data.append(pd.Series(year_df).rename(df))
            
    data = pd.concat(data,axis=1).T.loc['2006':]
    data.index = pd.to_datetime(data.index,format='%Y')+pd.offsets.YearEnd()
    
    fig,ax=plt.subplots(ncols=2,nrows=2,figsize=(10,5))
    ax[0,0].plot(data['Below 23 years old, Men'], color = 'black', marker = '.')
    ax[0,0].plot(data['Below 23 years old, Women'], color = 'red', marker = '.')
    ax[0,0].yaxis.set_major_formatter(mtick.PercentFormatter(1))
    ax[0,0].legend(['Men','Women'],loc=0)
    ax[0,0].set_title('Below 23 years old',fontweight='bold',loc='left')
    
    ax[0,1].plot(data['23-35 years old, Men'], color = 'black', marker = '.')
    ax[0,1].plot(data['23-35 years old, Women'], color = 'red', marker = '.')
    ax[0,1].yaxis.set_major_formatter(mtick.PercentFormatter(1))
    ax[0,1].set_title('23-35 years old',fontweight='bold',loc='left')
    
    ax[1,0].plot(data['36-54 years old, Men'], color = 'black', marker = '.')
    ax[1,0].plot(data['36-54 years old, Women'], color = 'red', marker = '.')
    ax[1,0].yaxis.set_major_formatter(mtick.PercentFormatter(1))
    ax[1,0].set_title('36-54 years old',fontweight='bold',loc='left')
    
    ax[1,1].plot(data['55+ years old, Men'], color = 'black', marker = '.')
    ax[1,1].plot(data['55+ years old, Women'], color = 'red', marker = '.')
    ax[1,1].yaxis.set_major_formatter(mtick.PercentFormatter(1))
    ax[1,1].set_title('55+ years old',fontweight='bold',loc='left')
    
    plt.tight_layout()

    #########################
    ## Confounding Example ##
    #########################
    y23 = data_dict[2023]    
    
    pension   = y23['amount_received_pension']
    lf_status = y23['lf_status']
    fig,ax=plt.subplots(figsize=(10,5))
    sns.scatterplot(x=pension,y=lf_status,ax=ax,color='grey')
    mod = OLS(lf_status,pension).fit()
    yvals = ax.get_xticks() * mod.params[0]
    ax.plot(ax.get_xticks(),yvals,color='darkblue')
    ax.set_ylim([-0.2,1.2])
    ax.set_xlim([-1000,40000])
    ax.set_ylabel('Participation in the Labour Market Dummy')    
    ax.set_xlabel('Amount Received in Pension')
    ax.grid(linestyle='--')
    ax.set_title('Univariate Regression of Labour Force Participation on Pension Income',fontweight='bold',loc='left')
    
    mod.summary()
    y23.loc[y23['female'] == 1]['amount_received_pension'].mean()
    y23.loc[y23['female'] == 0]['amount_received_pension'].mean()
    
    
