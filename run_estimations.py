# -*- coding: utf-8 -*-
"""
Created on Fri May 24 22:14:40 2024

@author: lfval
"""

import shap
import pickle
import warnings
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from MLModel import MLModelsEstimation,MLModelsCV,MLModelsKFoldCV
from statsmodels.discrete.discrete_model import Logit
from sklearn.metrics import roc_auc_score,accuracy_score
from concurrent.futures import ThreadPoolExecutor, as_completed
from statsmodels.stats.outliers_influence import variance_inflation_factor

def importEstimationDataset(path):
    with open(f'{path}/Data/input/data_final.p','rb') as f:
        data_dict = pickle.load(f)
    
    # We will drop the dummies with multiple NAs
    for d in data_dict:
        data_dict[d] = data_dict[d].dropna(axis=1)
        
    # Now dropping unuseful variables such as fipscode & state
    for d in data_dict:
        to_exc_cols = ['fipscode','state','industry','ind_weight']
        data_dict[d] = data_dict[d].drop(to_exc_cols,axis=1,errors='ignore')
        data_dict[d] = data_dict[d].rename(columns = {'v001_rawvalue':'prem_deaths',
                                                      'v011_rawvalue':'adult_obseity',
                                                      'v049_rawvalue':'exc_drinking',
                                                      'v014_rawvalue':'teen_births',
                                                      'v155_rawvalue':'flu_vacc',
                                                      'v136_rawvalue':'sev_hous_prob',
                                                      'v137_rawvalue':'long_comute',
                                                      'v138_rawvalue':'drug_overdose',
                                                      'HI':'hires',
                                                      'JO':'job_openings',
                                                      'LD':'layoffs',
                                                      'QU':'quits',
                                                      'TS':'tot_sep',
                                                      'd@deficiency':'d@handicap'})
        data_dict[d] = data_dict[d][[i for i in data_dict[d] if 'industry_most_time' not in i]]
        
    return data_dict

def fitSimpleLogit(year, data):
    logit = Logit(data['lf_status'], data.drop('lf_status', axis=1)).fit(maxiter=250)
    
    return year, logit

def createLogitEstTimeSeries(data_dict,probit_res,var):
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

def retLatexLogitEst(logit_res,year):
    summ = logit_res[year].summary()
    lat_str = summ.as_latex()
    
    return lat_str

def modelOptimizationYear(data_dict,year,model_list):
    year_data = data_dict[year]
    X = year_data.drop('lf_status',axis=1)
    y = year_data['lf_status']

    mlmodelcv = MLModelsCV(X,y,0.3)
    opt_params = {}
    
    for mod in model_list:
        temp_opt_params = mlmodelcv.optimizeOptuna(method_name=mod)
        opt_params.update({mod:temp_opt_params})
    
    return opt_params

def modelHorseRace(data_dict,year,model_list,params_dict):
    year_data = data_dict[year]
    X = year_data.drop('lf_status',axis=1)
    y = year_data['lf_status']
    
    mlmodelest = MLModelsKFoldCV(X,y,5)
    models = {'True Positive':pd.DataFrame(),
              'True Negative':pd.DataFrame(),
              'False Positive':pd.DataFrame(),
              'False Negative':pd.DataFrame(),
              'AUC':pd.DataFrame(),
              'Accuracy':pd.DataFrame()}
    
    warnings.filterwarnings("ignore")
    for mod in tqdm(model_list):
        estmod = getattr(mlmodelest,mod)(params_dict[year][mod])
        
        models['True Positive'][mod]  = estmod['cfs_matrix']['test_tp']
        models['True Negative'][mod]  = estmod['cfs_matrix']['test_tn']
        models['False Positive'][mod] = estmod['cfs_matrix']['test_fp']
        models['False Negative'][mod] = estmod['cfs_matrix']['test_fn']
        models['AUC'][mod]            = estmod['roc_auc']['test_score']
        models['Accuracy'][mod]       = estmod['accuracy']['test_score']
        
    warnings.resetwarnings()
    
    return models

def horseRaceResTable(data,name):
    df   = data[name].T
    mean = df.mean(axis=1)
    std  = df.std(axis=1)
    minv = df.min(axis=1)
    maxv = df.max(axis=1)
    df['Mean'] = mean
    df['Standard Deviation'] = std
    df['Min'] = minv
    df['Max'] = maxv
    return df

def finalEstimation(data_dict,year,model,params_dict):
    year_data = data_dict[year]
    X = year_data.drop('lf_status',axis=1)
    y = year_data['lf_status']
    
    mlmodelest = MLModelsEstimation(X,y,0.00000000001)
    mod = getattr(mlmodelest,model)(params_dict[year][model])
    
    if model in ['RandomForestRegressor','XGBoost','XGBoostRF','CatBoost']:
        shap_values = shap.TreeExplainer(mod).shap_values(X)
    else:
        raise ValueError(f'No explainer available for {model} yet.')
    
    return mod,shap_values,X

def plotMarginalSHAPEffect(year_mod,var,int_var,vid,ax,add_line,color,linestyle,plot,label):
    if plot:
        shap.dependence_plot(vid[var], finests[year_mod][1], finests[year_mod][2],interaction_index=vid[int_var],ax=ax)
    else:
        pass

    x = finests[year_mod][2].iloc[:,vid[var]]
    y = pd.DataFrame(finests[year_mod][1]).iloc[:,vid[var]]

    if add_line:
        quantiles = np.arange(0, 1.0, 0.01)
        unique_values = np.round(np.quantile(x, quantiles)).astype(int)
        
        x = pd.Series([unique_values[(np.abs(unique_values - i)).argmin()] for i in x])
        
        mean_y_values = [y.loc[x.loc[x==i].index].mean() for i in unique_values]
        
        if plot:
            plt.plot(unique_values, mean_y_values, color=color, linestyle=linestyle)
        else:
            ax.plot(unique_values, mean_y_values, color=color, linestyle=linestyle,label=label)
    
    else:
        if (var == 'd@social_security') | (var == 'd@handicap'):
            y1 = y.loc[x.loc[x==1].index].mean()
            y0 = y.loc[x.loc[x==0].index].mean()
            
            ax.plot([0,1], [y0,y1], color='black', linestyle='--',marker='D')
            ax.text(0.5, -0.1, f'Mean diff: {(y1-y0).round(3)}',ha='center',bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=1'))

def plot3DCreator(xvar,xlabel,yvar,ylabel,zvar,zlabel,title,finest,shap):
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    scat = ax.scatter(finest['2023 - CatBoost'][2][xvar],finest['2023 - CatBoost'][2][yvar],finest['2023 - CatBoost'][2][zvar],c=shap[zvar],cmap='viridis')
    plt.colorbar(scat, ax=ax, label='Shapley Value',orientation='vertical',location='left',shrink=0.5)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_zlabel(zlabel)
    ax.set_title(title,fontweight='bold',loc='left')
    plt.tight_layout()
    plt.subplots_adjust(right=0.88)

if __name__ == '__main__':
    path = r'C:\Users\lfval\OneDrive\Documentos\FGV\Monografia\DeterminantsofParticipation'
    data_dict = importEstimationDataset(path)
    variables = list(data_dict[2023].columns)
    
    # Checking starting year
    y = {i:list(data_dict[i].columns) for i in data_dict}
    x = [list(data_dict[i].columns) for i in data_dict]
    
    h = []
    for i in x:
        h+=i
    h = list(set(h))
    
    start_year = {}
    for year in y:
        for v in h:
            if v in y[year]:
                if v not in start_year:
                    start_year.update({v:year})
                else:
                    pass
    pd.Series(start_year).sort_index().to_excel(r'C:\Users\lfval\OneDrive\Documentos\FGV\Monografia\DeterminantsofParticipation\Data\input\vars_dict.xlsx')
    
    
    # Simple Logit Estimations
    probit_res = {}
    with ThreadPoolExecutor() as executor:
        futures = {executor.submit(fitSimpleLogit, year, data_dict[year]): year for year in data_dict}
        for future in tqdm(as_completed(futures), total=len(futures)):
            try:
                year, logit = future.result()
                probit_res[year] = logit
                del logit
            except: pass
   
    lat_str = retLatexLogitEst(probit_res,2023)    
    
    soc_sec_param = createLogitEstTimeSeries(data_dict,probit_res,'d@social_security')
    age_param     = createLogitEstTimeSeries(data_dict,probit_res,'age')
    
    fig,ax=plt.subplots(nrows=2,figsize=(15,7))
    
    ax[0].plot(soc_sec_param['Estimate'], color='blue',label='Estimates',marker='o')
    ax[0].fill_between(x=soc_sec_param.index,y1=soc_sec_param['LB'],y2=soc_sec_param['UB'],color='lightblue',alpha=0.5,label='95% C.I.')
    ax[0].set_title('Social Security Dummy',fontweight='bold')
    ax[0].grid(linestyle=':')
    ax[0].legend(loc=0)
    
    ax[1].plot(age_param['Estimate'], color='blue',label='Estimates',marker='o')
    ax[1].fill_between(x=age_param.index,y1=age_param['LB'],y2=age_param['UB'],color='lightblue',alpha=0.5,label='95% C.I.')
    ax[1].set_title('Age',fontweight='bold')
    ax[1].grid(linestyle=':')
    
    
    # Run hyperparameters tunned versions of the models
    estmodels = False
    model_list = ['Logit','RandomForestRegressor','XGBoost','XGBoostRF','CatBoost']
    
    if estmodels:
        opt_params2023 = modelOptimizationYear(data_dict,2023,model_list)
        opt_params2020 = modelOptimizationYear(data_dict,2020,model_list)
        opt_params2019 = modelOptimizationYear(data_dict,2019,model_list)
        opt_params2016 = modelOptimizationYear(data_dict,2016,model_list)
        opt_params2010 = modelOptimizationYear(data_dict,2010,model_list)
        opt_params2007 = modelOptimizationYear(data_dict,2007,model_list)
        
        tunned_hyperparameters = {2007:opt_params2007,
                                  2010:opt_params2010,
                                  2016:opt_params2016,
                                  2019:opt_params2019,
                                  2020:opt_params2020,
                                  2023:opt_params2023}
        
        with open(f'{path}/Data/output/hyperparameters.txt','wb') as f:
            pickle.dump(tunned_hyperparameters,f)
    else:
        with open(f'{path}/Data/output/hyperparameters.txt','rb') as f:
            tunned_hyperparameters = pickle.load(f)
            
    for year in tunned_hyperparameters:
        tunned_hyperparameters[year]['CatBoost'].update({'iterations':100})
    
    if estmodels:
        hr2023 = modelHorseRace(data_dict,2023,model_list,tunned_hyperparameters)
        hr2020 = modelHorseRace(data_dict,2020,model_list,tunned_hyperparameters)
        hr2019 = modelHorseRace(data_dict,2019,model_list,tunned_hyperparameters)
        hr2016 = modelHorseRace(data_dict,2016,model_list,tunned_hyperparameters)
        hr2010 = modelHorseRace(data_dict,2010,model_list,tunned_hyperparameters)
        hr2007 = modelHorseRace(data_dict,2007,model_list,tunned_hyperparameters)
        dfs   = [hr2007.copy(),hr2010.copy(),hr2016.copy(),hr2019.copy(),hr2020.copy(),hr2023.copy()]
        
        with open(f'{path}/Data/output/hrdfs.txt','wb') as f:
            pickle.dump(dfs,f)
    else:
        with open(f'{path}/Data/output/hrdfs.txt','rb') as f:
            dfs = pickle.load(f)
    
    # For each of the years, let us estimate and create Shapley Values for our champion model
    finests = {}
    years = [  2007  ,2010,  2016,  2019,  2020,  2023]
    
    i = 0
    for df,year in zip(dfs,years):
        out_path = r'C:\Users\lfval\OneDrive\Documentos\FGV\Monografia\DeterminantsofParticipation\Data\output'
        
        tp = horseRaceResTable(df,'True Positive')
        tn = horseRaceResTable(df,'True Negative')
        fp = horseRaceResTable(df,'False Positive')
        fn = horseRaceResTable(df,'False Negative')
        au = horseRaceResTable(df,'AUC')
        ac = horseRaceResTable(df,'Accuracy')
        
        tp.to_excel(f'{out_path}/{year}_true positive_performance.xlsx')
        tn.to_excel(f'{out_path}/{year}_true negative_performance.xlsx')
        fp.to_excel(f'{out_path}/{year}_false positive_performance.xlsx')
        fn.to_excel(f'{out_path}/{year}_false negative_performance.xlsx')
        au.to_excel(f'{out_path}/{year}_auc_performance.xlsx')
        ac.to_excel(f'{out_path}/{year}_accuracy_performance.xlsx')
        
        dfs[i] = {'True Positive':tp,
                     'True Negative':tn,
                     'False Positive':fp,
                     'False Negative':fn,
                     'AUC':au,
                     'Accuracy':ac}
        i+=1
        
        mod = 'CatBoost'
        est = finalEstimation(data_dict,year,mod,tunned_hyperparameters)
        finests.update({f'{year} - {mod}':est})
    
    # All vars Shapley Values
    fig = plt.figure(figsize=(20, 10))
    ax = fig.add_subplot(2, 2, 1)
    shap.summary_plot(finests['2007 - CatBoost'][1], finests['2007 - CatBoost'][2], plot_size=None, show=False)
    ax.set_title('2007',fontweight='bold')
    ax = fig.add_subplot(2, 2, 2)
    shap.summary_plot(finests['2016 - CatBoost'][1], finests['2016 - CatBoost'][2], plot_size=None, show=False)
    ax.set_title('2016',fontweight='bold')
    ax = fig.add_subplot(2, 2, 3)
    shap.summary_plot(finests['2019 - CatBoost'][1], finests['2019 - CatBoost'][2], plot_size=None, show=False)
    ax.set_title('2019',fontweight='bold')
    ax = fig.add_subplot(2, 2, 4)
    shap.summary_plot(finests['2023 - CatBoost'][1], finests['2023 - CatBoost'][2], plot_size=None, show=False)
    ax.set_title('2023',fontweight='bold')
    fig.savefig(r'C:\Users\lfval\OneDrive\Documentos\FGV\Monografia\DeterminantsofParticipation\Data\output\shapley_values_all_years.png')




    shap2023 = pd.DataFrame(finests['2023 - CatBoost'][1],columns=finests['2023 - CatBoost'][2].columns)


    # 3D Plots
    plot3DCreator('hires','County Hires Rate','age','Age','educ_college','College Education Dummy','Shapley Values Distribution',finests,shap2023)
    plot3DCreator('adult_obseity','County Hires Rate','age','Age','educ_college','College Education Dummy','Shapley Values Distribution',finests,shap2023)
    plot3DCreator('educ_college','educ','age','age','Shapley Values Distribution',finests,shap2023)


    ## SHAP Plots ##
    
    # Age
    vid2007 = {column_name: i for i, column_name in enumerate([i for i in data_dict[2007].columns if i!='lf_status'])}
    vid2016 = {column_name: i for i, column_name in enumerate([i for i in data_dict[2016].columns if i!='lf_status'])}
    vid2019 = {column_name: i for i, column_name in enumerate([i for i in data_dict[2019].columns if i!='lf_status'])}
    vid2023 = {column_name: i for i, column_name in enumerate([i for i in data_dict[2023].columns if i!='lf_status'])}
        
    fig,ax = plt.subplots(figsize=(14,5))
    plotMarginalSHAPEffect('2007 - CatBoost','age','female',vid2007,ax=ax,add_line=True,color='darkblue',linestyle='-',plot=False,label='2007')
    plotMarginalSHAPEffect('2016 - CatBoost','age','female',vid2016,ax=ax,add_line=True,color='darkgreen',linestyle='-',plot=False,label='2016')
    plotMarginalSHAPEffect('2019 - CatBoost','age','female',vid2019,ax=ax,add_line=True,color='red',linestyle='-',plot=False,label='2019')
    plotMarginalSHAPEffect('2023 - CatBoost','age','female',vid2023,ax=ax,add_line=True,color='purple',linestyle='-',plot=False,label='2023')
    ax.legend()
    ax.grid(linestyle=':')
    ax.set_ylabel('Shapley Values')
    ax.set_xlabel('Age')
    ax.set_title('Mean Shapley Value within Age Group', fontweight='bold',loc='left')
    
    # Age/Soc Sec
    shap2023['hi_dummy'] = finests['2023 - CatBoost'][2]['d@social_security'].replace(2,1).replace(3,0)    
    shap2023['age_val'] = finests['2023 - CatBoost'][2]['age']
    
    hi1 = shap2023.loc[shap2023['hi_dummy']==1]
    hi0 = shap2023.loc[shap2023['hi_dummy']==0]

    m1 = hi1.groupby('age_val').mean()['age']  
    m0 = hi0.groupby('age_val').mean()['age']
    
    s1 = hi1.groupby('age_val').std()['age'].ffill()
    s0 = hi0.groupby('age_val').std()['age'].ffill()
    
    fig,ax = plt.subplots(figsize=(14,5))
    ax.plot(m1.index,m1,color='darkblue',label='Receive Social Security')
    ax.plot(m0.index,m0,color='red',label="Doesn't receive Social Security")
    ax.fill_between(m1.index,m1-(s1*1),m1+(s1*1),color='darkblue',alpha=0.4,label='1 S.D. Range')
    ax.fill_between(m0.index,m0-(s0*1),m0+(s0*1),color='red',alpha=0.4)
    ax.legend(loc=0)
    ax.grid(linestyle=':')
    ax.set_title('Mean Shapley Value within Age Group conditional on receiving Social Security', fontweight='bold',loc='left')
    ax.set_ylabel('Shapley Values')
    ax.set_xlabel('Age')
    
    # Age + Soc Sec
    shap2023['hi_dummy'] = finests['2023 - CatBoost'][2]['d@social_security'].replace(2,1).replace(3,0)    
    shap2023['age_val'] = finests['2023 - CatBoost'][2]['age']
    shap2023['sum'] = shap2023['age'] + shap2023['d@social_security']
    
    hi1 = shap2023.loc[shap2023['hi_dummy']==1]
    hi0 = shap2023.loc[shap2023['hi_dummy']==0]

    m1 = hi1.groupby('age_val').mean()['sum']  
    m0 = hi0.groupby('age_val').mean()['sum']
    
    s1 = hi1.groupby('age_val').std()['sum'].ffill()
    s0 = hi0.groupby('age_val').std()['sum'].ffill()
    
    fig,ax = plt.subplots(figsize=(14,5))
    ax.plot(m1.index,m1,color='darkblue',label='Receive Social Security')
    ax.plot(m0.index,m0,color='red',label="Doesn't receive Social Security")
    ax.fill_between(m1.index,m1-(s1*1),m1+(s1*1),color='darkblue',alpha=0.4,label='1 S.D. Range')
    ax.fill_between(m0.index,m0-(s0*1),m0+(s0*1),color='red',alpha=0.4)
    ax.legend(loc=0)
    ax.grid(linestyle=':')
    ax.set_title('Sum of Age and Social Security Dummy Shapley Values by Age Group', fontweight='bold',loc='left')
    ax.set_ylabel('Shapley Values')
    ax.set_xlabel('Age')
    
    # County URate
    fig = plt.figure(figsize=(20, 10))
    ax = fig.add_subplot(2, 2, 1)
    plotMarginalSHAPEffect('2007 - CatBoost','County_URate','d@social_security',vid2007,ax=ax,add_line=True,color='black',linestyle='--',plot=True,label='_')
    ax.set_title('2007',fontweight='bold')
    ax = fig.add_subplot(2, 2, 2)
    plotMarginalSHAPEffect('2016 - CatBoost','County_URate','d@social_security',vid2016,ax=ax,add_line=True,color='black',linestyle='--',plot=True,label='_')
    ax.set_title('2016',fontweight='bold')
    ax = fig.add_subplot(2, 2, 3)
    plotMarginalSHAPEffect('2019 - CatBoost','County_URate','d@social_security',vid2019,ax=ax,add_line=True,color='black',linestyle='--',plot=True,label='_')
    ax.set_title('2019',fontweight='bold')
    ax = fig.add_subplot(2, 2, 4)
    plotMarginalSHAPEffect('2023 - CatBoost','County_URate','d@social_security',vid2023,ax=ax,add_line=True,color='black',linestyle='--',plot=True,label='_')
    ax.set_title('2023',fontweight='bold')
    
    # Social Security
    fig = plt.figure(figsize=(20, 10))
    ax = fig.add_subplot(2, 2, 1)
    plotMarginalSHAPEffect('2007 - CatBoost','d@social_security','age',vid2007,ax=ax,add_line=False,color='black',linestyle='--',plot=True,label='_')
    ax.set_title('2007',fontweight='bold')
    ax = fig.add_subplot(2, 2, 2)
    plotMarginalSHAPEffect('2016 - CatBoost','d@social_security','age',vid2016,ax=ax,add_line=False,color='black',linestyle='--',plot=True,label='_')
    ax.set_title('2016',fontweight='bold')
    ax = fig.add_subplot(2, 2, 3)
    plotMarginalSHAPEffect('2019 - CatBoost','d@social_security','age',vid2019,ax=ax,add_line=False,color='black',linestyle='--',plot=True,label='_')
    ax.set_title('2019',fontweight='bold')
    ax = fig.add_subplot(2, 2, 4)
    plotMarginalSHAPEffect('2023 - CatBoost','d@social_security','age',vid2023,ax=ax,add_line=False,color='black',linestyle='--',plot=True,label='_')
    ax.set_title('2023',fontweight='bold')
    
    # Handicap
    fig = plt.figure(figsize=(20, 10))
    ax = fig.add_subplot(2, 2, 1)
    plotMarginalSHAPEffect('2007 - CatBoost','d@handicap','age',vid2007,ax=ax,add_line=False,color='black',linestyle='--',plot=True,label='_')
    ax.set_title('2007',fontweight='bold')
    ax = fig.add_subplot(2, 2, 2)
    plotMarginalSHAPEffect('2016 - CatBoost','d@handicap','age',vid2016,ax=ax,add_line=False,color='black',linestyle='--',plot=True,label='_')
    ax.set_title('2016',fontweight='bold')
    ax = fig.add_subplot(2, 2, 3)
    plotMarginalSHAPEffect('2019 - CatBoost','d@handicap','age',vid2019,ax=ax,add_line=False,color='black',linestyle='--',plot=True,label='_')
    ax.set_title('2019',fontweight='bold')
    ax = fig.add_subplot(2, 2, 4)
    plotMarginalSHAPEffect('2023 - CatBoost','d@handicap','age',vid2023,ax=ax,add_line=False,color='black',linestyle='--',plot=True,label='_')
    ax.set_title('2023',fontweight='bold')
    
    