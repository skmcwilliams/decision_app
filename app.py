#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 20 12:51:40 2022

@author: skm
"""

import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt

#%% funcs
def t_test(t_statistic):
    confidence_levels = [90,95,99]
    c_vals = [1.645,1.96,2.575]
    combo = list(zip(confidence_levels,c_vals))
    
    count = 0
    for k,v in combo:
        count += 1
        if abs(t_statistic) > v:
            print(f"T-statistic of {round(t_statistic,2)} is greater than {v}, passes {k}% confidence level, reject null hypothesis with {k}% confidence \n")
        
        elif count == 3:
            return f"T-statistic of {round(t_statistic,2)} does not pass 90% confidence level, accept null"
        
        else:
            print(f"t-statistic of {t_statistic} does not pass {k}% confidence level")       

#%% READ DATA DROP WISCONSIN COLUMN
wfh = pd.read_csv('workingfromhome.csv').rename(columns={'wisconsin':"treat"}).set_index('employeeid')

"""synthetically create more years, original assignemnt split 2016 and 2017, follow same thinking"""
wfh['mod_days'] = wfh['year'].apply(lambda x: np.random.choice([i*-1 for i in range(1,365)]) if x == 2016 else np.random.choice(list(range(1,365))))
# check year distribution
yr_wfh = pd.DataFrame(wfh['mod_days'].value_counts()).reset_index().rename(columns={'mod_days':'count','index':'mod_days'})
yr_wfh.insert(2,"pct_total", round(yr_wfh['count']/yr_wfh['count'].sum(),2)*100)

""" original ppost treatment period was 2017, set to 2020+, original control period of 2016 set to 2018 and 2019"""
wfh['post'] = np.where(wfh['mod_days'] >= 1, 1, 0) # greater than 0 indicates treatment
wfh['post_treat'] = wfh['post'] * wfh['treat'] # Interaction Term  to make dummy for post_treatment group = Wisconsin and 2017#%
#%% charts
wfh.groupby('mod_days').size().plot(kind='kde',title='Normal Distribution of mod_days is Acceptable') # normal distribution of mod_days is acceptable
plt.show()

wfh.groupby('state').size().plot(kind='bar',title='Count of Employees in Each State is Acceptable') # count of employees in each year is about even, data is fair
plt.show()

wfh.groupby('state')['sales'].plot(kind='kde',legend=True,title='Normal Distribution of Sales by State is Acceptable') # Visualize no sales bias towards state prior to doing anything
plt.show()

wfh.groupby('mod_days')['sales'].sum().plot(kind='line',title='Sales Pre-Post Treatment')
plt.show()

#%%nonparametric analysis
sales_2016 = sum(wfh['sales'][wfh['post']==0])
firstdiff = sum(wfh['sales'][wfh['post_treat'] == 1])\
    - sum(wfh['sales'][(wfh['treat']== 1) * (wfh['post'] == 0)])
seconddiff = sum(wfh['sales'][(wfh['treat'] == 0) * (wfh['post'] == 1)])\
    - sum(wfh['sales'][(wfh['treat'] == 0) * (wfh['post'] == 0)])
did = firstdiff - seconddiff
did_conclusion = f"DID of {did} shows 2017's sales are a {round((did/sales_2016)*100,2)}% sales increase over 2016 total sales"
diffdf = pd.DataFrame([firstdiff,seconddiff,did]).T.rename(columns={0:"First Difference",1:"Second Difference",2:"DID"})
print(diffdf,did_conclusion,sep="\n")
#%%wfh parametric
results = smf.ols('sales ~ post + treat + post_treat',wfh).fit()
results_wfh = pd.read_html(results.summary().tables[1].as_html(),header=0,index_col=0)[0]
results_wfh = results_wfh.reset_index()
para_t_stat = results_wfh.at[3,'t']
para_conclusion = t_test(para_t_stat)
#%% 
rdd_df = wfh.assign(threshold=(wfh["mod_days"] > 0).astype(int)) #years 2020 and 2021
rdd = smf.ols('sales ~ mod_days * threshold',rdd_df).fit()
rdd_results = pd.read_html(rdd.summary().tables[1].as_html(),header=0,index_col=0)[0]
rdd_results = rdd_results.reset_index()
rdd_t_stat = rdd_results.at[2,'t']
rdd_ttest = t_test(rdd_t_stat)
