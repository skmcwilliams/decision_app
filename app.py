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
    t_test = zip(confidence_levels,c_vals)
    
    for k,v in t_test:
        if abs(t_statistic) > v:
            return f"T-statistic passes {k}% confidence level with t-statistic of {round(t_statistic,2)}, greater than {v}, reject null hypothesis \n"
        else:
            return f"T-statistic of {round(t_statistic,2)} does not pass {k}% confidence level, accept null"

#%% READ DATA DROP WISCONSIN COLUMN
wfh = pd.read_csv('workingfromhome.csv').rename(columns={'wisconsin':"treat"}).set_index('employeeid')

"""synthetically create more years, original assignemnt split 2016 and 2017, follow same thinking"""
wfh['mod_year'] = wfh['year'].apply(lambda x: np.random.choice(list(range(2018,2020))) if x == 2016 else np.random.choice(list(range(2020,2022))))
# check year distribution
yr_wfh = pd.DataFrame(wfh['mod_year'].value_counts()).reset_index().rename(columns={'mod_year':'count','index':'mod_year'})
yr_wfh.insert(2,"pct_total", round(yr_wfh['count']/yr_wfh['count'].sum(),2)*100)

""" original ppost treatment period was 2017, set to 2020+, original control period of 2016 set to 2018 and 2019"""
wfh['post'] = np.where(wfh['mod_year'] >= 2020, 1, 0) # make 2020 as threshold of WFH, should be 
wfh['post_years'] = wfh['mod_year'] - 2019
wfh['post_treat'] = wfh['post'] * wfh['treat'] # Interaction Term  to make dummy for post_treatment group = Wisconsin and 2017#%
#%% charts
wfh.groupby('mod_year').size().plot(kind='bar') # number of each year is about even, data is fair
plt.show()

wfh.groupby('state').size().plot(kind='bar') # count of employees in each year is about even, data is fair
plt.show()

wfh.groupby('state')['sales'].plot(kind='kde',label=True) # Visualize no sales bias towards state prior to doing anything
plt.show()

wfh.groupby('mod_year')['sales'].plot(kind='kde',legend=True) 
plt.show() # Visualize no bias in sales by mod year prior to doing anything (checking to see if modified year assignments are effectively randomized)

#%%nonparametric analysis
sales_2016 = sum(wfh['sales'][wfh['post']==0])
firstdiff = sum(wfh['sales'][wfh['post_treat'] == 1])\
    - sum(wfh['sales'][(wfh['treat']== 1) * (wfh['post'] == 0)])
seconddiff = sum(wfh['sales'][(wfh['treat'] == 0) * (wfh['post'] == 1)])\
    - sum(wfh['sales'][(wfh['treat'] == 0) * (wfh['post'] == 0)])
did = firstdiff - seconddiff
did_conclusion = f"DID of {did} equates to {round((did/sales_2016)*100,2)}% sales increase over 2016 total sales"
diffdf = pd.DataFrame([firstdiff,seconddiff,did]).T.rename(columns={0:"First Difference",1:"Second Difference",2:"DID"})
print(diffdf)

#%%wfh parametric
results = smf.ols('sales ~ post + treat + post_treat',wfh).fit()
results_wfh = pd.read_html(results.summary().tables[1].as_html(),header=0,index_col=0)[0]
results_wfh = results_wfh.reset_index()
para_t_stat = results_wfh.at[3,'t']
para_conclusion = t_test(para_t_stat)
print(para_conclusion)
#%% 
rdd_df = wfh.assign(threshold=(wfh["post_years"] > 0).astype(int)) #years 2020 and 2021
rdd = smf.ols('sales ~ post_years * threshold',rdd_df).fit()
rdd_results = pd.read_html(rdd.summary().tables[1].as_html(),header=0,index_col=0)[0]
rdd_results = rdd_results.reset_index()
rdd_t_stat = rdd_results.at[2,'t']
rdd_ttest = t_test(rdd_t_stat)
print(rdd_ttest)
