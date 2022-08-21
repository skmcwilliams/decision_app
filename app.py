#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 20 12:51:40 2022

@author: skm
"""

import pandas as pd
import numpy as np
import plotly.io as pio
import plotly.express as px
import dash
import dash_core_components as dcc
import dash_html_components as html
import statsmodels.formula.api as smf

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


def generate_table(dataframe, max_rows=26):
    return html.Table(
        # Header
        [html.Tr([html.Th(col) for col in dataframe.columns]) ] +
        # Body
        [html.Tr([
            html.Td(dataframe.iloc[i][col]) for col in dataframe.columns
        ]) for i in range(min(len(dataframe), max_rows))]
    )
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
year_bar = px.bar(wfh.groupby('mod_year').size()).update_xaxes(type='category')
state_bar = px.bar(wfh.groupby('state').size())
sales_by_state = px.histogram(wfh,x='sales',color='state',barmode='stack') # show distribution of sales by state
sales_by_year = px.histogram(wfh,x='sales',color='mod_year',barmode='stack') # distribution of sales by year
sales_over_time = px.line(wfh.groupby(['state','mod_year'])['sales'].sum().reset_index(),
                       x='mod_year',y='sales',color='state').update_xaxes(type='category') # chart of state and sales
#%%nonparametric analysis
sales_2016 = sum(wfh['sales'][wfh['post']==0])
firstdiff = sum(wfh['sales'][wfh['post_treat'] == 1])\
    - sum(wfh['sales'][(wfh['treat']== 1) * (wfh['post'] == 0)])
seconddiff = sum(wfh['sales'][(wfh['treat'] == 0) * (wfh['post'] == 1)])\
    - sum(wfh['sales'][(wfh['treat'] == 0) * (wfh['post'] == 0)])
did = firstdiff - seconddiff
did_conclusion = f"DID of {did} equates to {round((did/sales_2016)*100,2)}% sales increase over 2016 total sales"
diffdf = pd.DataFrame([firstdiff,seconddiff,did]).T.rename(columns={0:"First Difference",1:"Second Difference",2:"DID"}).reset_index()

#%%wfh parametric
results = smf.ols('sales ~ post + treat + post_treat',wfh).fit()
results_wfh = pd.read_html(results.summary().tables[1].as_html(),header=0,index_col=0)[0]
para_t_stat = results_wfh.at['post_treat','t']
para_conclusion = t_test(para_t_stat)
#%% 
rdd_df = wfh.assign(threshold=(wfh["post_years"] > 0).astype(int)) #years 2020 and 2021
rdd = smf.ols('sales ~ post_years * threshold',rdd_df).fit()
rdd_results = pd.read_html(rdd.summary().tables[1].as_html(),header=0,index_col=0)[0]
rdd_results = rdd_results.reset_index()
rdd_t_stat = rdd_results.at['threshold','t']
t_test(rdd_t_stat)
#%% # STANDARD DASH APP LANGUAGE
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
server = app.server
#%%
# CUSTOM DASH APP LANGUAGE
app.layout = html.Div(children=[
    html.Div([
        html.H4(children='Difference In Difference and Regression Discontinuity Design Models'),
        html.H5(children='DID and RDD models based on project conducted in Quantitative Analytics class as part of MS Data Analytics degree at University of Wisconsin-Whitewater'),
        html.H5(children='Data provided as part of coursework and is to assess performance after a chance. Change being workig from home, performance being measured as Sales'),
        html.H5(children='Null Hypothesis: Working From Home results in lower productivity.'),
        html.H6(children= 'Please see github for code detail: https://github.com/skmcwilliams/decision_app'),
        ]),
    
    
    html.Div([
        dcc.Markdown(children='Data provided was only for years 2016, synthetic years were created from 2018-2021, with 2020 onward being the treatment period of Working From Home.\
                     Resulting DataFrame is below'),
        dcc.Markdown(children='Databelow after modifying years and adding dummies for post, and post treatment'),
        generate_table(wfh),
        generate_table(yr_wfh),
        ]),
                     
    html.Div([
        dcc.Markdown(children='Data Check to make sure years are even and view distribution'),
        dcc.Graph(figure = year_bar),
        dcc.Graph(figure = state_bar),
        dcc.Graph(figure = sales_by_state),
        dcc.Graph(figure = sales_by_year),
        dcc.Graph(figure = sales_over_time),
        ]),
                     
    html.Div([
        dcc.Markdown(children = 'First, conduct nonparametric DID calclations on the post-treatment group (Wisconsinites) by running statsmodels ols with sales as RHS variable and post, treatment, post-treatment as LHS, Results below.'),
        generate_table(results_wfh),
        dcc.Markdown(children='Post-treatment t-statistic greater than 2.575, making the post-treatment relevant at 99% Confidence Level. Reject null hypothesis that working from home will hinder performance.'),
        dcc.Markdown(children = f'{did_conclusion}'),
        generate_table(diffdf)
        
        ]),
    
    html.Div([
        dcc.Markdown(children='Next, run parametric RDD calculations via regression model (statsmodels OLS)\
                     on post_years (# of years beyond 2019) and the threshold. Threshold = 1 where post_years > 0 else 0'),
        generate_table(rdd_df),
        generate_table((rdd_results)),
        dcc.Markdown(children='High t-statistic for Threshold variable indicates that years beyond 2019, the WFH years, produce higher sales with a 99% Confidene Level')
        ])
])

#%%RUN
if __name__=='__main__':
   app.run_server(debug=True)

