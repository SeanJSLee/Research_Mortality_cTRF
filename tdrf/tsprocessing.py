print('TS processing - Loaded')

import pandas as pd
import numpy as np

import calendar
import datetime
from statsmodels.tsa.arima.model import ARIMA


import warnings
from statsmodels.tools.sm_exceptions import ValueWarning
warnings.simplefilter('ignore', ValueWarning)
# import statsmodels.stats.valuewarning as valuewarning
# valuewarning.filterwarnings('ignore')


# normalize monthly death to dailiy average.
def date_to_month_days(date):
    if type(date) == str:
        date=datetime.date.fromisoformat(date)
    else :
        date
    days_of_m = calendar.monthrange(date.year, date.month)[1]
    return days_of_m

# use MA, extract SR component.
class sr_comp :
    def __init__(self, df, y_var, date_var, panel_var) :
        # data frame
        self.df = df
        # MA y variable
        self.y_var = y_var
        self.date_var = date_var
        # cross section id for iteration.
        self.panel_var = panel_var
        self.panel_var_list = df[panel_var].drop_duplicates().to_list()
        # result saving dict
        self.results = {}
        # result saving df
        self.df_results = pd.DataFrame()
        # temporary df
        self.df_i = pd.DataFrame()
        pass
    # 
    # subsetting data
    def _df_subsetting(self, i_county):
        # subsetting df for a county
        self.df_i = pd.DataFrame()
        df = self.df.copy()
        df = df[df[self.panel_var] == i_county].set_index(self.date_var)[[self.y_var]]
        self.df_i = df
        pass
    # 
    # run MA and generate fitted res (SR)
    def _run_MA(self, order = (0,0,12), i_county=None):
        # initialize
        df = self.df_i
        y_var = self.y_var
        # set model
        model = ARIMA(df, order = order)
        result = model.fit()
        # generate fitted y
        df[f'{y_var}_fitted'] = result.fittedvalues
        df[f'{y_var}_sr']  = df[y_var] - df[f'{y_var}_fitted']
        if i_county : df[self.panel_var] = i_county
        return df, result
    # 
    # run for each counties
    def sr_run(self, res_suffix = ''):
        counties = self.panel_var_list
        for i_county in counties :
            df = sr_comp._df_subsetting(self, i_county=i_county)
            # save results
            df_res, self.results[f'{i_county}{res_suffix}']  =sr_comp._run_MA(self, i_county=i_county)
            df_res = df_res.reset_index()
            # concating - drop y_var
            self.df_results = pd.concat([self.df_results, df_res], axis=0, ignore_index=True).drop(columns=self.y_var)
        # 
        # merging it to the original data
        df_merged = self.df.merge(self.df_results, on=[self.date_var, self.panel_var])
        return df_merged, self.results


        



    # for i_panel in self.panel_var_list :



