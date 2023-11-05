print('cTRF - Loaded')

import pandas as pd
import numpy as np

import re

import calendar
import datetime
from statsmodels.tsa.arima.model import ARIMA


import warnings
from statsmodels.tools.sm_exceptions import ValueWarning
warnings.simplefilter('ignore', ValueWarning)

from pandas.errors import SettingWithCopyWarning
warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)

# scickit
from sklearn.preprocessing import MinMaxScaler, StandardScaler
# 
import patsy as pat
import statsmodels.api as sm
# 
from itertools import chain






class ctrf :
    def __init__(self, df=pd.DataFrame, y=str, temp=str, x=list, date=str, order=dict
                 , n=1000, **kwargs )  :
        # df is dataframe containing dependant, temperature, covariates, and date variables.
        # cross-sectional id is not required in case of pulling multiple counties. 
        # 
        # Initialize variable
        self.df     = df.copy()
        self.df     = self.df.reset_index(drop=True) 
        self.y      = y       #
        self.temp   = temp    # 

        self.x      = x       # Think about x is empty => TRF. Need a code.
        self.date   = date
        self.order  = order             # order tuple, ie, {'tempC':{'p':4, 'q':2}}
                                        # can call self.order[self.temp]['p'] ==> 4
        self.n      = n     # prediction points
        # if len(x) == 0 :
        #     # no covariates => column name 'base' and fill 1 <= ctrf base definition
        #     self.df['base'] = 1
        # else :
        #     self.x = ['base'] + x
        # Initialize results
        self.scaler = {}
        self.reg_res = {}
        # Initialize DataFrame for polytransform and regression.
        self.df_reg = self.df[[self.date, self.y]].copy()
        self.df_temp_arry = pd.DataFrame({'place_holder':[1.0] * self.n})
        # self.df_estimates = pd.DataFrame({'Intercept':[1.0] * self.n})
        # Initialize df for ctrfs
        self.df_ctrf = pd.DataFrame()



    # Prep 1. Min-Max normalization for all variable.
    #           Scale mapping need to preverve. "MinMaxScaler()"
    def _normalize(self, show_df=False) :
        # temp variable => minmax transform
        # covariates => gussian transform
        vars = [self.temp] + self.x
        # show stats
        show_stats = ''
        # 
        for var in vars :
            # print(var)
            if var == self.temp :
                # minmax transform - initialize scaler
                self.scaler[var] = MinMaxScaler()
                # fit scale with temporary DataFrame: df_fit
                df_fit = pd.DataFrame({var:[self.df[var].min(), self.df[var].max()]})
                self.scaler[var] = self.scaler[var].fit(df_fit)
                show_stats += f'{var} - Min: {self.scaler[var].data_min_}, Max: {self.scaler[var].data_max_}, Range: {self.scaler[var].data_range_}'

            else :
                # standard transform - initialize scalewr
                self.scaler[var] = StandardScaler()
                self.scaler[var].fit(self.df[[var]])
                show_stats += f'\n{var} - Mean: {self.scaler[var].mean_}, Std: {self.scaler[var].scale_}, N: {self.scaler[var].n_samples_seen_}'

            # 
            # transform with fitted scaler.
            scaled_arry = self.scaler[var].transform(self.df[[var]])
            self.df.loc[:,var] = list(chain(*scaled_arry))
            scaled_arry = []
            # 
        if show_df :
            print(show_stats)
            return self.df
                

    # Prep 2. Construct DataFrame for regression while transforming data.
    def _transform(self, show_df=False) :
        # Initialize 'base' at original data which is a column of 1.
        # make easy to transform - 'base' is not in the data but in the pq order.
        # 'base' pretent as a dummy for temp itself in interation term.
        # Covariates are interected with transformed 'temp'
        # 'base_00' => Intercept.
        # instead of 'base' use order key
        name_base_from_dict_key = list(self.order.keys())[0]
        self.df.loc[:,name_base_from_dict_key] = 1
        # 
        for cov in self.order.keys() :
            # print(cov)
            # poly power
            for p in range(0, self.order[cov]['p']+1 ) :
                self.df_reg.loc[:,f'{cov}_{p}0'] = self.df[cov] * np.power(self.df[self.temp], p)
            # trignometric functions.
            for q in range(1, self.order[cov]['q']+1 ) :
                self.df_reg.loc[:,f'{cov}_0{q}c'] = self.df[cov] * np.cos(self.df[self.temp] * 2 * q *np.pi)
                self.df_reg.loc[:,f'{cov}_0{q}s'] = self.df[cov] * np.sin(self.df[self.temp] * 2 * q *np.pi)
                # print(q)
        
        # 
        if show_df :
            return self.df_reg

    # Running OLS
    def _run_ols(self, show_df=False, show_res = False) :
        # set index as date
        self.df_reg = self.df_reg.set_index([self.date])
        # 
        cols = self.df_reg.columns.values
        spec = f'{cols[0]}~{cols[2]}'
        for elm in cols[3:] :
            spec += f'+{elm}'
        # 
        # return self.df_reg, spec
        y, x = pat.dmatrices(spec, self.df_reg, return_type='dataframe')
        self.reg_res[0] = sm.OLS(y,x).fit()
        # 
        # fitted y
        self.df_reg[f'{self.y}_fit'] = self.reg_res[0].predict()
        


        if show_df or show_res :
            print(spec)
            # print(self.df_reg.columns.values)
            # print(self.reg_res[0].summary())
            # print(x)
            # self.reg_res[0].predict(x)
    

    # Generate 'Temp' dist [0,1] by 'base' order pq.
    def _gen_temp_array(self, show_df = False):
        order_var = list(self.order.keys())[0]
        # 
        for p in range(0, self.order[order_var]['p']+1 ) :
            self.df_temp_arry.loc[:,f'_{p}0'] = np.power((self.df_temp_arry.index / (self.df_temp_arry.index.max()+1)), p)
            # trignometric functions.
        for q in range(1, self.order[order_var]['q']+1 ) :
            self.df_temp_arry.loc[:,f'_0{q}c'] = np.cos((self.df_temp_arry.index / (self.df_temp_arry.index.max()+1)) * 2 * q *np.pi)
            self.df_temp_arry.loc[:,f'_0{q}s'] = np.sin((self.df_temp_arry.index / (self.df_temp_arry.index.max()+1)) * 2 * q *np.pi)
        #
        if show_df : return self.df_temp_arry 
        

    # var name in recovering ctrf helper
    def _gen_varname(self, var_ctrf):
        lst_grap_pram = []
        dict_chg_name = {}
        # 
        for p in range(0, self.order[var_ctrf]['p']+1):
            lst_elm = f'{var_ctrf}_{p}0'
            if lst_elm == 'base_00' :
                lst_elm = 'Intercept'
            lst_grap_pram += [lst_elm]
            dict_chg_name.update({f'_{p}0':lst_elm})
        for q in range(1, self.order[var_ctrf]['q']+1 ) :
            lst_grap_pram += [f'{var_ctrf}_0{q}c', f'{var_ctrf}_0{q}s']
            dict_chg_name.update({f'_0{q}c':f'{var_ctrf}_0{q}c', f'_0{q}s':f'{var_ctrf}_0{q}s'})
        # 
        return lst_grap_pram, dict_chg_name


    def _recover_ctrf_base(self, var_ctrf = 'base') :
        # use "esimated coefficients" and "temp array" => dot product => recover ctrf.
        # "baseCTRF" has weight 1, other cTRF can vary by seeting. Take it as a parameter.
        # "coefficients from res.param()"
        # 
        lst_grap_pram, dict_chg_name = ctrf._gen_varname(self, var_ctrf)
        #
        # Get params
        df_reg_pram = pd.DataFrame([self.reg_res[0].params])
        # 
        # Bring temp, change name and dot product
        self.df_ctrf[f'{var_ctrf}'] = np.dot(df_reg_pram[lst_grap_pram], 
                                             self.df_temp_arry.rename(columns=dict_chg_name)[lst_grap_pram].T )[0]
        # 
        return self.df_ctrf


    def _recover_ctrf_cov(self, var_ctrf = str, wgt_ctrf = float) :
        # similar as above
        lst_grap_pram, dict_chg_name = ctrf._gen_varname(self, var_ctrf)
        # 
        # Get prams of covariates
        # Note: covariate ctrf is interated with the "base". 
        df_reg_pram = pd.DataFrame([self.reg_res[0].params])
        # 
        self.df_ctrf['{var_ctrf}'.format(
                    var_ctrf=var_ctrf)] = ( np.dot(df_reg_pram[lst_grap_pram], 
                            self.df_temp_arry.rename(columns=dict_chg_name)[lst_grap_pram].T )[0])

        # self.df_ctrf['{var_ctrf}_{wgt_ctrf}'.format(
        #     var_ctrf=var_ctrf, wgt_ctrf=format(wgt_ctrf, '.2f')
        # )] = self.df_ctrf['base'] + wgt_ctrf * ( np.dot(df_reg_pram[lst_grap_pram], 
        #             self.df_temp_arry.rename(columns=dict_chg_name)[lst_grap_pram].T )[0])
        self.df_ctrf['{var_ctrf}_{wgt_ctrf}'.format(
            var_ctrf=var_ctrf, wgt_ctrf=format(wgt_ctrf, '.2f')
        )] = self.df_ctrf['base'] +  wgt_ctrf * self.df_ctrf['{var_ctrf}'.format(
                    var_ctrf=var_ctrf)]
        # 
        return self.df_ctrf
        # pass
        













