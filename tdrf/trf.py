print('TRF - Loaded')

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
from sklearn.preprocessing import MinMaxScaler
# 
import patsy as pat
import statsmodels.api as sm
# 
from itertools import chain

# 

class trf :
    def __init__(self, df, model_setting, **kwargs) :
        # 
        # df    : df contaiing y, and x. temp, income, etc, need to be normailzied
        self.df = df
        # initialize df with scaled data
        self.df_reg = pd.DataFrame()
        self.scalers = {}
        # specification reading
        self.model_setting = model_setting
        # np.power, np.cos, np.sin sequence
        self.model_transform_list = []
        # regression result
        self.reg_result = {}
        self.df_pred = pd.DataFrame()
        pass


    # This is the main function for trf.
    def estimates(self):
        # normalize temp
        # self._normalize(self)
        pass


    # 
    # normaize raw data to scaler - minmax
    def _normalize(self,**kwargs):
    # def _normalize(self, var_and_minmax = dict, **kwargs):
        scaler  = self.scalers
        df      = self.df
        for i_setting in self.model_setting :
            var_org = i_setting['name']
            var_scaled = i_setting['name_scaled']
            scale_minmax = (i_setting['scale_min'], i_setting['scale_max'])
            #
            # if scaler assigned
            if scale_minmax == (None, None) :
                df.loc[:,var_scaled] = df[var_org].copy()
            else :
                # initialize scaler
                scaler[var_scaled] = MinMaxScaler()
                # generate df for fitting scale
                df_fit = pd.DataFrame({var_org:[scale_minmax[0], scale_minmax[1]]})
                # fit scaler with given variable given, min, and max.
                scaler[var_scaled] = scaler[var_scaled].fit(df_fit)
                print(var_scaled, scaler[var_scaled].data_min_, scaler[var_scaled].data_max_, scaler[var_scaled].data_range_)
                # tranforming data with scaler
                # print(df_fit)
                # print(df[[var_org]])
                arrry_scaled = scaler[var_scaled].transform(df[[var_org]])
                # save it to the original df
                df.loc[:,var_scaled] = list(chain(*arrry_scaled))
                return df[[var_scaled]]


    
    
    # 
    # iter
    def _iter_transform(i_var_model_setting):#, lst_order):
    # def _iter_transform(self, i_var_model_setting):#, lst_order):
        # print(i_var_model_setting)
        lst_order = []
        order_p = i_var_model_setting['order_pq'][0]
        order_q = i_var_model_setting['order_pq'][1]
        if (order_p == None) and (order_q == None) :
            lst_order = [None, None]
        else :
            for p in np.arange(1,order_p+1, 1):
                lst_order += [{'order_pq':[p,0]}]
            for q in np.arange(1,order_q+1,1):
                lst_order += [{'order_pq':[0,q]}]
        # 
        # initialize
        res_lst = []
        # print(res_lst, lst_order)
        i_dict = {}
        if lst_order == [None, None] :
            i_dict = i_var_model_setting.copy()
            res_lst += [i_dict]
        else :
            for i_order in lst_order : 
                i_dict = i_var_model_setting.copy()
                i_dict.update(i_order)
                res_lst += [i_dict]
        return res_lst
    # 
    # _iter_transform(model_setting[1])
    #  [{'var_type': 'tempvar','name': 'tmean_c_m_c','name_short': 'T','scale_min': 10,'scale_max': 35,'order_pq': [4, 0]},
    #   {'var_type': 'tempvar','name': 'tmean_c_m_c','name_short': 'T','scale_min': 10,'scale_max': 35,'order_pq': [0, 1]}]
    
    def gen_model_transform_list(self) :
        model_transform_list = []
        for i_setting in self.model_setting :
            # print(i_setting)
            # trf._iter_transform(i_setting)
            model_transform_list += trf._iter_transform(i_setting)
        # 
        self.model_transform_list = model_transform_list
        # print('tt')
        # return model_transform_list
    
    
    # 
    # polynomial conversion - p order : x_p1
    def _poly_transform(df, df_out, dict_transform, covar = None):
        (order_p, order_q) = dict_transform['order_pq']
        # var type
        if dict_transform['var_type'] == 'tempvar' :
            varname = (dict_transform['name_scaled'], covar)
        elif dict_transform['var_type'] == 'covar' :
            varname = (dict_transform['tempvar'], dict_transform['name_scaled'])
        else : 
            pass
        # poly transform
        if (order_p == None) or (order_q == None) :
            df_out[dict_transform['name_scaled']] = df[dict_transform['name_scaled']]
            pass
            # print(order_p)
        else :
            if (order_p > 0) and (order_q == 0) :
                df_out[f'{varname[0]}_p{order_p}'] = np.power(df[dict_transform['name_scaled']], order_p)
                # pass
            elif (order_p == 0) and (order_q > 0) : 
                df_out[f'{varname[0]}_cos_q{order_q}'] = np.cos(df[dict_transform['name_scaled']] * np.pi *2 * order_q )
                df_out[f'{varname[0]}_sin_q{order_q}'] = np.sin(df[dict_transform['name_scaled']] * np.pi *2 * order_q )
            else :
                pass
     
    def gen_df_reg(self):
        for iter_var in self.model_transform_list :
            # print(iter_var)
            trf._poly_transform(self.df, self.df_reg, iter_var)
        pass

    def _gen_spec(self):
        # print(self.df_reg.columns.values)
        # df_reg_columns = list(self.df_reg.columns.values)
        # print(df_reg_columns)
        spec = ''
        for idx, i_var in enumerate(self.df_reg.columns) :
            if idx == 0 :
                spec += f'{i_var} ~ '
            elif idx == 1 :
                spec += f'{i_var}'
            else: 
                spec += f' + {i_var}'
        return spec


    def reg_ols(self):
        spec = trf._gen_spec(self) 
        y, x = pat.dmatrices(spec, self.df_reg, return_type='dataframe')
        result = sm.OLS(y,x).fit()
        self.reg_result = result


    def _gen_df_pred(self)  :        
        df_pred = pd.DataFrame(columns=list(self.reg_result.params.index))
        
        for idx, i_var in enumerate(self.df_reg.columns) :
            # print(i_var)
            if idx == 0 :
                continue
            elif re.search('_p',i_var) :
                if re.search('_p1', i_var):
                    var_1_power = i_var
                    order_p = 1
                    df_pred[i_var] = np.arange(0, 1, 0.001) + 0.0005
                    df_pred['Intercept'] = 1
                else :
                    order_p += 1
                    df_pred[i_var] = np.power(df_pred[var_1_power], order_p )
                # print('here')
            # elif re.search('_q', i_var) :
            else :
                if re.search('_q1', i_var) :
                    order_q = 1
                    if re.search('cos', i_var) :
                        df_pred[i_var] = np.cos(df_pred[var_1_power] * 2 * order_q * np.pi)
                    else : 
                        df_pred[i_var] = np.sin(df_pred[var_1_power] * 2 * order_q * np.pi)
                else :
                    order_q += 1
                    if re.search('cos', i_var) :
                        df_pred[i_var] = np.cos(df_pred[var_1_power] * 2 * order_q * np.pi)
                    else : 
                        df_pred[i_var] = np.sin(df_pred[var_1_power] * 2 * order_q * np.pi)
                    
                continue
            # else :
            #     print('there')
        # generate temperature response with predict
        df_pred['fit_sr_death_avg_d'] = self.reg_result.predict(df_pred)
        # re-construct temp
        df_pred['temp_c'] = self.scalers['T'].inverse_transform(pd.DataFrame({'T':(df_pred.index)/1000}))

        self.df_pred = df_pred
        # return df_pred
        pass

