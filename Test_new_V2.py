import streamlit as st
import multiprocessing as multiprocess
import numpy as np
import time

from concurrent.futures import ThreadPoolExecutor
from evalml.automl.engine.cf_engine import CFEngine, CFClient
import warnings
warnings.filterwarnings("ignore")
import dask.dataframe as dd
import pandas as pd
import re
from pathlib import Path
from flask import Flask, render_template
import evalml_ARO_Regression_working_04_May_2023 
import psycopg2
import Reorder_Safety_Stock
from Reorder_Safety_Stock import re_oreder
import demo4
import matplotlib.pyplot as plt
import math
from sklearn.utils import resample
@st.cache_resource
def aro(df,i,val_period):
    try:
        start = time.time()
        num = re.findall(r'\d+', i) 
        store_val = int(num[0])
        item_val = int(num[1])
        df.drop("Unnamed: 0",axis=1,inplace=True)
        cols = ['Date', 'store', 'item', 'category', 'sub_category', 'product', 'sales']
        df.columns = cols
        df = df.loc[(df['store'] == store_val) & (df['item'] == item_val)]
        
        
        df['Date'] = pd.to_datetime(df['Date'], format='%Y-%m-%d')
        
        
        
        df['month'] = [i.month for i in df['Date']]
        df['year'] = [i.year for i in df['Date']]
        df['day_of_week'] = [i.dayofweek for i in df['Date']]
        df['day_of_year'] = [i.dayofyear for i in df['Date']]
        df['store'] = df['store'].astype(np.int64)
        df['item'] = df['item'].astype(np.int64)
    
        df['Date'] = df['Date'].dt.strftime('%d-%m-%Y')
        df['store'] = ['store_' + str(i) for i in df['store']]
        df['item'] = ['item_' + str(i) for i in df['item']]
        df['time_series'] = df[['store', 'item']].apply(lambda x: '_'.join(x), axis=1)
        df.drop(['store', 'item'], axis=1, inplace=True)
        df = df[['Date', 'category', 'sub_category', 'product', 'sales',  'time_series',
               'month', 'year', 'day_of_week', 'day_of_year']]
        df_metrics = pd.DataFrame(columns = [['store_item','train_test_val','mean_absolute_error','median_absolute_error','mean_absolute_percentage_error','mean_squared_error','rmse','r2_score','Time_taken']])
        cpu_cnt = multiprocess.cpu_count()
        
        final_df = []
        end = time.time() - start
        Model_start = time.time()
        df_metrics_tmp = pd.DataFrame()
        df_forecast_tmp = pd.DataFrame()
            
    
        df_metrics = pd.DataFrame()
        df_forecast = pd.DataFrame()
        
        if __name__ == "__main__":
            cpu_cnt = multiprocess.cpu_count()
            df_metrics = pd.DataFrame()
            df_forecast = pd.DataFrame()
            final_df = []
            with multiprocess.Pool(processes=cpu_cnt) as pool:
                results = [pool.apply_async(demo4.eval_ml, args=[i, df, val_period])]
                name = "output_" + i
                name = [p.get() for p in results]
                df_tuple = (name[0])
                df_metrics_tmp = df_tuple[0]
                df_forecast_tmp = df_tuple[1]
                df_metrics = pd.concat([df_metrics, df_metrics_tmp], axis=0)
                df_forecast = pd.concat([df_forecast, df_forecast_tmp], axis=0)
                df_forecast["error"] = ((df_forecast["pred_val"]- df_forecast["sales"])/df_forecast["sales"])
                df_forecast["Accuracy"] = 1 - ((df_forecast["pred_val"]- df_forecast["sales"])/df_forecast["sales"])  
                df_forecast["Avg_Accuracy"] = df_forecast["Accuracy"].mean()
                df_reorder = re_oreder(df_forecast)
                df_reorder = df_reorder[['Date','store_item','sales','pred_val','Safety_Stock','Reorder_Point','Quantity_to_Order','Closing_stock']]
                df_reorder.append(name)
    except:
        print("NO data  available")
        pass

    return df_metrics,df_forecast,df_reorder,df

def load_data():
    uploaded_file = st.file_uploader("Upload CSV file", type="csv")
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        df = df.rename(columns={"0":"Date","1":"store","2":"item","3":"category","4":"sub_category","5":"product","6":"sales"})
        df = df.dropna(axis = 0)
       
        return df

def main():
    st.set_page_config(page_title="Supply Quotient Forecasting", page_icon=":rocket:", layout="wide", initial_sidebar_state="auto",)
    
    # Load the data
    df = load_data()
    
    
    
    
    if df is not None:
        st.subheader("Data Preview")
        st.dataframe(df.head())  # Display the first few rows of the data
        
        # Select independent and dependent variables
        st.subheader("Select Independent and Dependent Variables")
        variables = df.columns.tolist()
        X = st.radio("Select Independent Variable", variables)
        
        # Select multiple dependent variables
        y = st.multiselect("Select Dependent Variables", variables)
        
    # Dropdown options
        options = ['store_2_item_1' ,'store_3_item_1' ,'store_4_item_1' ,'store_5_item_1' ,'store_6_item_1' ,'store_7_item_1' ,'store_8_item_1' ,'store_9_item_1' ,'store_10_item_1' ,'store_1_item_2' ,'store_2_item_2' ,'store_3_item_2' ,'store_4_item_2' ,'store_5_item_2' ,'store_6_item_2' ,'store_7_item_2' ,'store_8_item_2' ,'store_9_item_2' ,'store_10_item_2' ,'store_1_item_3' ,'store_2_item_3' ,'store_3_item_3' ,'store_4_item_3' ,'store_5_item_3' ,'store_6_item_3' ,'store_7_item_3' ,'store_8_item_3' ,'store_9_item_3' ,'store_10_item_3' ,'store_1_item_4' ,'store_2_item_4' ,'store_3_item_4' ,'store_4_item_4' ,'store_5_item_4' ,'store_6_item_4' ,'store_7_item_4' ,'store_8_item_4' ,'store_9_item_4' ,'store_10_item_4' ,'store_1_item_5' ,'store_2_item_5' ,'store_3_item_5' ,'store_4_item_5' ,'store_5_item_5' ,'store_6_item_5' ,'store_7_item_5' ,'store_8_item_5' ,'store_9_item_5' ,'store_10_item_5' ,'store_1_item_6' ,'store_2_item_6' ,'store_3_item_6' ,'store_4_item_6' ,'store_5_item_6' ,'store_6_item_6' ,'store_7_item_6' ,'store_8_item_6' ,'store_9_item_6' ,'store_10_item_6' ,'store_1_item_7' ,'store_2_item_7' ,'store_3_item_7' ,'store_4_item_7' ,'store_5_item_7' ,'store_6_item_7' ,'store_7_item_7' ,'store_8_item_7' ,'store_9_item_7' ,'store_10_item_7' ,'store_1_item_8' ,'store_2_item_8' ,'store_3_item_8' ,'store_4_item_8' ,'store_5_item_8' ,'store_6_item_8' ,'store_7_item_8' ,'store_8_item_8' ,'store_9_item_8' ,'store_10_item_8' ,'store_1_item_9' ,'store_2_item_9' ,'store_3_item_9' ,'store_4_item_9' ,'store_5_item_9' ,'store_6_item_9' ,'store_7_item_9' ,'store_8_item_9' ,'store_9_item_9' ,'store_10_item_9' ,'store_1_item_10' ,'store_2_item_10' ,'store_3_item_10' ,'store_4_item_10' ,'store_5_item_10' ,'store_6_item_10' ,'store_7_item_10' ,'store_8_item_10' ,'store_9_item_10' ,'store_10_item_10' ,'store_1_item_11' ,'store_2_item_11' ,'store_3_item_11' ,'store_4_item_11' ,'store_5_item_11' ,'store_6_item_11' ,'store_7_item_11' ,'store_8_item_11' ,'store_9_item_11' ,'store_10_item_11' ,'store_1_item_12' ,'store_2_item_12' ,'store_3_item_12' ,'store_4_item_12' ,'store_5_item_12' ,'store_6_item_12' ,'store_7_item_12' ,'store_8_item_12' ,'store_9_item_12' ,'store_10_item_12' ,'store_1_item_13' ,'store_2_item_13' ,'store_3_item_13' ,'store_4_item_13' ,'store_5_item_13' ,'store_6_item_13' ,'store_7_item_13' ,'store_8_item_13' ,'store_9_item_13' ,'store_10_item_13' ,'store_1_item_14' ,'store_2_item_14' ,'store_3_item_14' ,'store_4_item_14' ,'store_5_item_14' ,'store_6_item_14' ,'store_7_item_14' ,'store_8_item_14' ,'store_9_item_14' ,'store_10_item_14' ,'store_1_item_15' ,'store_2_item_15' ,'store_3_item_15' ,'store_4_item_15' ,'store_5_item_15' ,'store_6_item_15' ,'store_7_item_15' ,'store_8_item_15' ,'store_9_item_15' ,'store_10_item_15' ,'store_1_item_16' ,'store_2_item_16' ,'store_3_item_16' ,'store_4_item_16' ,'store_5_item_16' ,'store_6_item_16' ,'store_7_item_16' ,'store_8_item_16' ,'store_9_item_16' ,'store_10_item_16' ,'store_1_item_17' ,'store_2_item_17' ,'store_3_item_17' ,'store_4_item_17' ,'store_5_item_17' ,'store_6_item_17' ,'store_7_item_17' ,'store_8_item_17' ,'store_9_item_17' ,'store_10_item_17' ,'store_1_item_18' ,'store_2_item_18' ,'store_3_item_18' ,'store_4_item_18' ,'store_5_item_18' ,'store_6_item_18' ,'store_7_item_18' ,'store_8_item_18' ,'store_9_item_18' ,'store_10_item_18' ,'store_1_item_19' ,'store_2_item_19' ,'store_3_item_19' ,'store_4_item_19' ,'store_5_item_19' ,'store_6_item_19' ,'store_7_item_19' ,'store_8_item_19' ,'store_9_item_19' ,'store_10_item_19' ,'store_1_item_20' ,'store_2_item_20' ,'store_3_item_20' ,'store_4_item_20' ,'store_5_item_20' ,'store_6_item_20' ,'store_7_item_20' ,'store_8_item_20' ,'store_9_item_20' ,'store_10_item_20' ,'store_1_item_21' ,'store_2_item_21' ,'store_3_item_21' ,'store_4_item_21' ,'store_5_item_21' ,'store_6_item_21' ,'store_7_item_21' ,'store_8_item_21' ,'store_9_item_21' ,'store_10_item_21' ,'store_1_item_22' ,'store_2_item_22' ,'store_3_item_22' ,'store_4_item_22' ,'store_5_item_22' ,'store_6_item_22' ,'store_7_item_22' ,'store_8_item_22' ,'store_9_item_22' ,'store_10_item_22' ,'store_1_item_23' ,'store_2_item_23' ,'store_3_item_23' ,'store_4_item_23' ,'store_5_item_23' ,'store_6_item_23' ,'store_7_item_23' ,'store_8_item_23' ,'store_9_item_23' ,'store_10_item_23' ,'store_1_item_24' ,'store_2_item_24' ,'store_3_item_24' ,'store_4_item_24' ,'store_5_item_24' ,'store_6_item_24' ,'store_7_item_24' ,'store_8_item_24' ,'store_9_item_24' ,'store_10_item_24' ,'store_1_item_25' ,'store_2_item_25' ,'store_3_item_25' ,'store_4_item_25' ,'store_5_item_25' ,'store_6_item_25' ,'store_7_item_25' ,'store_8_item_25' ,'store_9_item_25' ,'store_10_item_25' ,'store_1_item_26' ,'store_2_item_26' ,'store_3_item_26' ,'store_4_item_26' ,'store_5_item_26' ,'store_6_item_26' ,'store_7_item_26' ,'store_8_item_26' ,'store_1_item_1' ,'store_9_item_26' ,'store_10_item_26' ,'store_1_item_27' ,'store_2_item_27' ,'store_3_item_27' ,'store_4_item_27' ,'store_5_item_27' ,'store_6_item_27' ,'store_7_item_27' ,'store_8_item_27' ,'store_9_item_27' ,'store_10_item_27' ,'store_1_item_28' ,'store_2_item_28' ,'store_3_item_28' ,'store_4_item_28' ,'store_5_item_28' ,'store_6_item_28' ,'store_7_item_28' ,'store_8_item_28' ,'store_9_item_28' ,'store_10_item_28' ,'store_1_item_29' ,'store_2_item_29' ,'store_3_item_29' ,'store_4_item_29' ,'store_5_item_29' ,'store_6_item_29' ,'store_7_item_29' ,'store_8_item_29' ,'store_9_item_29' ,'store_10_item_29' ,'store_1_item_30' ,'store_2_item_30' ,'store_3_item_30' ,'store_4_item_30' ,'store_5_item_30' ,'store_6_item_30' ,'store_7_item_30' ,'store_8_item_30' ,'store_9_item_30' ,'store_10_item_30' ,'store_1_item_31' ,'store_2_item_31' ,'store_3_item_31' ,'store_4_item_31' ,'store_5_item_31' ,'store_6_item_31' ,'store_7_item_31' ,'store_8_item_31' ,'store_9_item_31' ,'store_10_item_31' ,'store_1_item_32' ,'store_2_item_32' ,'store_3_item_32' ,'store_4_item_32' ,'store_5_item_32' ,'store_6_item_32' ,'store_7_item_32' ,'store_8_item_32' ,'store_9_item_32' ,'store_10_item_32' ,'store_1_item_33' ,'store_2_item_33' ,'store_3_item_33' ,'store_4_item_33' ,'store_5_item_33' ,'store_6_item_33' ,'store_7_item_33' ,'store_8_item_33' ,'store_9_item_33' ,'store_10_item_33' ,'store_1_item_34' ,'store_2_item_34' ,'store_3_item_34' ,'store_4_item_34' ,'store_5_item_34' ,'store_6_item_34' ,'store_7_item_34' ,'store_8_item_34' ,'store_9_item_34' ,'store_10_item_34' ,'store_1_item_35' ,'store_2_item_35' ,'store_3_item_35' ,'store_4_item_35' ,'store_5_item_35' ,'store_6_item_35' ,'store_7_item_35' ,'store_8_item_35' ,'store_9_item_35' ,'store_10_item_35' ,'store_1_item_36' ,'store_2_item_36' ,'store_3_item_36' ,'store_4_item_36' ,'store_5_item_36' ,'store_6_item_36' ,'store_7_item_36' ,'store_8_item_36' ,'store_9_item_36' ,'store_10_item_36' ,'store_1_item_37' ,'store_2_item_37' ,'store_3_item_37' ,'store_4_item_37' ,'store_5_item_37' ,'store_6_item_37' ,'store_7_item_37' ,'store_8_item_37' ,'store_9_item_37' ,'store_10_item_37' ,'store_1_item_38' ,'store_2_item_38' ,'store_3_item_38' ,'store_4_item_38' ,'store_5_item_38' ,'store_6_item_38' ,'store_7_item_38' ,'store_8_item_38' ,'store_9_item_38' ,'store_10_item_38' ,'store_1_item_39' ,'store_2_item_39' ,'store_3_item_39' ,'store_4_item_39' ,'store_5_item_39' ,'store_6_item_39' ,'store_7_item_39' ,'store_8_item_39' ,'store_9_item_39' ,'store_10_item_39' ,'store_1_item_40' ,'store_2_item_40' ,'store_3_item_40' ,'store_4_item_40' ,'store_5_item_40' ,'store_6_item_40' ,'store_7_item_40' ,'store_8_item_40' ,'store_9_item_40' ,'store_10_item_40' ,'store_1_item_41' ,'store_2_item_41' ,'store_3_item_41' ,'store_4_item_41' ,'store_5_item_41' ,'store_6_item_41' ,'store_7_item_41' ,'store_8_item_41' ,'store_9_item_41' ,'store_10_item_41' ,'store_1_item_42' ,'store_2_item_42' ,'store_3_item_42' ,'store_4_item_42' ,'store_5_item_42' ,'store_6_item_42' ,'store_7_item_42' ,'store_8_item_42' ,'store_9_item_42' ,'store_10_item_42' ,'store_1_item_43' ,'store_2_item_43' ,'store_3_item_43' ,'store_4_item_43' ,'store_5_item_43' ,'store_6_item_43' ,'store_7_item_43' ,'store_8_item_43' ,'store_9_item_43' ,'store_10_item_43' ,'store_1_item_44' ,'store_2_item_44' ,'store_3_item_44' ,'store_4_item_44' ,'store_5_item_44' ,'store_6_item_44' ,'store_7_item_44' ,'store_8_item_44' ,'store_9_item_44' ,'store_10_item_44' ,'store_1_item_45' ,'store_2_item_45' ,'store_3_item_45' ,'store_4_item_45' ,'store_5_item_45' ,'store_6_item_45' ,'store_7_item_45' ,'store_8_item_45' ,'store_9_item_45' ,'store_10_item_45' ,'store_1_item_46' ,'store_2_item_46' ,'store_3_item_46' ,'store_4_item_46' ,'store_5_item_46' ,'store_6_item_46' ,'store_7_item_46' ,'store_8_item_46' ,'store_9_item_46' ,'store_10_item_46' ,'store_1_item_47' ,'store_2_item_47' ,'store_3_item_47' ,'store_4_item_47' ,'store_5_item_47' ,'store_6_item_47' ,'store_7_item_47' ,'store_8_item_47' ,'store_9_item_47' ,'store_10_item_47' ,'store_1_item_48' ,'store_2_item_48' ,'store_3_item_48' ,'store_4_item_48' ,'store_5_item_48' ,'store_6_item_48' ,'store_7_item_48' ,'store_8_item_48' ,'store_9_item_48' ,'store_10_item_48' ,'store_1_item_49' ,'store_2_item_49' ,'store_3_item_49' ,'store_4_item_49' ,'store_5_item_49' ,'store_6_item_49' ,'store_7_item_49' ,'store_8_item_49' ,'store_9_item_49' ,'store_10_item_49' ,'store_1_item_50' ,'store_2_item_50' ,'store_3_item_50' ,'store_4_item_50' ,'store_5_item_50' ,'store_6_item_50' ,'store_7_item_50' ,'store_8_item_50' ,'store_9_item_50' ,'store_10_item_50']
        # Selectbox with manual input
        selected_options = st.selectbox("Select an option or enter a value", options )
        st.write("Selected option:", selected_options)
        
        # val_period = st.selectbox("Select an option", ["store_3_item_1", "store_3_item_2", "store_3_item_3"])
        # val_period = st.radio("Select a val_period", [7, 14, 21,30])
        options = [7, 14, 21,30]
    
        # Radio button with manual input
        val_period = st.radio("Select a validation period or enter a value", options + ['Enter manually'])
        
        if val_period == 'Enter manually':
            # Text input for manual entry
            val_period = st.text_input("Enter a value")
            st.write("You entered:", val_period)
        else:
            try:
                val_period = int(val_period)
                st.write("Selected option:", val_period)
            except ValueError:
                st.write("Invalid option selected")
        min_value = 0.00
        max_value = 1.00
        step = 0.05
        
        confidence_level =  st.slider("Select a confidence value", min_value, max_value, step,format="%.2f")
        percentage_value = format((1-confidence_level) * 100, '.0f')
        percentage_string = '{}%'.format(percentage_value)
        st.write("Selected confidence level:", percentage_string)
        # # confidence_level = st.number_input(label="confidence level",step=0.05.,format="%.2f")
        # percentage_value = format((1-confidence_level) * 100, '.0f')
        # percentage_string = '{}%'.format(percentage_value)
        
        # st.write("Selected confidence level:", percentage_string)
        
        
        
       
            # Check if there is input data
        if  len(df) ==0:
            st.warning("There is no Data for this selection, Please select another one.")
             # Exit the function if there is no input data
     
        # Process the input data
        if st.button('submit'):
            val_period = int(val_period)
            df_metrics,df_forecast,df_reorder,df1 = aro(df,selected_options,val_period)
            df_reorder = df_reorder[['Date','store_item','sales','pred_val','Safety_Stock','Reorder_Point','Quantity_to_Order']]
            df1 = df1[:-val_period]
            st.write("metrics")
            st.dataframe(df_metrics)
         # st.write("forecast")
         # st.dataframe(df_forecast)
            st.write("reorder")
            st.dataframe(df_reorder)
            pred_val = pd.DataFrame(df_forecast[['Date','pred_val']])
            act_val = pd.DataFrame(df1[['Date','sales']]).tail(val_period)
            act_val = act_val['sales'].tolist()
            pred_val = pred_val['pred_val'].tolist()
         
            def weighted_quantile_loss(y_true, y_pred, alpha, weights):
                  errors = np.array(act_val) - np.array(pred_val)
                  quantiles = [alpha, 1 - alpha]
                  loss = np.maximum(quantiles[0] * errors, quantiles[1] * errors)
                  weighted_loss = np.dot(loss, weights)
                  return weighted_loss
 
            def calculate_confidence_intervals(act_val, pred_val, alpha, weights, num_bootstraps):
                  losses = []
                  
                  for i in range(num_bootstraps):
                      # Resample the data with replacement
                      resampled_true = resample(act_val)
                      resampled_pred = resample(pred_val)
                      
                      # Calculate the weighted quantile loss for the resampled data
                      loss = weighted_quantile_loss(resampled_true, resampled_pred, alpha, weights)
                      losses.append(loss)
                  
                  # Calculate the lower and upper percentiles
                  lower_percentile = alpha * 100 / 2
                  upper_percentile = 100 - lower_percentile
                  
                  # Calculate the confidence intervals
                  lower_bound = np.percentile(losses, lower_percentile)
                  upper_bound = np.percentile(losses, upper_percentile)
                  
                  return lower_bound, upper_bound 
         
        
         # confidence_level =  st.slider("Select a confidence value", min_value, max_value, step,format="%.2f")
       
       
        
            alpha = confidence_level  
            weights = np.ones_like(act_val)  # Equal weights for each observation
            num_bootstraps = 1000
 
            lower_bound, upper_bound = calculate_confidence_intervals(act_val, pred_val, alpha, weights, num_bootstraps)
        
            if lower_bound < 0:
                
                lower_bound = [math.ceil(round(num)) + lower_bound for num in pred_val]
                upper_bound = [math.ceil(round(num)) - upper_bound for num in pred_val]
            else:
                
                lower_bound = [math.ceil(round(num)) - lower_bound for num in pred_val]
                upper_bound = [math.ceil(round(num)) + upper_bound for num in pred_val]
                 
        
            df_forecast['lower_bound'] = pd.DataFrame(lower_bound)
            df_forecast['upper_bound'] = pd.DataFrame(upper_bound)
    
            df_forecast['lower_bound'] = round(df_forecast['lower_bound'],0)
            df_forecast['upper_bound'] = round(df_forecast['upper_bound'],0)
    
            df_forecast = df_forecast[['Date', 'store_item', 'sales', 'pred_val','lower_bound','upper_bound','error', 'Accuracy',
                         'Avg_Accuracy']]
    
            st.write("forecast")
            st.dataframe(df_forecast)
        
            fig, ax = plt.subplots(figsize=(4, 4))
         # fig, ax = plt.subplots()
            ax.plot(act_val, label='Actual')
            ax.plot(pred_val, label='Forecasted')
            ax.fill_between(range(len(act_val)), lower_bound, upper_bound, alpha=confidence_level, label='Confidence Intervals')
     
     
     
            ax.legend()
     
         # Display the plot using Streamlit
            st.write("Selected confidence value:", confidence_level)  
            st.title("Actual, Forecasted, and Confidence Intervals")
     
            st.pyplot(fig)

           
    return
               
     
if __name__=="__main__":
    main()
       
