import pandas as pd
import time
from dask.distributed import LocalCluster
from concurrent.futures import ThreadPoolExecutor
from evalml.automl.engine.cf_engine import CFEngine, CFClient

start_time = time.time()
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import mean_absolute_error,median_absolute_error,mean_absolute_percentage_error,mean_squared_error,mean_squared_log_error,r2_score
import numpy as np
from tqdm import tqdm
from importlib import reload
import evalml
from evalml.automl import AutoMLSearch
from evalml.automl import AutoMLSearch
import joblib
import multiprocess
import numpy as np
import time
import warnings
warnings.filterwarnings("ignore")
import multiprocess
import pandas as pd
# df = pd.read_excel("D:/ARO Forecasting/Final_Data.xlsx",engine='openpyxl')

#################################################################

#############################################################

 

def eval_ml(i,df,val_period):
    # for i in tqdm(all_ts):
        import time
        start_time = time.time()
        
        data = df.copy(deep=True)
        try:
            print (i)
            df_metrics_temp = pd.DataFrame(columns = [['store_item','train_test_val','mean_absolute_error','median_absolute_error','mean_absolute_percentage_error','mean_squared_error','rmse','r2_score','Time_taken']])
            val_y_pred =  pd.DataFrame()
            
            df_metrics = pd.DataFrame()
            df_subset= data[data['time_series'] == i]
            # df_subset = df_subset.drop(['Unnamed: 12'],axis=1)
            df_subset = df_subset.dropna(how='any')
            
            
            
            val_data = df_subset.tail(val_period)
            df_subset = df_subset[:-val_period]
            
            X = df_subset.drop(["sales"],axis=1)
            y = df_subset[["sales"]]
            X = X.fillna(0)
            y = y.fillna(0)
            X = X.squeeze() 
            y = y.squeeze() 
            
            X_train, X_test, y_train, y_test = evalml.preprocessing.split_data(X, y, problem_type='regression')
            # evalml.problem_types.ProblemTypes.all_problem_types
            # from evalml import AutoMLSearch
            cpu_count = multiprocess.cpu_count()
            cf_engine = CFEngine(CFClient(ThreadPoolExecutor(max_workers=cpu_count)))
            # automl = AutoMLSearch(X_train=X_train, y_train=y_train, problem_type="regression",
            #                       objective="r2",engine="cf_threaded",n_jobs=-1) 
            

            automl = AutoMLSearch(X_train=X, y_train=y,
                                  problem_type="regression",                                 
                                  engine=cf_engine)
            automl.search()
            automl.close_engine()             
            # automl = AutoMLSearch(X_train=X_train, y_train=y_train, problem_type="regression",
            #                       objective="r2")

            # automl = AutoMLSearch(X_train=X_train, y_train=y_train, problem_type="regression",
            #                       objective="r2",engine="dask_engine_p4")              
            # automl.search()
            # automl.rankings
            pipeline = automl.best_pipeline
            pipeline.predict(X_test)
            
            val_y_pred['pred_val'] = pipeline.predict(val_data.drop(['sales'],axis=1))
            
            val_y_pred['sales'] = val_data['sales']
            val_y_pred['Date'] = val_data['Date']
            val_y_pred['store_item'] = val_data['time_series']
            
            
            
            
            # save the model to disk
            filename = 'finalized_evalml_model_'+i+'.sav'
            joblib.dump(pipeline, filename)
            
            df_metrics_temp['store_item']  = [i]
            df_metrics_temp['train_test_val']  = ['Train']
            df_metrics_temp['mean_absolute_error']  = [mean_absolute_error(y_train,pipeline.predict(X_train))]
            df_metrics_temp['median_absolute_error']  = [median_absolute_error(y_train,pipeline.predict(X_train))]
            df_metrics_temp['mean_absolute_percentage_error']  = [mean_absolute_percentage_error(y_train,pipeline.predict(X_train))]
            df_metrics_temp['mean_squared_error']  = [mean_squared_error(y_train,pipeline.predict(X_train))]
            df_metrics_temp['rmse']  = [np.sqrt(df_metrics_temp['mean_squared_error'].values[0][0])]
            df_metrics_temp['r2_score']  = [r2_score(y_train,pipeline.predict(X_train))]
            df_metrics_temp['Time_taken'] = [time.time() - start_time]
            df_metrics = pd.concat([df_metrics, df_metrics_temp], ignore_index=True, sort=False)
            
            # preedict the test data & validation data and capture the metrics
            df_metrics_temp['store_item']  = [i]
            df_metrics_temp['train_test_val']  = ['Test']
            df_metrics_temp['mean_absolute_error']  = [mean_absolute_error(y_test,pipeline.predict(X_test))]
            df_metrics_temp['median_absolute_error']  = [median_absolute_error(y_test,pipeline.predict(X_test))]
            df_metrics_temp['mean_absolute_percentage_error']  = [mean_absolute_percentage_error(y_test,pipeline.predict(X_test))]
            df_metrics_temp['mean_squared_error']  = [mean_squared_error(y_test,pipeline.predict(X_test))]
            df_metrics_temp['rmse']  = [np.sqrt(df_metrics_temp['mean_squared_error'].values[0][0])]
            df_metrics_temp['r2_score']  = [r2_score(y_test,pipeline.predict(X_test))]
            df_metrics_temp['Time_taken'] = [time.time() - start_time]
            df_metrics = pd.concat([df_metrics, df_metrics_temp], ignore_index=True, sort=False)   
            
            # j = 2
            # for col in ['category', 'sub_category', 'product', 'time_series']:
            #     val_data[col] = val_data[col].astype('category')
            # #automl = AutoMLSearch(X_train=X, y_train=y, problem_type="regression", objective="r2") 
            # for j in range(val_data.shape[0]):
            # # fit the model on all previous data
                
            #     # make a prediction for the next row
            #     future = pipeline.predict(val_data.drop(["sales"],axis=1).iloc[[j]])
            #     temp_val = pd.DataFrame(columns = ['Date','store_item','pred_val'])
            #     temp_val['store_item']  = [i]
                
            #     temp_val["pred_val"] = [future.values[0]]
            #     temp_val["Date"] = val_data["Date"].values[j]
                
            #     #val_y_pred = pd.concat([val_y_pred, temp_val], ignore_index=True, sort=False)
            #     val_y_pred = val_y_pred.append(temp_val)
            #     val_y_pred["sales"] = val_data['sales'].values[j]
            
            val_y_pred = val_y_pred[['Date','store_item','sales','pred_val']].reset_index(drop=True)
        
            df_metrics_temp['store_item']  = [i]
            df_metrics_temp['train_test_val']  = ['Val']
            df_metrics_temp['mean_absolute_error']  = [mean_absolute_error(val_data[['sales']],pipeline.predict(val_data.drop(['sales'],axis = 1)))]
            df_metrics_temp['median_absolute_error']  = [median_absolute_error(val_data[['sales']],pipeline.predict(val_data.drop(['sales'],axis = 1)))]
            df_metrics_temp['mean_absolute_percentage_error']  = [mean_absolute_percentage_error(val_data[['sales']],pipeline.predict(val_data.drop(['sales'],axis = 1)))]
            df_metrics_temp['mean_squared_error']  = [mean_squared_error(val_data[['sales']],pipeline.predict(val_data.drop(['sales'],axis = 1)))]
            df_metrics_temp['rmse']  = [np.sqrt(df_metrics_temp['mean_squared_error'].values[0][0])]
            df_metrics_temp['r2_score']  = [r2_score(val_data[['sales']],pipeline.predict(val_data.drop(['sales'],axis = 1)))]
            df_metrics_temp['Time_taken'] = [time.time() - start_time]
            
            df_metrics = pd.concat([df_metrics, df_metrics_temp], ignore_index=True, sort=False)
            
        except:
            df_metrics_temp['train_test_val']  = ['Train']
            df_metrics_temp['store_item']  = [i]
            df_metrics_temp['train_test_val']  = ['Test']
            df_metrics_temp['store_item']  = [i]
            df_metrics_temp['train_test_val']  = ['Val']
            df_metrics_temp['store_item']  = [i]
            
            df_metrics = pd.concat([df_metrics, df_metrics_temp], ignore_index=True, sort=False)
            pass
        
        return df_metrics,val_y_pred
   









