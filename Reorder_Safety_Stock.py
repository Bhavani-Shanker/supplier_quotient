import streamlit as st
import time
start = time.time()
import pandas as pd
import numpy as np
from scipy import stats
NormalDist = stats.norm
from termcolor import colored
@st.cache_resource
def re_oreder(df_forecast):
    df_forecast["pred_val"] = round(df_forecast["pred_val"],0)
    df_sub = df_forecast.copy(deep=True)
    '''
    
    # The below data is for testing purpose
    i = 1
    j = "SAMSUNG 40-inch Class LED Smart FHD TV 1080P"
    k = 7
    Expected_lead_Time = 3
    '''
    
    df_final= pd.DataFrame()
    
    # df_sub = df[(df['Store_ID']==i) & (df['Product']==j)]
    df_sub['SMA7'] = df_sub['pred_val'].rolling(7).mean()
    df_sub['Sd7'] = df_sub['pred_val'].rolling(7).std()
    df_sub["Safety_Stock"] = 0.0
    df_sub["Reorder_Point"] = 0.0
    df_sub["Quantity_to_Order"] = 0.0
    df_sub["Expected_lead_Time"] = 3
    df_sub["Closing_Stock"] = np.random.randint(9,23, size=len(df_sub))
    for k in range(0,len(df_sub)):
        if df_sub['SMA7'].values[k] > 0 :
         D  = df_sub['SMA7'].values[k]
         sD = df_sub['Sd7'].values[k]
         L =  (df_sub['Expected_lead_Time'].values[k])/7
         T = 4/7
         CSL= 0.95
         timeunit = "week"
         print(colored("This is what you entered: \n________________________\n",color='blue'))
         print("Demand                       = {} per {}".format(D, timeunit))
         print("Standard deviation of demand = {} per {}".format(sD, timeunit))
         print("Leadtime                     = {} {}s".format(L,timeunit))
         print("Length of Review Period      = {} {}s".format(T,timeunit))
         print("Desired service level        = {}%".format(CSL*100.0))
         DTL = D*(T+L)
         sL = sD*np.sqrt(T+L)
         print("Mean demand during leadtime + review period  = {}".format(DTL))
         print("SD of demand during leadtime + review period = {:.2f}".format(sL))
         ss = NormalDist.isf(1-CSL, 0, 1)*sL
         OUL = DTL + ss
         df_sub["Safety_Stock"].values[k] = ss
         df_sub["Reorder_Point"].values[k] = OUL
         print("Safety Stock, ss   = {:.2f}".format(ss))
         print("Order up to Level, OUL = {:.2f}".format(OUL))
         a = 1
         while a == 1:
            I = df_sub['Closing_Stock'].values[k]
            try:
              inI = float(I)
            except ValueError:
              print("Inventory has to be a number")
            else:
              a = 0   
            df_sub['Quantity_to_Order'].values[k] =  OUL - inI 
    df_final = df_final.append(df_sub)
    print("*********************************************\n")
    print("Inventory on Hand                    = {:.2f}".format(inI))
    print("Order up to Level, OUL               = {:.2f}".format(OUL))
    print("Quantity to Order, Q                 = {:.2f}".format(OUL - inI))
    df_final = round(df_final,0)
    
    end = time.time() - start
    print(f'Finished in {round(end, 2)} second(s)')
    return df_final

if __name__=="__main__":
    main()
'''
df_final = re_oreder(df_forecast)


'''
