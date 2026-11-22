import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def data_cleaning(df):
    df = df.copy()
    df['DlyCalDt'] = pd.to_datetime(df['DlyCalDt'])
    df = df.sort_values(by=['PERMNO', 'DlyCalDt']).reset_index(drop=True)
    assert df.duplicated(subset=['PERMNO', 'DlyCalDt']).sum()==0 #check duplicates
    ## only keep stocks that cover the whole period  
    dcal = df['PERMNO'].value_counts()
    stock_ids = dcal[dcal==1510].index
    df = df[df['PERMNO'].isin(stock_ids)]

    #drop columns that only has 1 values
    drop_cols = [col for col in df.columns if df[col].nunique() == 1]
    drop_cols+=['HdrCUSIP','MbrStartDt','MbrEndDt','DlyRetMissFlg','DlyMMCnt','DlyNumTrd']
    df = df.drop(columns=drop_cols)
    print ("Observations:", len(df))
    print ("Variables:", len(df.columns))

    small_nan_cols = df.columns[df.isnull().any()].difference(['DlyRetMissFlg','DlyNumTrd', 'DlyMMCnt']).to_list()
    # dn = df[df[small_nan_cols].isna().any(axis=1)]
    # print(dn)
    ### patch for BIIB 
    m = df['Ticker'].eq('BIIB')
    df.loc[m, ['DlyVol']] = df.loc[m, ['DlyVol']].fillna(0)
    df.loc[m] = df.loc[m].ffill()
    return df



def data_featuring(df):
    df = df.copy()
    g = df.groupby('PERMNO')
    df['LogRet'] = np.log1p(df['DlyRet'])
    df['RetLead1m'] = df['DlyRet'].shift(-1) ### Y target
    df['mom5']=  g['LogRet'].transform(lambda s: s.rolling(5).sum())
    df['mom21']=  g['LogRet'].transform(lambda s: s.rolling(21).sum())
    df['vol5'] = g['LogRet'].transform(lambda s: s.rolling(3, min_periods=3).std()) * np.sqrt(252)
    df['vol10'] = g['LogRet'].transform(lambda s: s.rolling(21, min_periods=10).std()) * np.sqrt(252)
    df['spread'] = (df['DlyAsk']-df['DlyBid'])/(df['DlyBid']+df['DlyAsk'])/2  #(Ask - Bid) / ((Ask + Bid) / 2
    df['ShrOut'] = df['ShrOut'] * df['DlyCumFacShr'] #drop cumfacshr after use
    df['mktcap'] = np.log10(df['DlyClose']*df['ShrOut'])
    df = df.drop(columns=['DlyCumFacShr'])
    # print (df.loc[df['Ticker']=='CBOE',['Ticker','DlyCalDt','DlyClose','DlyRet','RetLead1m','mom5']].head(10))
    return df


df = pd.read_csv('data/ewwi65k5su2a3ly7.csv')
df_filter = data_cleaning(df)
df_output = data_featuring(df_filter)
df_output.to_csv('cleaned_stock.csv')