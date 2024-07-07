import yfinance as yf
import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller, coint
import matplotlib.pyplot as plt



def go(startDate, endDate):
    
    july24 = yf.download('HG=F', start=startDate, end=endDate)
    copx = yf.download('copx', start=startDate, end=endDate)

    july24["july24_dailyReturns"] = july24["Close"].pct_change().dropna()
    copx["copx_dailyReturns"] = copx["Close"].pct_change().dropna()

    #stationary test
    print("Test for stationary futures")
    stationaryTest(july24["july24_dailyReturns"].dropna())
    print("Test for stationary etf")
    stationaryTest(copx["copx_dailyReturns"].dropna())
    print("cointegration Test for stationary futures vs etf")
    cointegrationTest(july24["july24_dailyReturns"].dropna(), copx["copx_dailyReturns"].dropna())

    combined_df = pd.concat([july24, copx], axis=1)
    combined_df["spread"] = combined_df["july24_dailyReturns"] - combined_df["copx_dailyReturns"]
    combined_df["abs_sprd"] = abs(combined_df["spread"])
    combined_df["july24_rolling avg"] = combined_df["july24_dailyReturns"].rolling(window=30).std()
    combined_df["copx_rolling avg"] = combined_df["copx_dailyReturns"].rolling(window=30).std()

    plt.figure(figsize=(12,6))
    plt.plot(combined_df["july24_dailyReturns"], label="futures DR")
    plt.plot(combined_df["copx_dailyReturns"], label="copx DR")
    plt.plot(combined_df["july24_rolling avg"], label="fut rolling")
    plt.plot(combined_df["copx_rolling avg"], label="etf rolling")
    plt.plot(combined_df["abs_sprd"], label="sprd", linestyle="--")
    plt.legend()
    plt.show()
    
    df = signal(combined_df)
    df = df_column_uniquify(df)
    df = pnl(df, 1)
    
    print()
    
    print(df['fut cash'].sum())
    print(df["etf cash"].sum())
    print("total pnl = {}".format(df["fut cash"].sum()+df["etf cash"].sum()))

    df.to_csv("copper Fut VS ETF.csv", index=False)

    print("abs sprd = {}".format(df["abs_sprd"].tail(1).values))
    print("fut 30-day rolling avg = {}".format(df["july24_rolling avg"].tail(1).values))
    print("copx 30-day rolling avg = {}".format(df["copx_rolling avg"].tail(1).values))
    
    return None
    
def signal(df):
    
    df.reset_index(inplace=True)
    
    df["isSprdPos"] = np.where(df["spread"] >= 0, 1, 0)
    df["signal"] = 0 
    
    i = 30
    
    while i < len(df.index):
        
        absSprd = df.loc[i, "abs_sprd"]
        fut_rolling_avg = df.loc[i, "july24_rolling avg"]
        etf_rolling_avg = df.loc[i, "copx_rolling avg"]
        
        if absSprd > fut_rolling_avg and absSprd > etf_rolling_avg:
            df.loc[i, "signal"] = 1
            currSprdDir = df.loc[i, "isSprdPos"]
            # print(i, currSprdDir)
            
            i += 1
            
            while i < len(df.index) and currSprdDir == df.loc[i, "isSprdPos"]:
                df.loc[i, "signal"] = 1
                i += 1
            
            if i < len(df.index):
                df.loc[i, "signal"] = 0      
        
        i += 1
    
    df["axe"] = df["signal"].diff()
    
    return df

def df_column_uniquify(df):
    df_columns = df.columns
    new_columns = []
    for item in df_columns:
        counter = 0
        newitem = item
        while newitem in new_columns:
            counter += 1
            newitem = "{}_{}".format(item, counter)
        new_columns.append(newitem)
    df.columns = new_columns
    return df

def pnl(df, leverage):
    df["leverage"] = leverage
    df["ratio"] = abs(df["Close_1"] * df["copx_dailyReturns"]) / abs(df["Close"] * df["july24_dailyReturns"])
    
    # Initialize new columns
    df["fut ntl"] = 0.0
    df["etf ntl"] = 0.0
    df["fut cash"] = 0.0
    df["etf cash"] = 0.0
    
    i = 30
    currFutNtl, currEtfNtl = 0, 0
    
    while i < len(df.index):
        
        if df.loc[i, "axe"] == 1:
            
            if abs(df.loc[i, 'july24_dailyReturns']) > abs(df.loc[i, 'copx_dailyReturns']):
                if df.loc[i, 'july24_dailyReturns'] > 0:
                    df.loc[i, "fut ntl"] = df.loc[i, "ratio"] * -1 * df.loc[i, "leverage"]
                    df.loc[i, "etf ntl"] = df.loc[i, "leverage"]
                else:
                    df.loc[i, "fut ntl"] = df.loc[i, "ratio"] * df.loc[i, "leverage"]
                    df.loc[i, "etf ntl"] = -1 * df.loc[i, "leverage"]
            else:
                if df.loc[i, 'copx_dailyReturns'] > 0:
                    df.loc[i, "fut ntl"] = df.loc[i, "ratio"] * df.loc[i, "leverage"]
                    df.loc[i, "etf ntl"] = -1 * df.loc[i, "leverage"]
                else:
                    df.loc[i, "fut ntl"] = df.loc[i, "ratio"] * -1 * df.loc[i, "leverage"]
                    df.loc[i, "etf ntl"] = df.loc[i, "leverage"]
                    
            df.loc[i, "fut cash"] = df.loc[i, "fut ntl"] * df.loc[i, "Close"] * -1
            df.loc[i, "etf cash"] = df.loc[i, "etf ntl"] * df.loc[i, "Close_1"] * -1
                    
            currFutNtl = df.loc[i, "fut ntl"]
            currEtfNtl = df.loc[i, "etf ntl"]

        elif df.loc[i, "axe"] == -1:
            df.loc[i, "fut cash"] = currFutNtl * df.loc[i, "Close"]
            df.loc[i, "etf cash"] =  currEtfNtl * df.loc[i, "Close_1"]
            
            currFutNtl, currEtfNtl = 0, 0
    
        i += 1
        
    return df

def stationaryTest(val):
    print()
    result = adfuller(val)
    # print(result)
    
    print('ADF Statistic:', result[0])
    print('p-value:', result[1])
    
    for key, value in result[4].items():
        print('Critical Value ({}): {:.3f}'.format(key, value))

    # Interpreting the result
    if result[1] < 0.05:
        print("The series is stationary.")
    else:
        print("The series is non-stationary.")
        
def cointegrationTest(val1, val2):
    
    print()
    
    coint_result = coint(val1, val2)
    coint_statistic = coint_result[0]
    p_value = coint_result[1]
    critical_values = coint_result[2]

    print(f'Cointegration Test Statistic: {coint_statistic}')
    print(f'p-value: {p_value}')
    print('Critical Values:')
    for cv, value in zip(['1%', '5%', '10%'], critical_values):
        print(f'   {cv}: {value}')

    # Interpretation
    if p_value < 0.05:
        print("The series are cointegrated (stationary together).")
    else:
        print("The series are not cointegrated (not stationary together).")


startDate = "2023-12-01"
endDate = "2024-07-05"

go(startDate, endDate)
