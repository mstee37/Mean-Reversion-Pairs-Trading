import yfinance as yf
import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller, coint
import matplotlib.pyplot as plt

    
def signal(df,ticker1,ticker2):
    
    df.reset_index(inplace=True)
    
    df["isSprdPos"] = np.where(df["spread"] >= 0, 1, 0)
    df["signal"] = 0 
    
    i = 30
    
    while i < len(df.index):
        
        absSprd = df.loc[i, "abs_sprd"]
        stock1_rolling_avg = df.loc[i, ticker1+"_rolling avg"]
        stock2_rolling_avg = df.loc[i, ticker2+"_rolling avg"]
        
        if absSprd > stock1_rolling_avg and absSprd > stock2_rolling_avg:
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

def pnl(df, leverage, ticker1, ticker2):
    df["leverage"] = leverage
    df["ratio"] = abs(df["Close_1"] * df[ticker2+"_dailyReturns"]) / abs(df["Close"] * df[ticker1+"_dailyReturns"])
    
    # Initialize new columns
    df["stock1 ntl"] = 0.0
    df["stock2 ntl"] = 0.0
    df["stock1 cash"] = 0.0
    df["stock2 cash"] = 0.0
    
    i = 30
    currFutNtl, currEtfNtl = 0, 0
    
    # naming convention
    dailyReturn1 = ticker1+"_dailyReturns"
    dailyReturn2 = ticker2+"_dailyReturns"
    
    while i < len(df.index):
        
        if df.loc[i, "axe"] == 1:
            
            if abs(df.loc[i, dailyReturn1]) > abs(df.loc[i, dailyReturn2]):
                if df.loc[i, dailyReturn1] > 0:
                    df.loc[i, "stock1 ntl"] = df.loc[i, "ratio"] * -1 * df.loc[i, "leverage"]
                    df.loc[i, "stock2 ntl"] = df.loc[i, "leverage"]
                else:
                    df.loc[i, "stock1 ntl"] = df.loc[i, "ratio"] * df.loc[i, "leverage"]
                    df.loc[i, "stock2 ntl"] = -1 * df.loc[i, "leverage"]
            else:
                if df.loc[i, dailyReturn2] > 0:
                    df.loc[i, "stock1 ntl"] = df.loc[i, "ratio"] * df.loc[i, "leverage"]
                    df.loc[i, "stock2 ntl"] = -1 * df.loc[i, "leverage"]
                else:
                    df.loc[i, "stock1 ntl"] = df.loc[i, "ratio"] * -1 * df.loc[i, "leverage"]
                    df.loc[i, "stock2 ntl"] = df.loc[i, "leverage"]
                    
            df.loc[i, "stock1 cash"] = df.loc[i, "stock1 ntl"] * df.loc[i, "Close"] * -1
            df.loc[i, "stock2 cash"] = df.loc[i, "stock2 ntl"] * df.loc[i, "Close_1"] * -1
                    
            currFutNtl = df.loc[i, "stock1 ntl"]
            currEtfNtl = df.loc[i, "stock2 ntl"]

        elif df.loc[i, "axe"] == -1:
            df.loc[i, "stock1 cash"] = currFutNtl * df.loc[i, "Close"]
            df.loc[i, "stock2 cash"] =  currEtfNtl * df.loc[i, "Close_1"]
            
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

def go(ticker1, ticker2, startDate, endDate, leverage):
    
    stock1 = yf.download(ticker1, start=startDate, end=endDate)
    stock2 = yf.download(ticker2, start=startDate, end=endDate)
    
    stock1[ticker1+"_dailyReturns"] = stock1["Close"].pct_change().dropna()
    stock2[ticker2+"_dailyReturns"] = stock2["Close"].pct_change().dropna()

    #stationary test
    print(f"Test for stationary {ticker1}")
    stationaryTest(stock1[ticker1+"_dailyReturns"].dropna())
    print(f"Test for stationary {ticker2}")
    stationaryTest(stock2[ticker2+"_dailyReturns"].dropna())
    print(f"cointegration Test for stationary {ticker1} vs {ticker2}")
    cointegrationTest(stock1[ticker1+"_dailyReturns"].dropna(), stock2[ticker2+"_dailyReturns"].dropna())

    combined_df = pd.concat([stock1, stock2], axis=1)
    combined_df["spread"] = combined_df[ticker1+"_dailyReturns"] - combined_df[ticker2+"_dailyReturns"]
    combined_df["abs_sprd"] = abs(combined_df["spread"])
    combined_df[ticker1+"_rolling avg"] = combined_df[ticker1+"_dailyReturns"].rolling(window=30).std()
    combined_df[ticker2+"_rolling avg"] = combined_df[ticker2+"_dailyReturns"].rolling(window=30).std()
    
    plt.figure(figsize=(12,6))
    plt.plot(combined_df[ticker1+"_dailyReturns"], label=ticker1+" dailyReturns")
    plt.plot(combined_df[ticker2+"_dailyReturns"], label=ticker2+" dailyReturns")
    plt.plot(combined_df[ticker1+"_rolling avg"], label=ticker1+" rolling")
    plt.plot(combined_df[ticker2+"_rolling avg"], label=ticker2+" rolling")
    plt.plot(combined_df["abs_sprd"], label="sprd", linestyle="--")
    plt.legend()
    plt.grid(True)
    plt.show()
    
    df = signal(combined_df,ticker1,ticker2)
    df = df_column_uniquify(df)
    df = pnl(df, leverage,ticker1,ticker2)
    
    print()
    
    print("{} pnl = {}".format(ticker1, df['stock1 cash'].sum()))
    print("{} pnl = {}".format(ticker2, df["stock2 cash"].sum()))
    print("total pnl = {}".format(df["stock1 cash"].sum()+df["stock2 cash"].sum()))

    df.to_csv(ticker1+" VS "+ticker2+".csv", index=False)

    print("abs sprd = {}".format(df["abs_sprd"].tail(1).values))
    print("{} 30-day rolling avg = {}".format(ticker1,df[ticker1+"_rolling avg"].tail(1).values))
    print("{} 30-day rolling avg = {}".format(ticker2,df[ticker2+"_rolling avg"].tail(1).values))
    
    return None

ticker1 = "HG=F"
ticker2 = "copx"
startDate = "2024-01-01"
endDate = "2024-07-23"
leverage = 1

go(ticker1, ticker2, startDate, endDate, leverage)
