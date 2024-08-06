import yfinance as yf
import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller, coint
import matplotlib.pyplot as plt
    
def signal(df,ticker1,ticker2,rollingWindow):
    
    df.reset_index(inplace=True)
    
    df["isSprdPos"] = np.where(df["spread"] >= 0, 1, 0)
    df["signal"] = 0 
    
    # based on rolling window; to be accounted in para
    i = rollingWindow
    
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

def pnl(df, leverage, ticker1, ticker2, rollingWindow):
    df["leverage"] = leverage
    df["ratio"] = abs(df["Close_1"] * df[ticker2+"_dailyReturns"]) / abs(df["Close"] * df[ticker1+"_dailyReturns"])
    
    # naming
    
    # Initialize new columns
    
    ticker1Ntl = ticker1+"_Size"
    ticker2Ntl = ticker2+"_Size"
    ticker1Cash = ticker1+"_Cashflow"
    ticker2Cash = ticker2+"_Cashflow"
    
    df[ticker1Ntl] = 0.0
    df[ticker2Ntl] = 0.0
    df[ticker1Cash] = 0.0
    df[ticker2Cash] = 0.0
    
    ticker1MTM = ticker1+"_MTM"
    ticker2MTM = ticker2+"_MTM"
    df[ticker1MTM] = 0.0
    df[ticker2MTM] = 0.0
    
    i = rollingWindow # to update as rolling window number
    
    currStock1Ntl, currStock2Ntl = 0, 0
    currStock1Cash, currStock2Cash = 0, 0
    
    # naming convention
    dailyReturn1 = ticker1+"_dailyReturns"
    dailyReturn2 = ticker2+"_dailyReturns"
    
    while i < len(df.index):
        
        if df.loc[i, "axe"] == 1:
            
            if abs(df.loc[i, dailyReturn1]) > abs(df.loc[i, dailyReturn2]):
                if df.loc[i, dailyReturn1] > 0:
                    df.loc[i, ticker1Ntl] = np.floor(df.loc[i, "ratio"] * -1 * df.loc[i, "leverage"])
                    df.loc[i, ticker2Ntl] = df.loc[i, "leverage"]
                else:
                    df.loc[i, ticker1Ntl] = np.ceil(df.loc[i, "ratio"] * df.loc[i, "leverage"])
                    df.loc[i, ticker2Ntl] = -1 * df.loc[i, "leverage"]
            else:
                if df.loc[i, dailyReturn2] > 0:
                    df.loc[i, ticker1Ntl] = np.ceil(df.loc[i, "ratio"] * df.loc[i, "leverage"])
                    df.loc[i, ticker2Ntl] = -1 * df.loc[i, "leverage"]
                else:
                    df.loc[i, ticker1Ntl] = np.floor(df.loc[i, "ratio"] * -1 * df.loc[i, "leverage"])
                    df.loc[i, ticker2Ntl] = df.loc[i, "leverage"]
                    
            df.loc[i, ticker1Cash] = df.loc[i, ticker1Ntl] * df.loc[i, "Close"] * -1
            df.loc[i, ticker2Cash] = df.loc[i, ticker2Ntl] * df.loc[i, "Close_1"] * -1
                    
            currStock1Ntl = df.loc[i, ticker1Ntl]
            currStock2Ntl = df.loc[i, ticker2Ntl]
            currStock1Cash, currStock2Cash = df.loc[i, ticker1Cash], df.loc[i, ticker2Cash]

        elif df.loc[i, "axe"] == -1: # Closed
            df.loc[i, ticker1Cash] = currStock1Ntl * df.loc[i, "Close"]
            df.loc[i, ticker2Cash] =  currStock2Ntl * df.loc[i, "Close_1"]
            
            df.loc[i, ticker1Ntl] = -currStock1Ntl
            df.loc[i, ticker2Ntl] = -currStock2Ntl
            
            # MTM calculation
            df.loc[i,ticker1MTM] = df.loc[i, ticker1Cash] + currStock1Cash
            df.loc[i,ticker2MTM] = df.loc[i, ticker2Cash] + currStock2Cash
            
            # reset to 0 after closing trade
            currStock1Ntl, currStock2Ntl = 0, 0
            currStock1Cash, currStock2Cash = 0, 0

        elif df.loc[i, "signal"] == 1: # live trade
            
            # calculate MTM exposure for opened positions
            
            df.loc[i, ticker1Cash] = currStock1Ntl * df.loc[i, "Close"]
            df.loc[i, ticker2Cash] = currStock2Ntl * df.loc[i, "Close_1"]
            
            df.loc[i, ticker1Ntl] = -currStock1Ntl
            df.loc[i, ticker2Ntl] = -currStock2Ntl
            
            # MTM calculation
            df.loc[i,ticker1MTM] = df.loc[i, ticker1Cash] + currStock1Cash
            df.loc[i,ticker2MTM] = df.loc[i, ticker2Cash] + currStock2Cash
        
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

def backtest_PairsStrat(ticker1, ticker2, startDate, endDate, leverage, rollingWindow):
    
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
    combined_df[ticker1+"_rolling avg"] = combined_df[ticker1+"_dailyReturns"].rolling(window=rollingWindow).std()
    combined_df[ticker2+"_rolling avg"] = combined_df[ticker2+"_dailyReturns"].rolling(window=rollingWindow).std()
    
    plt.figure(figsize=(12,6))
    plt.plot(combined_df[ticker1+"_dailyReturns"], label=ticker1+" dailyReturns")
    plt.plot(combined_df[ticker2+"_dailyReturns"], label=ticker2+" dailyReturns")
    plt.plot(combined_df[ticker1+"_rolling avg"], label=ticker1+" rollingWindow = "+str(rollingWindow))
    plt.plot(combined_df[ticker2+"_rolling avg"], label=ticker2+" rollingWindow = "+str(rollingWindow))
    plt.plot(combined_df["abs_sprd"], label="abs sprd", linestyle="--")
    plt.legend()
    plt.grid(True)
    plt.show()
    
    df = signal(combined_df,ticker1,ticker2,rollingWindow)
    df = df_column_uniquify(df)
    df = pnl(df, leverage,ticker1,ticker2,rollingWindow)

    # func to label opened/live/closed trades
    df["Live Trades"] = ""
    df["Live Trades"] = np.where(df["axe"] == 1, "Opened", df["Live Trades"])
    df["Live Trades"] = np.where((df["signal"] == 1) & (df["axe"] == 0), "Live", df["Live Trades"])
    df["Live Trades"] = np.where(df["axe"] == -1, "Closed", df["Live Trades"])
    
    ticker1Cash = ticker1+"_Cashflow"
    ticker2Cash = ticker2+"_Cashflow"
    
    df[ticker1+"_PnL"] = 0
    df[ticker2+"_PnL"] = 0
    df[ticker1+"_PnL"] = np.where((df["Live Trades"] == "Opened") | (df["Live Trades"] == "Closed"), df[ticker1Cash], 0)
    df[ticker2+"_PnL"] = np.where((df["Live Trades"] == "Opened") | (df["Live Trades"] == "Closed"), df[ticker2Cash], 0)
    
    df["Total PnL"] = 0
    df["Total PnL"] = np.where((df["Live Trades"] == "Closed"),df[ticker1+"_MTM"] + df[ticker2+"_MTM"],0)
    
    print()
    
    print("{} pnl = {:.2f}".format(ticker1, df[ticker1+"_PnL"].sum()))
    print("{} pnl = {:.2f}".format(ticker2, df[ticker2+"_PnL"].sum()))
    print("total pnl = {:.2f}".format(df[ticker1+"_PnL"].sum()+df[ticker2+"_PnL"].sum()))

    df.to_csv(ticker1+" VS "+ticker2+" RollingWindow = "+str(rollingWindow)+".csv", index=False)

    print("abs sprd = {}".format(df["abs_sprd"].tail(1).values))
    print("{} {}-day rolling avg = {}".format(ticker1,rollingWindow,df[ticker1+"_rolling avg"].tail(1).values))
    print("{} {}-day rolling avg = {}".format(ticker2,rollingWindow,df[ticker2+"_rolling avg"].tail(1).values))
    
    win = ((df[ticker1+"_MTM"] + df[ticker2+"_MTM"] > 0) & (df["Live Trades"] == "Closed")).sum()
    totalTrades = (df["Live Trades"] == "Closed").sum()
    print("total trades = {}".format(totalTrades))
    print("winning trades = {}".format(win))
    print("win % = {:.2f}".format(win/totalTrades))
    print("Avg PnL = {:.2f}".format(df["Total PnL"].mean()))
    print("Std PnL = {:.2f}".format(df["Total PnL"].std()))
    
    return None

# parameters input 
# ticker 1 -> lower spot price
# ticker 2 -> higher spot price

ticker1 = "2330.TW" # HG=F (copper futures)
ticker2 = "2454.TW" # copx (copper ETF)
startDate = "2024-01-01"
endDate = "2024-08-05"
leverage = 5
rollingWindow = 5 # to add in as variable

backtest_PairsStrat(ticker1, ticker2, startDate, endDate, leverage, rollingWindow)
