import yfinance as yf
import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller, coint
import matplotlib.pyplot as plt
from scipy.stats import norm
    
def signal(df,ticker1,ticker2,rollingWindow):
    
    df.reset_index(inplace=True)
    
    df["isSprdPos"] = np.where(df["spread"] >= 0, 1, 0)
    df["signal"] = 0 
    
    # based on rolling window; to be accounted in param
    i = rollingWindow
    
    while i < len(df.index):
        
        absSprd = df.loc[i, "abs_sprd"]
        stock1_rolling_avg = df.loc[i, ticker1+"_rolling avg"]
        stock2_rolling_avg = df.loc[i, ticker2+"_rolling avg"]
        
        stock1_dailyReturns = df.loc[i, ticker1+"_dailyReturns"]
        stock2_dailyReturns = df.loc[i, ticker2+"_dailyReturns"]
        isNotZero = stock1_dailyReturns and stock2_dailyReturns
        
        if absSprd > stock1_rolling_avg and absSprd > stock2_rolling_avg and isNotZero:
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

def pnl(df, leverage, ticker1, ticker2, rollingWindow):
    
    df["leverage"] = leverage
    df["ratio"] = abs(df[ticker2+"_Close"] * df[ticker2+"_dailyReturns"]) / abs(df[ticker1+"_Close"] * df[ticker1+"_dailyReturns"])
    
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
                    
            df.loc[i, ticker1Cash] = df.loc[i, ticker1Ntl] * df.loc[i, ticker1+"_Close"] * -1
            df.loc[i, ticker2Cash] = df.loc[i, ticker2Ntl] * df.loc[i, ticker2+"_Close"] * -1
                    
            currStock1Ntl = df.loc[i, ticker1Ntl]
            currStock2Ntl = df.loc[i, ticker2Ntl]
            currStock1Cash, currStock2Cash = df.loc[i, ticker1Cash], df.loc[i, ticker2Cash]

        elif df.loc[i, "axe"] == -1: # Closed
            df.loc[i, ticker1Cash] = currStock1Ntl * df.loc[i, ticker1+"_Close"]
            df.loc[i, ticker2Cash] =  currStock2Ntl * df.loc[i, ticker2+"_Close"]
            
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
            
            df.loc[i, ticker1Cash] = currStock1Ntl * df.loc[i, ticker1+"_Close"]
            df.loc[i, ticker2Cash] = currStock2Ntl * df.loc[i, ticker2+"_Close"]
            
            df.loc[i, ticker1Ntl] = -currStock1Ntl
            df.loc[i, ticker2Ntl] = -currStock2Ntl
            
            # MTM calculation
            df.loc[i,ticker1MTM] = df.loc[i, ticker1Cash] + currStock1Cash
            df.loc[i,ticker2MTM] = df.loc[i, ticker2Cash] + currStock2Cash
        
        i += 1
        
    return df

def stationaryTest(val):

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
    
    print()
        
def cointegrationTest(val1, val2):
    
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
        
    print()

def plot_pnl_distribution(df):
    """
    Plots the histogram and normal PDF of the specified column in the dataframe
    and labels the mean and standard deviation lines.

    Parameters:
    df (pandas.DataFrame): The dataframe containing the data ie df["Total PnL"]
    """
    # Calculate mean and standard deviation
    mean = df.mean()
    std_dev = df.std()
    n = (df != 0).sum()

    # Calculate histogram bins and values
    hist, bins = np.histogram(df, bins='auto', density=True)

    # Generate x-values for the normal PDF
    x = np.linspace(df.min(), df.max(), 100)

    plt.figure(figsize=(12,8))
    
    # Plot the histogram
    plt.hist(df, bins=bins, density=True, alpha=0.6, label='Histogram')

    # Plot the normal PDF
    plt.plot(x, norm.pdf(x, mean, std_dev), 'r-', lw=2, label='Normal PDF')

    # Plot vertical lines for mean and standard deviation
    plt.axvline(x=mean, color='b', linestyle='dashed', label='Mean = '+str(round(mean,2)))
    plt.axvline(x=mean + std_dev, color='b', linestyle='dashed', label='Mean + Std = '+str(round(mean+std_dev,2)))
    plt.axvline(x=mean - std_dev, color='b', linestyle='dashed', label='Mean - Std = '+str(round(mean-std_dev,2)))

    # Calculate probability P(x > 0)
    prob = 1 - norm.cdf(0, mean, std_dev)
    print(f'Probability P(x > 0): {prob:.4f}')
    
    plt.title("Trades executed = {}".format(n))
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_timeSeries_sprds(combined_df, ticker1, ticker2, rollingWindow):
    
    plt.figure(figsize=(12,8))
    plt.plot(combined_df[ticker1+"_dailyReturns"], label=ticker1+" dailyReturns")
    plt.plot(combined_df[ticker2+"_dailyReturns"], label=ticker2+" dailyReturns")
    plt.plot(combined_df[ticker1+"_rolling avg"], label=ticker1+" rollingWindow = "+str(rollingWindow))
    plt.plot(combined_df[ticker2+"_rolling avg"], label=ticker2+" rollingWindow = "+str(rollingWindow))
    plt.plot(combined_df["abs_sprd"], label="abs sprd", linestyle="--")
    plt.legend()
    plt.grid(True)
    plt.show()

def arrange_cols(df, ticker):
    
    df = df[["Close"]]
    df = df.rename(columns={"Close": ticker+"_Close"})
    
    return df
    
def calculation1(combined_df, ticker1, ticker2):
    
    combined_df["spread"] = combined_df[ticker1+"_dailyReturns"] - combined_df[ticker2+"_dailyReturns"]
    combined_df["abs_sprd"] = abs(combined_df["spread"])
    combined_df[ticker1+"_rolling avg"] = combined_df[ticker1+"_dailyReturns"].rolling(window=rollingWindow).std()
    combined_df[ticker2+"_rolling avg"] = combined_df[ticker2+"_dailyReturns"].rolling(window=rollingWindow).std()
    
    return combined_df

def riskAdjusted_Pnl(df):

    i = 0
    df["Adjusted PnL"] = 0.0
    
    while i < len(df.index):
        
        if df.loc[i, "Live Trades"] == "Opened":

            isStoppedFlag = False
            innerIdx = i
            
            while innerIdx < len(df.index) and df.loc[innerIdx, "Live Trades"] != "Closed":

                if df.loc[innerIdx, "isStoppedOut"] == 1 and not isStoppedFlag:
                    df.loc[innerIdx, "Adjusted PnL"] = df.loc[innerIdx, "totalMTM"]
                    isStoppedFlag = True
                
                innerIdx += 1
                 
            if not isStoppedFlag and innerIdx < len(df.index) and df.loc[innerIdx, "Live Trades"] == "Closed":
                df.loc[innerIdx, "Adjusted PnL"] = df.loc[innerIdx, "totalMTM"]
            
            i = innerIdx
        else:
            i += 1
        
    return df

def riskStrategy(df, ticker1, ticker2):
    
    df["riskThreshold"] = np.maximum(df[ticker1+"_rolling avg"], df[ticker2+"_rolling avg"])
    df["grossCashflow"] = np.absolute(df[ticker1+"_Cashflow"]) + np.absolute(df[ticker2+"_Cashflow"])
    df["totalMTM"] = df[ticker1+"_MTM"] + df[ticker2+"_MTM"]
    df["totalMTM in %"] = np.where(df["grossCashflow"]>0, df["totalMTM"]/df["grossCashflow"],0)
    df["isStoppedOut"] = np.where((df["totalMTM in %"] < 0) & (np.absolute(df["totalMTM in %"])>df["riskThreshold"]),1,0)
    
    df = riskAdjusted_Pnl(df)
    return df

def pnl2(df, ticker1, ticker2):
    
    # func to label opened/live/closed trades
    df["Live Trades"] = ""
    df["Live Trades"] = np.where(df["axe"] == 1, "Opened", df["Live Trades"])
    df["Live Trades"] = np.where((df["signal"] == 1) & (df["axe"] == 0), "Live", df["Live Trades"])
    df["Live Trades"] = np.where(df["axe"] == -1, "Closed", df["Live Trades"])
    
    # # to label ticker 1 and 2 PnL
    ticker1Cash = ticker1+"_Cashflow"
    ticker2Cash = ticker2+"_Cashflow"
    
    df[ticker1+"_PnL"] = 0
    df[ticker2+"_PnL"] = 0
    df[ticker1+"_PnL"] = np.where((df["Live Trades"] == "Opened") | (df["Live Trades"] == "Closed"), df[ticker1Cash], 0)
    df[ticker2+"_PnL"] = np.where((df["Live Trades"] == "Opened") | (df["Live Trades"] == "Closed"), df[ticker2Cash], 0)
    
    df["Total PnL"] = 0
    df["Total PnL"] = np.where((df["Live Trades"] == "Closed"),df[ticker1+"_MTM"] + df[ticker2+"_MTM"],0)    

    return df

def printStats(df, ticker1, ticker2, rollingWindow):
    
    # print("{} pnl = {:.2f}".format(ticker1, df[ticker1+"_PnL"].sum()))
    # print("{} pnl = {:.2f}".format(ticker2, df[ticker2+"_PnL"].sum()))
    print("Total PnL = {:.2f}".format(df["Total PnL"].sum()))
    print("Risk Adjusted PnL = {:.2f}".format(df["Adjusted PnL"].sum()))

    print("abs sprd = {:.5f}".format(df["abs_sprd"].tail(1).values[0]))
    print("{} {}-day rolling avg = {:.5f}".format(ticker1,rollingWindow,df[ticker1+"_rolling avg"].tail(1).values[0]))
    print("{} {}-day rolling avg = {:.5f}".format(ticker2,rollingWindow,df[ticker2+"_rolling avg"].tail(1).values[0]))
    
    win = ((df[ticker1+"_MTM"] + df[ticker2+"_MTM"] > 0) & (df["Live Trades"] == "Closed")).sum()
    totalTrades = (df["Live Trades"] == "Closed").sum()
    print("total trades = {}".format(totalTrades))
    print("winning trades = {}".format(win))
    print("win % = {:.2f}".format(win/totalTrades))
    print("Avg PnL per Trade = {:.2f}".format(df["Total PnL"].mean()))
    print("Std PnL per Trade = {:.2f}".format(df["Total PnL"].std()))
    
    print("Risk Adjusted Avg PnL per Trade = {:.2f}".format(df["Adjusted PnL"].mean()))
    print("Risk Adjusted Std PnL per Trade = {:.2f}".format(df["Adjusted PnL"].std()))
    
    df["Gross Capital"] = 0
    df["Gross Capital"] = np.where(df["Live Trades"] == "Opened",abs(df[ticker1+"_PnL"]) + abs(df[ticker2+"_PnL"]),0)
    GrossCapital = df["Gross Capital"].mean()
    print("Avg Capital per Trade = {:.2f}".format(GrossCapital))
    print("Max Capital to execute one trade = {:.2f}".format(max(df["Gross Capital"])))    

    return df

def backtest_PairsStrat(ticker1, ticker2, startDate, endDate, leverage, rollingWindow):
    
    stock1 = yf.download(ticker1, start=startDate, end=endDate)
    stock2 = yf.download(ticker2, start=startDate, end=endDate)
    
    stock1 = arrange_cols(stock1, ticker1)
    stock2 = arrange_cols(stock2, ticker2)
    
    stock1[ticker1+"_dailyReturns"] = stock1[ticker1+"_Close"].pct_change().dropna()
    stock2[ticker2+"_dailyReturns"] = stock2[ticker2+"_Close"].pct_change().dropna()
    
    #stationary test
    print(f"Test for stationary {ticker1}")
    stationaryTest(stock1[ticker1+"_dailyReturns"].dropna())
    print(f"Test for stationary {ticker2}")
    stationaryTest(stock2[ticker2+"_dailyReturns"].dropna())

    combined_df = pd.concat([stock1, stock2], axis=1)
    combined_df = combined_df.dropna()
    
    #cointegration test
    print(f"Cointegration Test for stationary {ticker1} vs {ticker2}")
    cointegrationTest(combined_df[ticker1+"_dailyReturns"].dropna(), combined_df[ticker2+"_dailyReturns"].dropna())
 
    combined_df = calculation1(combined_df, ticker1, ticker2)
    
    # plot_timeSeries_sprds(combined_df,ticker1,ticker2,rollingWindow)
    
    df = signal(combined_df,ticker1,ticker2,rollingWindow)
    df = pnl(df, leverage,ticker1,ticker2,rollingWindow)
    df = pnl2(df, ticker1, ticker2)
    df = riskStrategy(df, ticker1, ticker2)
    printStats(df, ticker1, ticker2, rollingWindow)
    plot_pnl_distribution(df["Total PnL"])
    plot_pnl_distribution(df["Adjusted PnL"])
    
    df.to_csv(ticker1+" VS "+ticker2+" RollingWindow = "+str(rollingWindow)+".csv", index=False)
    
    return None

ticker1 = "2330.TW" # HG=F (copper futures), 2330.TW, NVDA, GLD
ticker2 = "2454.TW" # copx (copper ETF), 2454.TW, AMD, GC=F
startDate = "2019-01-01"
endDate = "2024-09-02"
leverage = 5
rollingWindow = 5

backtest_PairsStrat(ticker1, ticker2, startDate, endDate, leverage, rollingWindow)
