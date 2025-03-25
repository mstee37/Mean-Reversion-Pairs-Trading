import pandas as pd
import numpy as np
import calculate_stats
import warnings

warnings.filterwarnings("ignore")

def process_data(file_path):
    """
    Process the input CSV file to filter relevant columns and calculate additional metrics.
    """
    # Load the data
    data = pd.read_csv(file_path).rename(columns={"index": "date"})

    # Filter columns to only keep those with "_Close", "_dailyReturns", "_Size", or "Live Trades"
    filtered_columns = [col for col in data.columns if any(keyword in col for keyword in ["date", "_Close", "_dailyReturns", "_Size", "Live Trades"])]
    df = data[filtered_columns]
    df.dropna(inplace=True)
    df.reset_index(inplace=True, drop=True)
    # print(df)
    
    # Extract tickers
    close_columns = [col for col in df.columns if "_Close" in col]
    left_of_close = [col.split("_Close")[0] for col in close_columns]
    ticker1 = left_of_close[0]
    ticker2 = left_of_close[1]
    # print(ticker1, ticker2)
    
    # Add directional columns
    dir = 0
    for i in df.index:
        if df.loc[i, "Live Trades"] == "Opened":
            if df.loc[i, ticker1 + "_Size"] > 0:
                dir = 1
            else:
                dir = 0

            while i < len(df) and df.loc[i, "Live Trades"] != "Closed":
                if dir == 1:
                    df.loc[i, ticker1 + "_dir"] = "Long"
                    df.loc[i, ticker2 + "_dir"] = "Short"
                else:
                    df.loc[i, ticker1 + "_dir"] = "Short"
                    df.loc[i, ticker2 + "_dir"] = "Long"
                i += 1

    df.ffill(inplace=True)
    # print(df)

    # Calculate MTM and TotalMTM
    for i in df.index:
        if df.loc[i, "Live Trades"] == "Opened":
            df.loc[i, ticker1 + "_MTM"] = abs(df.loc[i, ticker1 + "_Size"] * df.loc[i, ticker1 + "_Close"])
            df.loc[i, ticker2 + "_MTM"] = abs(df.loc[i, ticker2 + "_Size"] * df.loc[i, ticker2 + "_Close"])
            i += 1
            while i < len(df) and df.loc[i, "Live Trades"] != "Opened":
                if df.loc[i, ticker1 + "_dir"] == "Long":
                    df.loc[i, ticker1 + "_MTM"] = df.loc[i - 1, ticker1 + "_MTM"] * (1 + df.loc[i, ticker1 + "_dailyReturns"])
                    df.loc[i, ticker2 + "_MTM"] = df.loc[i - 1, ticker2 + "_MTM"] * (1 - df.loc[i, ticker2 + "_dailyReturns"])
                else:
                    df.loc[i, ticker1 + "_MTM"] = df.loc[i - 1, ticker1 + "_MTM"] * (1 - df.loc[i, ticker1 + "_dailyReturns"])
                    df.loc[i, ticker2 + "_MTM"] = df.loc[i - 1, ticker2 + "_MTM"] * (1 + df.loc[i, ticker2 + "_dailyReturns"])
                i += 1
            i -= 1

    df["TotalMTM"] = df[ticker1 + "_MTM"] + df[ticker2 + "_MTM"]
    df["pct_change"] = df["TotalMTM"].pct_change()
    df["pct_change"] = np.where(df["Live Trades"] == "Opened", 0, df["pct_change"])
    df["pnl"] = np.where(df["Live Trades"] != "Opened", df["TotalMTM"] - df["TotalMTM"].shift(1), 0)

    return df, ticker1, ticker2

def get_sharpe(file_path):

    # Process the data
    df, ticker1, ticker2 = process_data(file_path)
    # print(df)

    df_index = df[["date", "pct_change"]]
    
    # Calculate the Sharpe ratio
    sharpe_ratio = calculate_stats.get_sharpe_ratio(df_index)
    
    df = df_index.copy()
    df["prod"] = (df["pct_change"]+1).cumprod()
    mdd = calculate_stats.calculate_draw_down(df, "prod")["mdd"].iloc[-1]
    
    return sharpe_ratio, mdd

if __name__ == "__main__":
    # File path to the input CSV
    file_path = "GLD VS GC=F RollingWindow = 180.csv"
    
    # Calculate the Sharpe ratio
    sharpe_ratio, mdd = get_sharpe(file_path)

    # Print the result
    print(f"Sharpe Ratio: {sharpe_ratio}")
    print(f"MDD: {mdd}")