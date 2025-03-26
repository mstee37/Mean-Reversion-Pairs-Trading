import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import io
import requests
from pyrate_limiter import Duration, Limiter, RequestRate
from requests_ratelimiter import LimiterSession

rate_per_second = RequestRate(1, Duration.SECOND)
rate_per_minute = RequestRate(60, Duration.MINUTE)
limiter = Limiter(rate_per_second, rate_per_minute)
rate_limiter = LimiterSession(limiter=limiter)

FED_API = "https://fred.stlouisfed.org"


def get_risk_free_rate(start_date, end_date):
    # US 10Y yield
    res = requests.get(f"{FED_API}/graph/fredgraph.csv?id=DGS10&cosd=2015-01-01&coed={end_date}&fq=Daily")
    df_risk_free = pd.read_csv(io.StringIO(res.content.decode("UTF-8")))
    df_risk_free.columns = ["date", "dgs10"]
    df_risk_free["date"] = pd.to_datetime(df_risk_free["date"])
    df_risk_free = df_risk_free.set_index("date")
    df_risk_free["dgs10"] = df_risk_free["dgs10"].replace({".": np.nan}).astype("float")
    df_risk_free = df_risk_free.resample("D").last().ffill()
    # print(df_risk_free)
    # df_risk_free = df_risk_free[(df_risk_free.index >= start_date) & (df_risk_free.index <= end_date)]

    return df_risk_free

def get_sharpe_ratio(df_index=None, start_date=None, end_date=None, col=None, portfolio_ids=[], freq="YE", full_period=False):
    """
    Generate annualized sharpe ratio for indices. Input may be given as df_index or portfolio_ids.

    :param df_index: DataFrame with date index and columns of asset prices
    :param portfolio_ids: list of portfolio_ids
    :param freq: frequency of the sharpe ratios returned, either ME (month end) or YE (year end)
    :param full_period: if True, this precedes freq and returns the sharpe ratio over the full period
    """
    # if (not full_period) and (freq not in ["ME", "YE"]):
    #     raise ValueError("freq must be ME or YE.")

    df_index = df_index.loc[start_date:end_date]
    df_index.set_index("date",inplace=True, drop=True)
    df_index.index = pd.to_datetime(df_index.index)
    # print(df_index.head(50))
    df_risk_free = get_risk_free_rate(start_date, end_date)
    # print(df_risk_free)

    df_risk_free.index = pd.to_datetime(df_risk_free.index)
    df_merge = pd.merge(df_index, df_risk_free, left_index=True, right_index=True, how="left")
    df_merge = df_merge.ffill().bfill()
    # print(df_merge.isna().sum())
    # print(df_merge)

    df_merge["dgs10"] = df_merge["dgs10"]/252/100
    returns = (1+df_merge["pct_change"]-df_merge["dgs10"]).prod()-1

    # print(returns)
    
    total_days = df_merge.index[-1] - df_merge.index[0]
    total_days = total_days.days
    # print(total_days)
    sd = df_merge["pct_change"].std()*np.sqrt(total_days)
    # print(sd)
    sharpe = returns / sd * np.sqrt(252/total_days)
    
    downside_risk = df_merge["pct_change"].apply(lambda x: min(x, 0)).std()*np.sqrt(total_days)
    # print(downside_risk)

    sortino = returns / downside_risk * np.sqrt(252/total_days)
    # print(sortino)
    
    return returns, sharpe, sortino

def calculate_draw_down(df, column, freq_days=60, full_period=True):
    """
    Calculates the drawdown and maximum drawdown for a financial time series.

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame containing the financial time series data.
    column : str
        Column name in df containing the time series values.
    freq_days : int, optional
        Number of days for each slice if full_period is False. Default is 60.
    full_period : bool, optional
        If True, calculates drawdown for the entire period. Default is True.

    Returns:
    --------
    pd.DataFrame
        DataFrame with columns "dd" (drawdown), "mdd" (maximum drawdown), and "avdd" (average drawdown) for the entire period
        if full_period is True, or a DataFrame with "mdd" and "avdd" for each slice if full_period is False.
    """

    def get_slices(df, n):
        slices = []
        for start in range(0, len(df), n):
            end = start + n
            slices.append(df.iloc[start:end])
        return slices

    def get_full_period_dd(df, column):
        df_result = pd.DataFrame(columns=["dd", "mdd"])
        peak = df.iloc[0][column]
        mdd = 0
        for i in df.index:
            peak = max(peak, df.loc[i, column])
            mdd = max(mdd, (peak - df.loc[i, column]) / peak)
            df_result.loc[i] = [(peak - df.loc[i, column]) / peak, mdd]
        df_result["avdd"] = df_result["dd"].expanding().mean()
        return df_result

    if full_period:
        return get_full_period_dd(df, column)
    else:
        slices = get_slices(df, freq_days)
        results = []
        for df_slice in slices:
            df_temp = get_full_period_dd(df_slice, column)[["mdd", "avdd"]]
            results.append(df_temp.iloc[-1])
        df_result = pd.concat(results, axis=1).transpose().reset_index(drop=True)

        return df_result


# def calculate_model_performance_stats(df, column, base=1000):
#     """
#     Calculates performance statistics for a given financial time series.

#     Parameters:
#     -----------
#     df : pd.DataFrame
#         DataFrame containing the financial time series data.
#     column : str
#         Column name in df containing the time series values.

#     Returns:
#     --------
#     dict
#         Dictionary with the raw value, Sharpe ratio, and maximum drawdown for the specified column.

#         Keys:
#         - {column} : float
#             The raw value of the time series at the last date.
#         - {column}_sharpe : float
#             The Sharpe ratio of the time series over the entire period.
#         - {column}_mdd : float
#             The maximum drawdown of the time series over the entire period.
#     """

#     raw_value = df[column].iloc[-1]
#     returns = raw_value/base-1
#     sharpe = get_sharpe_ratio(df.index[0], df.index[-1], df, column, full_period=True)
#     mdd = calculate_draw_down(df, column, 365, True)["mdd"].iloc[-1]

#     return {column: raw_value, column + "_returns":returns, column + "_sharpe": sharpe, column + "_mdd": mdd}


def main():
    # print(get_risk_free_rate().head())
    df = pd.read_csv("test.csv")
    # print(df.head())
    print(get_sharpe_ratio(df))
    # df["prod"] = (df["pct_change"]+1).cumprod()
    # # print(df)
    # print(calculate_draw_down(df, "prod")["mdd"].iloc[-1])
    
    
    pass

if __name__ == "__main__":
    main()
