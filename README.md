# Mean Reversion Pairs Trading


| Ticker 1 | Ticker 2 | Returns   | Sharpe Ratio | Sortino Ratio | Max Drawdown (MDD) |
|----------|----------|-----------|--------------|---------------|---------------------|
| HG=F     | COPX     | 1.485188  | 0.566666     | 0.942814      | 0.288997            |
| 2330.TW  | 2454.TW  | 2.329174  | 0.760188     | 1.247025      | 0.224490            |
| NVDA     | AMD      | -0.593534 | -0.112578    | -0.156977     | 0.610149            |
| **GLD**  | **GC=F** | **1.597792** | **0.678110** | **2.268978** | **0.029714**       |
| KO       | PEP      | 0.145311  | 0.076175     | 0.125131      | 0.143717            |


## Observations

most using 180-day rolling std as signals, 15 year TS

- **Commodity Pairs (30day for copper pair, 180day for gold pai):** Seem stable over a long period.
- **2330.TW and 2454.TW:** Performed better in the 5 year TS with a Sharpe ratio of ~1, indicating inefficiency and volatility.
- **KO and PEP:** Weak performance, likely due to low trading volume.
- **NVDA and AMD:** Negative performance, suggesting a highly efficient market where high-frequency trading has already arbitraged opportunities away.
