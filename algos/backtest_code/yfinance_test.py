import yfinance as yf

yf.download("7011.T", start="2025-12-01", end="2025-12-31", interval="1d", ignore_tz=True).to_csv("7011.T.csv")