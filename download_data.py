import yfinance as yf
import pandas as pd
from pathlib import Path

data_dir = Path("data")
data_dir.mkdir(exist_ok=True)

def read_tickers(path="tickers.txt"):
    return [t.strip() for t in open(path) if t.strip()]

def get_history(tickers, start="2010-01-01", interval="1d"):
    df = yf.download(tickers, start=start, interval=interval, group_by="ticker", auto_adjust=False)
    out = []
    for t in (tickers if isinstance(tickers, list) else [tickers]):
        if t not in df.columns.get_level_values(0):
            continue
        sub = df[t].reset_index().rename(columns=str.title)
        sub["Ticker"] = t
        out.append(sub)
    return pd.concat(out, ignore_index=True)

if __name__ == "__main__":
    tickers = read_tickers()
    data = get_history(tickers)
    data.to_csv(data_dir / "prices.csv", index=False)
    print(f"✅ Saved {len(data):,} rows to data/prices.csv")