# train_ml.py
import pandas as pd, numpy as np
from pathlib import Path
from ta.momentum import RSIIndicator
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_absolute_error, roc_auc_score
import joblib

data_dir = Path("data")
models_dir = Path("models")
models_dir.mkdir(exist_ok=True)
MIN_ROWS_PER_TICKER = 200   # filter tiny tickers after features
TEST_DAYS = 180             # primary time-based test window
FALLBACK_RATIO = 0.8        # 80/20 fallback if time split yields empty set
MAX_TICKERS_KEEP = 120      # max tickers to keep after feature engineering
def make_features(df: pd.DataFrame) -> pd.DataFrame:
    # Ensure types and order
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values(["Ticker", "Date"]).copy()

    # Basic returns and labels
    df["ret"] = df.groupby("Ticker")["Close"].pct_change(fill_method=None)
    df["y_reg"] = df.groupby("Ticker")["Close"].shift(-1) / df["Close"] - 1  # next-day return
    df["y_cls"] = (df["y_reg"] > 0).astype(int)                               # up/down label
# RSI via transform (no GroupBy.apply deprecation warning)
    df["rsi_14"] = df.groupby("Ticker")["Close"].transform(
    lambda s: RSIIndicator(s, window=14).rsi()
    )

# lags + rolling features
    for k in [1, 2, 3, 5, 10]:
        df[f"ret_lag{k}"] = df.groupby("Ticker")["ret"].shift(k)
        df[f"roll_mean_{k}"] = (
         df.groupby("Ticker")["Close"].rolling(k).mean().reset_index(0, drop=True)
    )
        df[f"roll_std_{k}"] = (
         df.groupby("Ticker")["ret"].rolling(k).std().reset_index(0, drop=True)
    )

# clean up rows with NaNs
    df = df.dropna().reset_index(drop=True)
    
    sizes =  df.groupby("Ticker").size().sort_values(ascending=False)
    good = sizes[sizes >= MIN_ROWS_PER_TICKER].index
    if len(good) == 0:
    # fallback: keep the top tickers by available rows
     keep = sizes.index[:min(MAX_TICKERS_KEEP, len(sizes))]
     print(f"⚠️ No tickers ≥ {MIN_ROWS_PER_TICKER} rows after features. "
          f"Keeping top {len(keep)} tickers by data size instead: {list(keep)}")
     df = df[df["Ticker"].isin(keep)].reset_index(drop=True)
    else:
     df = df[df["Ticker"].isin(good)].reset_index(drop=True)
     

    if df.empty:
     raise ValueError(
        "Still no rows after adaptive filter. "
        "Increase history (earlier start), check ticker symbols, or reduce feature windows."
    )
    return df

    

if __name__ == "__main__":
    # 1) load the price data produced by download_data.py
    prices_path = data_dir / "prices.csv"
    if not prices_path.exists():
        raise FileNotFoundError(
            "data/prices.csv not found. Run `python download_data.py` first."
        )
    df = pd.read_csv(prices_path)

    # 2) build features/labels
    df = make_features(df)
    # quick diagnostics
    print("After features:", len(df), "rows")
    print("Date range:", df["Date"].min(), "→", df["Date"].max())
    print("Rows per ticker (post-features):")
    print(df.groupby("Ticker").size().sort_values())

    # 3) simple time-based split: last 180 days = test
    # --- ROBUST TRAIN/TEST SPLIT WITH FALLBACK ---

    df = df.sort_values(["Date", "Ticker"]).reset_index(drop=True)

# Primary split: use last 180 days as test
    last_date = pd.to_datetime(df["Date"]).max()
    split_date = last_date - pd.Timedelta(days=TEST_DAYS)
    train = df[pd.to_datetime(df["Date"]) <= split_date].copy()
    test  = df[pd.to_datetime(df["Date"])  > split_date].copy()

# Fallback: if time-based split gives 0 rows (not enough history), use 80/20 ratio
    if len(train) == 0 or len(test) == 0:
     n = len(df)
     split_idx = max(1, int(n * FALLBACK_RATIO))
     train = df.iloc[:split_idx].copy()
     test  = df.iloc[split_idx:].copy()
     print("⚠️ Time-based split failed (not enough history) — using 80/20 split instead")

# Debug printout (helps confirm)
    print(f"Train rows: {len(train)}, Test rows: {len(test)}")
    if len(train) == 0 or len(test) == 0:
     raise ValueError("Still empty after fallback. Increase history or reduce min_rows.")


    # 4) choose feature columns (exclude non-features and labels)
    drop_cols = {
        "Date","Ticker","Open","High","Low","Close","Adj Close","Volume",
        "ret","y_reg","y_cls"
    }
    feature_cols = [c for c in df.columns if c not in drop_cols]
    if not feature_cols:
        raise ValueError("No feature columns found. Check your feature engineering.")

    # 5) train models
    # Regression: predict next-day return value
    reg = RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1)
    reg.fit(train[feature_cols], train["y_reg"])
    pred_reg = reg.predict(test[feature_cols])
    print("MAE (next-day return):", mean_absolute_error(test["y_reg"], pred_reg))

    # Classification: predict up/down probability
    clf = LogisticRegression(max_iter=2000)
    clf.fit(train[feature_cols], train["y_cls"])
    prob_up = clf.predict_proba(test[feature_cols])[:, 1]
    print("ROC-AUC (up/down):", roc_auc_score(test["y_cls"], prob_up))

    # 6) save models and test-time predictions for the dashboard
    joblib.dump({"model": reg, "features": feature_cols}, models_dir / "rf_reg.joblib")
    joblib.dump({"model": clf, "features": feature_cols}, models_dir / "logit_cls.joblib")

    out = test[["Date", "Ticker"]].copy()
    out["pred_return"] = pred_reg
    out["prob_up"] = prob_up
    out.to_csv(models_dir / "predictions.csv", index=False)
    print("✅ Saved models and models/predictions.csv")
