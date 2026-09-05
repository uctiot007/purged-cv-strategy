import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller


def get_weights(d, thresh=1e-5, max_size=10000):
    weights = [1.0]
    k = 1
    while True:
        w_k = -weights[-1] * (d - k + 1) / k
        weights.append(w_k)
        if abs(w_k) < thresh or k >= max_size:
            break
        k += 1
    return np.array(weights)


def frac_diff(series, d, thresh=1e-5):
    weights = get_weights(d, thresh)
    width = len(weights)
    diff_values = np.full(len(series), np.nan)
    
    for i in range(width - 1, len(series)):
        window = series.iloc[i - width + 1:i + 1].values[::-1]
        diff_values[i] = np.dot(weights, window)
    
    return pd.Series(diff_values, index=series.index)


def find_min_d(series, d_grid=None, thresh=1e-5, p_thresh=0.05):
    if d_grid is None:
        d_grid = np.linspace(0, 1, 11)
    
    for d in d_grid:
        diffed = frac_diff(series, d, thresh).dropna()
        if len(diffed) < 10:
            continue
        p_val = adfuller(diffed)[1]
        if p_val < p_thresh:
            return d
    return 1.0


if __name__ == "__main__":
    s = pd.Series(np.arange(1.0, 101.0))
    result = frac_diff(s, d=0.5, thresh=1e-2)
    print(result.tail(10))
    print("\nLength:", len(result))
    print("NaN count:", result.isna().sum())
    print("Non-NaN count:", result.notna().sum())
    
    # Test find_min_d on a random walk
    np.random.seed(42)
    rw = pd.Series(np.cumsum(np.random.randn(1000)))
    min_d = find_min_d(rw)
    print(f"\nMinimum d for stationarity: {min_d:.2f}")