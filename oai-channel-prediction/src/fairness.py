import pandas as pd
import numpy as np

df = pd.read_csv("kpi_live.csv")
df["timestamp"] = pd.to_datetime(df["timestamp"])
df = df.sort_values("timestamp")
df["dl_bytes"] = pd.to_numeric(df["dl_bytes"], errors="coerce").fillna(0)

def jains(x):
    x = x[x > 0]
    n = len(x)
    return (x.sum()**2) / (n * (x**2).sum()) if n > 0 and (x**2).sum() > 0 else float('nan')

result = df.groupby(pd.Grouper(key="timestamp", freq="10s")).apply(
    lambda g: pd.Series({
        "fairness":      jains(g.groupby("rnti")["dl_bytes"].sum()),
        "total_dl_mbps": g["dl_bytes"].sum() * 8 / 1e6 / 10,
        "ue_count":      g["rnti"].nunique(),
    }), include_groups=False
)

print(result.dropna().tail(20))
