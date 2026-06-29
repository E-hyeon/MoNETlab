import pandas as pd
import numpy as np

df = pd.read_csv("kpi_live.csv")
df["timestamp"] = pd.to_datetime(df["timestamp"])
df = df.sort_values("timestamp")
df["nprb"] = pd.to_numeric(df["nprb"], errors="coerce").fillna(0)

def jains(x):
    x = x[x > 0]
    n = len(x)
    return (x.sum()**2) / (n * (x**2).sum()) if n > 0 and (x**2).sum() > 0 else float('nan')

result = df.groupby(pd.Grouper(key="timestamp", freq="10s")).apply(
    lambda g: pd.Series({
        "fairness":       jains(g.groupby("rnti")["nprb"].sum()),
        "total_nprb":     g["nprb"].sum(),
        "ue1_nprb":       g[g["rnti"] == g["rnti"].unique()[0]]["nprb"].sum() if g["rnti"].nunique() >= 1 else 0,
        "ue2_nprb":       g[g["rnti"] == g["rnti"].unique()[-1]]["nprb"].sum() if g["rnti"].nunique() >= 2 else 0,
        "ue_count":       g["rnti"].nunique(),
    })
)

r = result.dropna()
print(r.tail(20))
print(f"\n평균 Jain's Fairness : {r['fairness'].mean():.4f}")
print(f"평균 UE1 nPRB       : {r['ue1_nprb'].mean():.1f}")
print(f"평균 UE2 nPRB       : {r['ue2_nprb'].mean():.1f}")
