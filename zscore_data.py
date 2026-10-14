import requests
import pandas as pd
import numpy as np
import time
import csv
import os
import subprocess
from datetime import datetime

# ─── 配置 ────────────────────────────────────────────────
CSV_FILE = "zscore_data.csv"

# ─── 获取1分钟K线 ─────────────────────────────────────────
def get_candles(limit=400):
    url = "https://indexer.dydx.trade/v4/candles/perpetualMarkets/BTC-USD"
    params = {"resolution": "1MIN", "limit": limit}
    try:
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
    except requests.RequestException as e:
        raise RuntimeError(f"❌ API请求失败: {e}")

    df = pd.DataFrame(resp.json()["candles"])
    df["close"]     = df["close"].astype(float)
    df["startedAt"] = pd.to_datetime(df["startedAt"])
    df = df.sort_values("startedAt").reset_index(drop=True)
    return df

# ─── 计算 MA270 + STD100 + Z-Score ───────────────────────
def add_zscore(df):
    df["MA270"]  = df["close"].rolling(window=270).mean()
    df["STD100"] = df["close"].rolling(window=100).std()
    df["zscore"] = (df["close"] - df["MA270"]) / df["STD100"]
    return df

# ─── 保存到CSV ────────────────────────────────────────────
def save_to_csv(row):
    file_exists = os.path.exists(CSV_FILE)
    with open(CSV_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow([
                "time", "price", "MA270", "STD100", "zscore",
                "mean", "min", "max",
                "p1", "p5", "p25", "p50", "p75", "p95", "p99"
            ])
        writer.writerow(row)

# ─── Git Push ─────────────────────────────────────────────
def git_push():
    try:
        subprocess.run(["git", "add", CSV_FILE], check=True)
        subprocess.run(["git", "commit", "-m", f"zscore update {datetime.now().strftime('%Y-%m-%d %H:%M')}"], check=True)
        subprocess.run(["git", "push"], check=True)
        print("  ✅ 数据已推送到GitHub")
    except Exception as e:
        print(f"  ⚠️  Git推送失败: {e}")

# ─── 主循环 ───────────────────────────────────────────────
def main():
    print("🚀 Z-Score 监控 | BTC-USD | MA270 | STD100 | 1MIN")
    print(f"   数据保存至: {CSV_FILE}")
    print("─" * 50)

    while True:
        try:
            df       = get_candles(400)
            df       = add_zscore(df)
            df_valid = df.dropna(subset=["zscore"])
            latest   = df_valid.iloc[-1]
            now      = datetime.now().strftime("%H:%M:%S")

            # ── 打印 ──────────────────────────────────────
            print(f"\n[{now}]")
            print(f"  价格    : ${latest['close']:,.2f}")
            print(f"  MA270   : ${latest['MA270']:,.2f}")
            print(f"  STD100  : ${latest['STD100']:,.2f}")
            print(f"  Z-Score : {latest['zscore']:+.4f}")
            print("─" * 50)
            print("  📊 当前统计分布")
            print(f"  均值    : {df_valid['zscore'].mean():+.4f}")
            print(f"  最小值  : {df_valid['zscore'].min():+.4f}")
            print(f"  最大值  : {df_valid['zscore'].max():+.4f}")
            print("─" * 50)
            print("  📈 百分位")
            quantiles = [0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99]
            for q in quantiles:
                val = df_valid["zscore"].quantile(q)
                print(f"  {int(q*100):>3}%  →  {val:+.4f}")
            print("─" * 50)

            # ── 保存到CSV ─────────────────────────────────
            save_to_csv([
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                latest["close"],
                round(latest["MA270"], 2),
                round(latest["STD100"], 2),
                round(latest["zscore"], 4),
                round(df_valid["zscore"].mean(), 4),
                round(df_valid["zscore"].min(), 4),
                round(df_valid["zscore"].max(), 4),
                round(df_valid["zscore"].quantile(0.01), 4),
                round(df_valid["zscore"].quantile(0.05), 4),
                round(df_valid["zscore"].quantile(0.25), 4),
                round(df_valid["zscore"].quantile(0.50), 4),
                round(df_valid["zscore"].quantile(0.75), 4),
                round(df_valid["zscore"].quantile(0.95), 4),
                round(df_valid["zscore"].quantile(0.99), 4),
            ])
            print(f"  💾 数据已保存到 {CSV_FILE}")

            # ── 每10分钟push到GitHub ──────────────────────
            if datetime.now().minute % 10 == 0:
                git_push()

        except Exception as e:
            print(f"⚠️  错误: {e}")

        time.sleep(60)

if __name__ == "__main__":
    main()