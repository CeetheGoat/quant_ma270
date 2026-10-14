import requests
import pandas as pd
import time
from datetime import datetime, timedelta

# ─── 配置 ────────────────────────────────────────────────
LEVERAGE      = 10
TOTAL_CAPITAL = 1000

MA_PERIOD    = 270
ENTRY_THRESH = 0.0035
ADD_THRESH   = 0.0055
STOP_LOSS    = 0.0100

# 反向单配置
REVERSE_TRIM  = 0.0120  # 偏离1.2% → 出75%
REVERSE_EXIT  = 0.0150  # 偏离1.5% → 全平
REVERSE_BACK1 = 0.0100  # 偏离回到1.0% → 全平
REVERSE_BACK2 = 0.0070  # 偏离回到0.7% → 全平并反转

# 熔断配置
MAX_LOSS_PCT    = 0.05
MAX_LOSS_WINDOW = timedelta(hours=3)
MAX_LOSS_AMOUNT = TOTAL_CAPITAL * MAX_LOSS_PCT

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

# ─── 计算 MA270 ───────────────────────────────────────────
def add_ma(df, period=MA_PERIOD):
    df[f"MA{period}"] = df["close"].rolling(window=period).mean()
    return df

# ─── 熔断器 ───────────────────────────────────────────────
class CircuitBreaker:
    def __init__(self):
        self.trades    = []
        self.is_halted = False
        self.halted_at = None

    def record(self, pnl: float):
        self.trades.append({"time": datetime.now(), "pnl": pnl})

    def check(self) -> bool:
        if self.is_halted:
            print(f"\n🚫 交易暂停中  (熔断触发于 {self.halted_at.strftime('%H:%M:%S')})")
            print(f"   请手动调用 circuit_breaker.reset() 恢复交易")
            return False

        now      = datetime.now()
        cutoff   = now - MAX_LOSS_WINDOW
        recent   = [t for t in self.trades if t["time"] >= cutoff]
        loss_sum = sum(t["pnl"] for t in recent if t["pnl"] < 0)

        if abs(loss_sum) >= MAX_LOSS_AMOUNT:
            self.is_halted = True
            self.halted_at = now
            print(f"\n🚨 熔断触发！过去3小时亏损: ${loss_sum:,.2f} (超过 ${MAX_LOSS_AMOUNT:.2f})")
            print(f"   交易已暂停，请手动输入 circuit_breaker.reset() 恢复")
            return False

        return True

    def reset(self):
        self.trades    = []
        self.is_halted = False
        self.halted_at = None
        print("✅ 熔断已重置，恢复交易")

# ─── 主仓位 ───────────────────────────────────────────────
class Position:
    def __init__(self):
        self.size           = 0.0
        self.entries        = []
        self.is_open        = False
        self.direction      = None
        self.added          = False
        self.stop_triggered = False

    def open(self, direction, price, amount, add_amount, balance, diff):
        self.direction      = direction
        self.entries        = [{"price": price, "amount": amount}]
        self.size           = amount
        self.is_open        = True
        self.added          = False
        self.stop_triggered = False
        icon = "📥🟢" if direction == "long" else "📥🔴"
        print(f"  {icon} 开{direction.upper()}  价格: ${price:,.2f}  金额: ${amount:,.2f}  | 总资产: ${balance:,.2f}")

        # 开仓时偏离已超过加仓线，直接加仓
        if abs(diff) >= ADD_THRESH:
            self.entries.append({"price": price, "amount": add_amount})
            self.size  += add_amount
            self.added  = True
            icon2 = "➕🟢" if direction == "long" else "➕🔴"
            print(f"  {icon2} 同时加仓  价格: ${price:,.2f}  金额: ${add_amount:,.2f}  总仓: ${self.size:,.2f}  | 总资产: ${balance:,.2f}")

    def add(self, price, amount, balance):
        self.entries.append({"price": price, "amount": amount})
        self.size  += amount
        self.added  = True
        icon = "➕🟢" if self.direction == "long" else "➕🔴"
        print(f"  {icon} 加仓  价格: ${price:,.2f}  金额: ${amount:,.2f}  总仓: ${self.size:,.2f}  | 总资产: ${balance:,.2f}")

    def close(self, price, reason, cb, balance):
        if not self.is_open:
            return balance
        avg_entry = sum(e["price"] * e["amount"] for e in self.entries) / self.size
        if self.direction == "long":
            pnl_pct  = (price - avg_entry) / avg_entry * 100
            pnl_usdc = self.size * (price - avg_entry) / avg_entry
        else:
            pnl_pct  = (avg_entry - price) / avg_entry * 100
            pnl_usdc = self.size * (avg_entry - price) / avg_entry

        balance += pnl_usdc
        print(f"  📤 平仓  原因: {reason}")
        print(f"     均入场价: ${avg_entry:,.2f}  平仓价: ${price:,.2f}")
        print(f"     盈亏: {'🟢' if pnl_usdc >= 0 else '🔴'} ${pnl_usdc:+,.2f} ({pnl_pct:+.2f}%)  | 总资产: ${balance:,.2f}")

        cb.record(pnl_usdc)
        self.__init__()
        return balance

# ─── 反向仓位 ─────────────────────────────────────────────
class ReversePosition:
    def __init__(self):
        self.size        = 0.0
        self.entry_price = 0.0
        self.is_open     = False
        self.direction   = None
        self.trimmed     = False

    def open(self, direction, price, amount, balance):
        self.direction   = direction
        self.size        = amount
        self.entry_price = price
        self.is_open     = True
        self.trimmed     = False
        icon = "🔀🟢" if direction == "long" else "🔀🔴"
        print(f"  {icon} 反向开{direction.upper()}  价格: ${price:,.2f}  金额: ${amount:,.2f}  | 总资产: ${balance:,.2f}")

    def trim(self, price, balance):
        trim_size   = self.size * 0.75
        remain_size = self.size * 0.25
        if self.direction == "long":
            pnl_usdc = trim_size * (price - self.entry_price) / self.entry_price
        else:
            pnl_usdc = trim_size * (self.entry_price - price) / self.entry_price

        self.size    = remain_size
        self.trimmed = True
        balance     += pnl_usdc
        print(f"  ✂️  反向仓出75%  价格: ${price:,.2f}  剩余: ${remain_size:,.2f}")
        print(f"     盈亏: {'🟢' if pnl_usdc >= 0 else '🔴'} ${pnl_usdc:+,.2f}  | 总资产: ${balance:,.2f}")
        return balance, pnl_usdc

    def close(self, price, reason, cb, balance):
        if not self.is_open:
            return balance, 0
        if self.direction == "long":
            pnl_pct  = (price - self.entry_price) / self.entry_price * 100
            pnl_usdc = self.size * (price - self.entry_price) / self.entry_price
        else:
            pnl_pct  = (self.entry_price - price) / self.entry_price * 100
            pnl_usdc = self.size * (self.entry_price - price) / self.entry_price

        balance += pnl_usdc
        print(f"  📤 反向仓平仓  原因: {reason}")
        print(f"     入场价: ${self.entry_price:,.2f}  平仓价: ${price:,.2f}")
        print(f"     盈亏: {'🟢' if pnl_usdc >= 0 else '🔴'} ${pnl_usdc:+,.2f} ({pnl_pct:+.2f}%)  | 总资产: ${balance:,.2f}")

        cb.record(pnl_usdc)
        self.__init__()
        return balance, pnl_usdc

# ─── 策略核心 ─────────────────────────────────────────────
def run_strategy(pos: Position, rev: ReversePosition, df: pd.DataFrame, cb: CircuitBreaker, balance: float):
    if not cb.check():
        return balance

    ma_col = f"MA{MA_PERIOD}"
    if pd.isna(df[ma_col].iloc[-1]):
        print("⚠️  MA270 数据不足，跳过")
        return balance

    price          = df["close"].iloc[-1]
    ma             = df[ma_col].iloc[-1]
    diff           = (price - ma) / ma
    order_size     = balance * LEVERAGE * 0.20   # 开仓 20%
    order_add_size = balance * LEVERAGE * 0.10   # 加仓 10%
    rev_size       = balance * LEVERAGE * 0.10   # 反向仓 10%

    now = datetime.now().strftime("%H:%M:%S")
    print(f"\n[{now}] 价格: ${price:,.2f}  MA270: ${ma:,.2f}  偏离: {diff*100:+.4f}%")

    # ══════════════════════════════════════════════════════
    # 反向仓管理（优先处理）
    # ══════════════════════════════════════════════════════
    if rev.is_open:
        if rev.direction == "long":
            if diff >= REVERSE_TRIM and not rev.trimmed:
                balance, pnl = rev.trim(price, balance)
                cb.record(pnl)
            elif diff >= REVERSE_EXIT and rev.trimmed:
                balance, _ = rev.close(price, "✅ 偏离达+1.5% 全平", cb, balance)
            elif diff <= REVERSE_BACK1 and rev.trimmed:
                balance, _ = rev.close(price, "✅ 偏离回到+1.0% 全平", cb, balance)
            elif diff <= REVERSE_BACK2 and not rev.trimmed:
                balance, _ = rev.close(price, "🔀 偏离回到+0.7% 反转开SHORT", cb, balance)
                if not pos.is_open:
                    rev.open("short", price, rev_size, balance)
            else:
                print(f"  🔀 反向LONG持有中  仓位: ${rev.size:,.2f}  {'(已减仓75%)' if rev.trimmed else ''}  | 总资产: ${balance:,.2f}")

        elif rev.direction == "short":
            if diff <= -REVERSE_TRIM and not rev.trimmed:
                balance, pnl = rev.trim(price, balance)
                cb.record(pnl)
            elif diff <= -REVERSE_EXIT and rev.trimmed:
                balance, _ = rev.close(price, "✅ 偏离达-1.5% 全平", cb, balance)
            elif diff >= -REVERSE_BACK1 and rev.trimmed:
                balance, _ = rev.close(price, "✅ 偏离回到-1.0% 全平", cb, balance)
            elif diff >= -REVERSE_BACK2 and not rev.trimmed:
                balance, _ = rev.close(price, "🔀 偏离回到-0.7% 反转开LONG", cb, balance)
                if not pos.is_open:
                    rev.open("long", price, rev_size, balance)
            else:
                print(f"  🔀 反向SHORT持有中  仓位: ${rev.size:,.2f}  {'(已减仓75%)' if rev.trimmed else ''}  | 总资产: ${balance:,.2f}")
        return balance

    # ══════════════════════════════════════════════════════
    # 主仓管理
    # ══════════════════════════════════════════════════════
    if pos.is_open and pos.direction == "short":
        if diff >= STOP_LOSS and not pos.stop_triggered:
            pos.stop_triggered = True
        if pos.stop_triggered:
            balance = pos.close(price, f"🛑 止损 SHORT (偏离 {diff*100:+.4f}%)", cb, balance)
            rev.open("long", price, rev_size, balance)
            return balance
        if price <= ma:
            balance = pos.close(price, "✅ 止盈 SHORT (触碰MA)", cb, balance)
            return balance
        if diff >= ADD_THRESH and not pos.added:
            pos.add(price, order_add_size, balance)
            return balance
        print(f"  ⏸  SHORT持仓中  总仓: ${pos.size:,.2f}  {'(已加仓)' if pos.added else ''}  | 总资产: ${balance:,.2f}")
        return balance

    if pos.is_open and pos.direction == "long":
        if diff <= -STOP_LOSS and not pos.stop_triggered:
            pos.stop_triggered = True
        if pos.stop_triggered:
            balance = pos.close(price, f"🛑 止损 LONG (偏离 {diff*100:+.4f}%)", cb, balance)
            rev.open("short", price, rev_size, balance)
            return balance
        if price >= ma:
            balance = pos.close(price, "✅ 止盈 LONG (触碰MA)", cb, balance)
            return balance
        if diff <= -ADD_THRESH and not pos.added:
            pos.add(price, order_add_size, balance)
            return balance
        print(f"  ⏸  LONG持仓中  总仓: ${pos.size:,.2f}  {'(已加仓)' if pos.added else ''}  | 总资产: ${balance:,.2f}")
        return balance

    # ══════════════════════════════════════════════════════
    # 无仓位，判断开仓
    # ══════════════════════════════════════════════════════
    if diff >= ENTRY_THRESH:
        if diff >= STOP_LOSS:
            print(f"  ⚠️  偏离过大不开仓 ({diff*100:+.4f}%) 已超止损线  | 总资产: ${balance:,.2f}")
        else:
            pos.open("short", price, order_size, order_add_size, balance, diff)
    elif diff <= -ENTRY_THRESH:
        if diff <= -STOP_LOSS:
            print(f"  ⚠️  偏离过大不开仓 ({diff*100:+.4f}%) 已超止损线  | 总资产: ${balance:,.2f}")
        else:
            pos.open("long", price, order_size, order_add_size, balance, diff)
    else:
        print(f"  ⏸  等待信号  (需偏离 ≥ ±{ENTRY_THRESH*100:.2f}%)  | 总资产: ${balance:,.2f}")

    return balance

# ─── 主循环 ───────────────────────────────────────────────
circuit_breaker = CircuitBreaker()

def main():
    pos     = Position()
    rev     = ReversePosition()
    balance = float(TOTAL_CAPITAL)

    print("🚀 策略启动 | BTC-USD | MA270 | 1MIN | 模拟模式")
    print(f"   总资金: ${balance:,.2f}  杠杆: {LEVERAGE}x")
    print(f"   开仓: balance×{LEVERAGE}×20%  加仓: balance×{LEVERAGE}×10%  反向仓: balance×{LEVERAGE}×10%")
    print(f"   熔断设置: 3小时内亏损超过 ${MAX_LOSS_AMOUNT:.2f} 暂停交易")
    print("─" * 55)

    while True:
        try:
            df      = get_candles(400)
            df      = add_ma(df)
            balance = run_strategy(pos, rev, df, circuit_breaker, balance)
        except Exception as e:
            print(f"⚠️  错误: {e}")
        time.sleep(60)

if __name__ == "__main__":
    main()