import os
import time
import logging
import ccxt
import pandas as pd
import numpy as np
from telegram import Bot
from ta.trend import EMAIndicator, MACD
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands

# ─── تنظیمات لاگ ───
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# ─── توکن تلگرام ───
TELEGRAM_TOKEN = os.getenv("BOT_TOKEN")
CHAT_ID = os.getenv("CHAT_ID")
bot = Bot(token=TELEGRAM_TOKEN) if TELEGRAM_TOKEN and CHAT_ID else None

# ─── صرافی ───
exchange = ccxt.kucoin({"enableRateLimit": True})

# ─── فیلتر ضدتکرار سیگنال ───
last_signals = {}
def should_send(symbol, signal):
    now = time.time()
    if symbol not in last_signals:
        last_signals[symbol] = (signal, now)
        return True
    last_signal, last_time = last_signals[symbol]
    if last_signal != signal:
        last_signals[symbol] = (signal, now)
        return True
    if now - last_time > 1800:
        last_signals[symbol] = (signal, now)
        return True
    return False

# ─── گرفتن 80 ارز برتر ───
def get_top_symbols(limit=80):
    try:
        markets = exchange.load_markets()
        sorted_markets = sorted(markets.values(), key=lambda x: x.get('quoteVolume',0), reverse=True)
        symbols = [s['symbol'] for s in sorted_markets if s['symbol'].endswith('/USDT')]
        return symbols[:limit]
    except Exception as e:
        logging.error(f"❌ خطا در بارگذاری مارکت‌ها: {e}")
        return []

# ─── گرفتن داده کندل ───
def fetch_ohlcv_safe(symbol, timeframe="5m", limit=500):
    try:
        data = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
        df = pd.DataFrame(data, columns=["time","open","high","low","close","volume"])
        df["time"] = pd.to_datetime(df["time"], unit='ms')
        return df
    except Exception as e:
        logging.error(f"❌ خطا در fetch_ohlcv {symbol} {timeframe}: {e}")
        return None

# ─── اندیکاتورها و ATR ───
def add_indicators(df):
    if df is None or df.empty:
        return df
    df = df.copy()
    df["EMA20"] = EMAIndicator(close=df["close"], window=20).ema_indicator()
    macd = MACD(close=df["close"])
    df["MACD"] = macd.macd()
    df["Signal"] = macd.macd_signal()
    df["RSI"] = RSIIndicator(close=df["close"], window=14).rsi()
    bb = BollingerBands(close=df["close"], window=20, window_dev=2)
    df["BB_High"] = bb.bollinger_hband()
    df["BB_Low"] = bb.bollinger_lband()
    # ایچیموکو ساده
    df["tenkan"] = (df["high"].rolling(9).max() + df["low"].rolling(9).min()) / 2
    df["kijun"] = (df["high"].rolling(26).max() + df["low"].rolling(26).min()) / 2
    # ATR
    df['H-L'] = df['high'] - df['low']
    df['H-C'] = abs(df['high'] - df['close'].shift())
    df['L-C'] = abs(df['low'] - df['close'].shift())
    df['TR'] = df[['H-L','H-C','L-C']].max(axis=1)
    df['ATR'] = df['TR'].rolling(14).mean()
    df.drop(['H-L','H-C','L-C','TR'], axis=1, inplace=True)
    return df

# ─── فیبوناچی ساده ───
def fibonacci_levels(df, lookback=100):
    if df is None or len(df) < lookback:
        return {}
    high = df["high"].iloc[-lookback:].max()
    low = df["low"].iloc[-lookback:].min()
    diff = high - low
    if diff == 0:
        return {}
    levels = {
        "0.236": high - diff * 0.236,
        "0.382": high - diff * 0.382,
        "0.5": high - diff * 0.5,
        "0.618": high - diff * 0.618,
        "0.786": high - diff * 0.786,
    }
    return levels

# ─── شناسایی کندل ساده ───
def detect_candlestick_from_idx(df, idx=-2):
    if df is None or len(df) < abs(idx):
        return "Unknown"
    last = df.iloc[idx]
    body = abs(last["close"] - last["open"])
    candle = last["high"] - last["low"]
    if candle == 0 or body < candle*0.2:
        return "Doji"
    return "Bullish" if last["close"] > last["open"] else "Bearish"

# ─── تشخیص Order Block با دقت بالا ───
def detect_order_blocks(df, lookback=200, bos_window=5, volume_threshold=1.5, expansion=0.002):
    if df is None or len(df) < 30:
        return []
    obs=[]
    highs, lows, closes, opens, volumes = df["high"].values, df["low"].values, df["close"].values, df["open"].values, df["volume"].values
    n = len(df)
    for i in range(bos_window, min(n, lookback)):
        curr_high, curr_low = highs[i], lows[i]
        prev_highs_max = highs[i-bos_window:i].max()
        prev_lows_min = lows[i-bos_window:i].min()
        # Bullish BOS
        if curr_high > prev_highs_max:
            j=i-1
            while j>=0 and closes[j]>=opens[j]: j-=1
            if j>=0:
                ob_low, ob_high = lows[j], highs[j]
                avg_vol = np.mean(volumes[max(0,i-10):i])
                vol_ratio = volumes[i]/(avg_vol+1e-9)
                body_ratio = abs(closes[j]-opens[j]) / max(1e-9, highs[j]-lows[j])
                score = vol_ratio*0.6 + body_ratio*0.4
                confirmed = vol_ratio>=volume_threshold and body_ratio>=0.25
                buff = expansion*closes[i]
                obs.append({"type":"demand","low":ob_low-buff,"high":ob_high+buff,"time":df["time"].iloc[j],"confirmed":confirmed,"score":score})
        # Bearish BOS
        if curr_low < prev_lows_min:
            j=i-1
            while j>=0 and closes[j]<=opens[j]: j-=1
            if j>=0:
                ob_low, ob_high = lows[j], highs[j]
                avg_vol = np.mean(volumes[max(0,i-10):i])
                vol_ratio = volumes[i]/(avg_vol+1e-9)
                body_ratio = abs(closes[j]-opens[j]) / max(1e-9, highs[j]-lows[j])
                score = vol_ratio*0.6 + body_ratio*0.4
                confirmed = vol_ratio>=volume_threshold and body_ratio>=0.25
                buff = expansion*closes[i]
                obs.append({"type":"supply","low":ob_low-buff,"high":ob_high+buff,"time":df["time"].iloc[j],"confirmed":confirmed,"score":score})
    # merge overlap
    merged=[]
    for ob in obs:
        merged_flag=False
        for m in merged:
            if m["type"]==ob["type"] and not (ob["high"]<m["low"] or ob["low"]>m["high"]):
                m["low"]=min(m["low"],ob["low"])
                m["high"]=max(m["high"],ob["high"])
                m["score"]=max(m["score"],ob["score"])
                m["confirmed"]=m["confirmed"] or ob["confirmed"]
                merged_flag=True
                break
        if not merged_flag: merged.append(ob)
    obs=sorted(merged,key=lambda x:x["time"],reverse=True)
    return obs

# ─── بررسی برخورد قیمت به زون ───
def price_in_zone(price, ob, tolerance_percent=0.0025):
    tol = price*tolerance_percent
    return (ob["low"]-tol)<=price<=(ob["high"]+tol)

# ─── Volume Spike ───
def is_volume_spike(df, idx=-2, lookback=20, threshold=2.0):
    if df is None or len(df)<lookback:
        return False
    last_vol=df["volume"].iloc[idx]
    avg_vol=df["volume"].iloc[idx-lookback:idx].mean()
    return last_vol>=avg_vol*threshold

# ─── Position sizing ───
def calculate_position_size(account_balance, entry_price, sl_price, risk_per_trade=0.02):
    risk_amount = account_balance * risk_per_trade
    return risk_amount / max(abs(entry_price-sl_price),1e-9)

# ─── تولید سیگنال ترکیبی و مدیریت سرمایه ───
def generate_signal_with_ob(symbol, ltf="15m", htf="4h", account_balance=1000):
    df_ltf = fetch_ohlcv_safe(symbol, ltf, limit=500)
    if df_ltf is None or df_ltf.empty or len(df_ltf)<50: return None, None, None
    df_ltf = add_indicators(df_ltf)
    last_price = df_ltf["close"].iloc[-2]
    df_htf = fetch_ohlcv_safe(symbol, htf, limit=500)
    time.sleep(0.5)
    ob_ltf = detect_order_blocks(df_ltf)
    ob_htf = detect_order_blocks(df_htf) if df_htf is not None else []
    pattern = detect_candlestick_from_idx(df_ltf)
    signals=[]
    # اندیکاتورها
    last=df_ltf.iloc[-2]
    if last["RSI"]<30: signals.append("BUY")
    if last["RSI"]>70: signals.append("SELL")
    if last["MACD"]>last["Signal"]: signals.append("BUY")
    if last["MACD"]<last["Signal"]: signals.append("SELL")
    if last["close"]>last["EMA20"]: signals.append("BUY")
    if last["close"]<last["EMA20"]: signals.append("SELL")
    # زون تایید شده Order Block
    in_demand=any(price_in_zone(last_price,ob) and ob["confirmed"] and ob["type"]=="demand" for ob in ob_ltf+ob_htf)
    in_supply=any(price_in_zone(last_price,ob) and ob["confirmed"] and ob["type"]=="supply" for ob in ob_ltf+ob_htf)
    if in_demand: signals.append("BUY")
    if in_supply: signals.append("SELL")
    # Volume Spike
    if is_volume_spike(df_ltf): signals.append("BUY" if signals.count("BUY")>=signals.count("SELL") else "SELL")
    # تایید حداقل 2 فاکتور
    if signals.count("BUY")>=2:
        atr=df_ltf["ATR"].iloc[-2]
        sl=last_price - atr*1.5
        tp=last_price + atr*3
        pos_size=calculate_position_size(account_balance, last_price, sl)
        return "BUY", pattern, {"entry": last_price, "sl": sl, "tp": tp, "position_size": pos_size}
    if signals.count("SELL")>=2:
        atr=df_ltf["ATR"].iloc[-2]
        sl=last_price + atr*1.5
        tp=last_price - atr*3
        pos_size=calculate_position_size(account_balance, last_price, sl)
        return "SELL", pattern, {"entry": last_price, "sl": sl, "tp": tp, "position_size": pos_size}
    return None, pattern, None

# ─── ارسال تلگرام ───
def send_telegram(msg):
    if bot:
        try:
            bot.send_message(chat_id=CHAT_ID, text=msg)
        except Exception as e:
            logging.error(f"❌ خطا در ارسال پیام تلگرام: {e}")

# ─── ربات اصلی ───
def run_bot():
    while True:
        symbols=get_top_symbols(limit=80)
        logging.info(f"🔝 ارزهای انتخاب‌شده: {symbols[:10]} ...")
        for symbol in symbols:
            signal, pattern, details = generate_signal_with_ob(symbol)
            if signal:
                msg=f"📊 سیگنال {signal} برای {symbol}\nEntry: {details['entry']:.4f}\nSL: {details['sl']:.4f}\nTP: {details['tp']:.4f}\nحجم معامله: {details['position_size']:.2f}\nکندل: {pattern}"
                if should_send(symbol, signal):
                    send_telegram(msg)
                    logging.info(msg)
            time.sleep(1)
        logging.info("⏱️ انتظار 5 دقیقه قبل از بررسی دوباره")
        time.sleep(300)

# ─── شروع ───
if __name__=="__main__":
    send_telegram("✅ ربات فعال شد و شروع به کار می‌کند (80 ارز، HTF+LTF OB، Volume, ATR, مدیریت ریسک)")
    logging.info("✅ ربات فعال شد و شروع به کار می‌کند.")
    run_bot()
