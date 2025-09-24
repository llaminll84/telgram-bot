import os
import tim
import loggin
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
bot = Bot(token=TELEGRAM_TOKEN)

# ─── صرافی ───
exchange = ccxt.kucoin()

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
    
    # اجازه ارسال دوباره بعد از 30 دقیقه
    if now - last_time > 1800:
        last_signals[symbol] = (signal, now)
        return True

    return False

# ─── گرفتن 80 ارز برتر ───
def get_top_symbols(limit=80):
    markets = exchange.load_markets()
    symbols = sorted(
        markets.values(),
        key=lambda x: x.get('quoteVolume', 0),
        reverse=True
    )
    symbols = [s['symbol'] for s in symbols if '/USDT' in s['symbol']]
    return symbols[:limit]

# ─── گرفتن داده کندل ───
def fetch_ohlcv(symbol, timeframe="5m", limit=200):
    try:
        data = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
        df = pd.DataFrame(data, columns=["time","open","high","low","close","volume"])
        return df
    except Exception as e:
        logging.error(f"❌ خطا در دریافت داده {symbol}: {e}")
        return None

# ─── افزودن اندیکاتورها ───
def add_indicators(df):
    df["EMA20"] = EMAIndicator(close=df["close"], window=20).ema_indicator()
    macd = MACD(close=df["close"])
    df["MACD"] = macd.macd()
    df["Signal"] = macd.macd_signal()
    df["RSI"] = RSIIndicator(close=df["close"]).rsi()
    bb = BollingerBands(close=df["close"])
    df["BB_High"] = bb.bollinger_hband()
    df["BB_Low"] = bb.bollinger_lband()
    return df

# ─── شناسایی الگوی ساده کندل ───
def detect_candlestick(df):
    last = df.iloc[-1]
    body = abs(last["close"] - last["open"])
    candle = last["high"] - last["low"]
    if candle == 0:
        return "Doji"
    if body < (candle * 0.2):
        return "Doji"
    if last["close"] > last["open"]:
        return "Bullish"
    else:
        return "Bearish"

# ─── تشخیص خط روند خیلی ساده ───
def detect_trendline(prices):
    if len(prices) < 5:
        return {"signal": None}
    if prices[-1] > np.mean(prices[-5:]):
        return {"signal": "BUY"}
    elif prices[-1] < np.mean(prices[-5:]):
        return {"signal": "SELL"}
    return {"signal": None}

# ─── تولید سیگنال ───
def generate_signal(df):
    last = df.iloc[-1]
    signals = []

    if last["RSI"] < 30:
        signals.append("BUY")
    if last["RSI"] > 70:
        signals.append("SELL")
    if last["MACD"] > last["Signal"]:
        signals.append("BUY")
    if last["MACD"] < last["Signal"]:
        signals.append("SELL")
    if last["close"] > last["EMA20"]:
        signals.append("BUY")
    if last["close"] < last["EMA20"]:
        signals.append("SELL")

    pattern = detect_candlestick(df)
    if pattern == "Bullish":
        signals.append("BUY")
    if pattern == "Bearish":
        signals.append("SELL")

    trend = detect_trendline(df["close"].values)
    if trend["signal"]:
        signals.append(trend["signal"])

    buy_count = signals.count("BUY")
    sell_count = signals.count("SELL")
    if buy_count >= 2:
        return "BUY", pattern, trend
    if sell_count >= 2:
        return "SELL", pattern, trend
    return None, pattern, trend

# ─── ارسال پیام تلگرام ───
def send_telegram(msg):
    try:
        bot.send_message(chat_id=CHAT_ID, text=msg)
    except Exception as e:
        logging.error(f"❌ خطا در ارسال پیام تلگرام: {e}")

# ─── اجرای ربات ───
def run_bot():
    while True:
        symbols = get_top_symbols(limit=80)
        logging.info(f"🔝 ارزهای انتخاب‌شده: {symbols[:10]} ...")

        for symbol in symbols:
            df = fetch_ohlcv(symbol)
            if df is None or df.empty:
                continue
            df = add_indicators(df)
            signal, pattern, trend = generate_signal(df)

            logging.info(f"🔎 بررسی {symbol} | کندل: {pattern}, خط روند: {trend['signal']}")

            if signal:
                entry = df["close"].iloc[-1]
                sl = entry * 0.98 if signal == "BUY" else entry * 1.02
                tp = entry * 1.03 if signal == "BUY" else entry * 0.97
                msg = (
                    f"📊 سیگنال {signal} برای {symbol}\n"
                    f"Entry: {entry:.4f}\nStop Loss: {sl:.4f}\nTake Profit: {tp:.4f}\n"
                    f"کندل: {pattern}\nخط روند: {trend['signal']}"
                )
                if should_send(symbol, signal):
                    send_telegram(msg)
                    logging.info(msg)

            time.sleep(2)  # فاصله بین بررسی هر ارز

        logging.info("⏱️ انتظار 5 دقیقه قبل از بررسی دوباره")
        time.sleep(300)

# ─── شروع ───
if __name__ == "__main__":
    send_telegram("✅ ربات فعال شد و شروع به کار می‌کند (80 ارز، هر 5 دقیقه بررسی)")
    logging.info("✅ ربات فعال شد و شروع به کار می‌کند.")
    run_bot()
