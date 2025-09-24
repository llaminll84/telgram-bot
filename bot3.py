import time
import os
import logging
import ccxt
import pandas as pd
import numpy as np
from telegram import Bot
from ta.trend import EMAIndicator, SMAIndicator, MACD
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands, AverageTrueRange

# ─── فعال کردن لاگ ───
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# ─── اطلاعات ربات تلگرام ───
TELEGRAM_TOKEN = os.getenv("BOT_TOKEN")
CHAT_ID = os.getenv("CHAT_ID")
bot = Bot(token=TELEGRAM_TOKEN)

# ─── تنظیمات صرافی ───
exchange = ccxt.kucoin()

# ─── گرفتن 60 ارز برتر بر اساس حجم 24 ساعته ───
def get_top_symbols(limit=60):
    tickers = exchange.fetch_tickers()
    sorted_pairs = sorted(tickers.items(), key=lambda x: x[1]['quoteVolume'], reverse=True)
    usdt_pairs = [s for s, data in sorted_pairs if s.endswith("/USDT")]
    return usdt_pairs[:limit]

# ─── گرفتن کندل‌ها ───
def get_ohlcv(symbol, timeframe="5m", limit=100):
    data = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
    df = pd.DataFrame(data, columns=["time", "open", "high", "low", "close", "volume"])
    df["time"] = pd.to_datetime(df["time"], unit="ms")
    return df
# ─── محاسبه اندیکاتورها ───
def add_indicators(df):
    df["EMA20"] = EMAIndicator(df["close"], window=20).ema_indicator()
    df["MA20"] = SMAIndicator(df["close"], window=20).sma_indicator()
    df["RSI"] = RSIIndicator(df["close"], window=14).rsi()

    macd = MACD(df["close"])
    df["MACD"] = macd.macd()
    df["Signal"] = macd.macd_signal()

    atr = AverageTrueRange(df["high"], df["low"], df["close"], window=14)
    df["ATR"] = atr.average_true_range()

    # محاسبه سطوح فیبوناچی
    high, low = df["high"].max(), df["low"].min()
    df["Fib38.2"] = high - (high - low) * 0.382
    df["Fib61.8"] = high - (high - low) * 0.618

    return df


# ─── تشخیص الگوهای کندلی ───
def detect_candlestick_patterns(df):
    last = df.iloc[-1]
    prev = df.iloc[-2]

    pattern = None

    # Doji
    if abs(last["close"] - last["open"]) <= (last["high"] - last["low"]) * 0.1:
        pattern = "Doji"

    # Hammer
    elif (last["close"] > last["open"]) and ((last["close"] - last["low"]) > 2 * (last["high"] - last["close"])):
        pattern = "Hammer"

    # Shooting Star
    elif (last["open"] > last["close"]) and ((last["high"] - last["close"]) > 2 * (last["close"] - last["low"])):
        pattern = "Shooting Star"

    # Bullish Engulfing
    elif (last["close"] > last["open"]) and (prev["close"] < prev["open"]) \
         and (last["close"] > prev["open"]) and (last["open"] < prev["close"]):
        pattern = "Bullish Engulfing"

    # Bearish Engulfing
    elif (last["close"] < last["open"]) and (prev["close"] > prev["open"]) \
         and (last["open"] > prev["close"]) and (last["close"] < prev["open"]):
        pattern = "Bearish Engulfing"

    return pattern


# ─── بررسی شرایط سخت‌تر ───
def check_strict_conditions(last):
    conditions = {"buy": 0, "sell": 0}

    # MA/EMA
    if last["close"] > last["MA20"] * 1.01 and last["close"] > last["EMA20"] * 1.01:
        conditions["buy"] += 1
    if last["close"] < last["MA20"] * 0.99 and last["close"] < last["EMA20"] * 0.99:
        conditions["sell"] += 1

    # RSI
    if last["RSI"] < 60:
        conditions["buy"] += 1
    if last["RSI"] > 40:
        conditions["sell"] += 1

    # MACD
    if last["MACD"] - last["Signal"] > 0.0001:
        conditions["buy"] += 1
    if last["MACD"] - last["Signal"] < -0.0001:
        conditions["sell"] += 1

    # Fibonacci
    if last["close"] > last["Fib38.2"]:
        conditions["buy"] += 1
    if last["close"] < last["Fib61.8"]:
        conditions["sell"] += 1

    return conditions
# ─── ارسال پیام به تلگرام ───
def send_telegram_message(msg):
    try:
        bot.send_message(chat_id=CHAT_ID, text=msg)
    except Exception as e:
        logging.error(f"❌ خطا در ارسال پیام تلگرام: {e}")


# ─── تولید سیگنال ───
def generate_signal(symbol):
    try:
        df = get_ohlcv(symbol, timeframe="5m", limit=100)
        df = add_indicators(df)
        last = df.iloc[-1]

        conditions = check_strict_conditions(last)
        pattern = detect_candlestick_patterns(df)

        # حداقل 2 شرط لازم برای سیگنال
        if conditions["buy"] >= 2:
            entry = last["close"]
            sl = entry - last["ATR"] * 2
            tp = entry + last["ATR"] * 3
            return {
                "symbol": symbol,
                "signal": "BUY",
                "entry": entry,
                "sl": sl,
                "tp": tp,
                "pattern": pattern
            }

        elif conditions["sell"] >= 2:
            entry = last["close"]
            sl = entry + last["ATR"] * 2
            tp = entry - last["ATR"] * 3
            return {
                "symbol": symbol,
                "signal": "SELL",
                "entry": entry,
                "sl": sl,
                "tp": tp,
                "pattern": pattern
            }

        return None

    except Exception as e:
        logging.error(f"⚠️ خطا در generate_signal برای {symbol}: {e}")
        return None


# ─── اجرای ربات ───
def run_bot():
    while True:
        symbols = get_top_symbols(limit=60)
        logging.info(f"🔝 ارزهای انتخاب‌شده: {symbols[:10]} ...")

        for symbol in symbols:
            sig = generate_signal(symbol)
            if sig:
                msg = (
                    f"📊 سیگنال {sig['signal']} برای {sig['symbol']}\n"
                    f"Entry: {sig['entry']:.4f}\n"
                    f"Stop Loss: {sig['sl']:.4f}\n"
                    f"Take Profit: {sig['tp']:.4f}"
                )
                if sig["pattern"]:
                    msg += f"\n📌 الگوی کندلی: {sig['pattern']}"

                send_telegram_message(msg)
                logging.info(msg)

            time.sleep(10)  # فاصله بین درخواست هر ارز

        time.sleep(300)  # بررسی دوباره هر 5 دقیقه


# ─── اجرای اصلی ───
if __name__ == "__main__":
    send_telegram_message("✅ ربات شروع شد (اندیکاتورها + کندل‌استیک‌ها + بررسی هر 5 دقیقه).")
    run_bot()