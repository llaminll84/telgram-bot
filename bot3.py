import os
import tim
import logging
import ccxt
import pandas as pd
import numpy as np
from telegram import Bot
from keep_alive import keep_alive

# ─── فعال کردن سرور کوچک ───
keep_alive()

# ─── اطلاعات ربات تلگرام ───
TELEGRAM_TOKEN = os.getenv("BOT_TOKEN")
CHAT_ID = os.getenv("CHAT_ID")
bot = Bot(token=TELEGRAM_TOKEN)

# ─── تنظیمات لاگ ───
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)

# ─── اتصال به صرافی ───
API_KEY = os.getenv("API_KEY")
API_SECRET = os.getenv("API_SECRET")

exchange = ccxt.kucoin({
    "apiKey": API_KEY,
    "secret": API_SECRET,
    "enableRateLimit": True
})

# ─── ارسال پیام به تلگرام ───
def send_telegram_message(message):
    try:
        bot.send_message(chat_id=CHAT_ID, text=message)
        logging.info(f"📩 پیام ارسال شد: {message}")
    except Exception as e:
        logging.error(f"❌ خطا در ارسال پیام تلگرام: {e}")

# ─── گرفتن ۶۰ ارز برتر بر اساس حجم ۲۴ ساعته ───
def get_top_symbols(limit=60):
    tickers = exchange.fetch_tickers()
    df = pd.DataFrame([
        {"symbol": sym, "volume": t["quoteVolume"]}
        for sym, t in tickers.items() if "/USDT" in sym
    ])
    df = df.sort_values("volume", ascending=False).head(limit)
    return df["symbol"].tolist()

# ─── گرفتن داده کندل ───
def get_ohlcv(symbol, timeframe="15m", limit=150):
    try:
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
        df = pd.DataFrame(ohlcv, columns=["time", "open", "high", "low", "close", "volume"])
        df["time"] = pd.to_datetime(df["time"], unit="ms")
        return df
    except Exception as e:
        logging.error(f"⚠️ خطا در دریافت داده {symbol} تایم‌فریم {timeframe}: {e}")
        return None

# ─── اندیکاتورها + فیبوناچی + ATR ───
def add_indicators(df):
    # MA
    df["MA20"] = df["close"].rolling(20).mean()
    df["MA50"] = df["close"].rolling(50).mean()

    # EMA
    df["EMA20"] = df["close"].ewm(span=20).mean()
    df["EMA50"] = df["close"].ewm(span=50).mean()

    # Bollinger Bands
    df["BB_MID"] = df["close"].rolling(20).mean()
    df["BB_STD"] = df["close"].rolling(20).std()
    df["BB_UPPER"] = df["BB_MID"] + 2 * df["BB_STD"]
    df["BB_LOWER"] = df["BB_MID"] - 2 * df["BB_STD"]

    # RSI
    delta = df["close"].diff()
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(14).mean()
    avg_loss = pd.Series(loss).rolling(14).mean()
    rs = avg_gain / avg_loss
    df["RSI"] = 100 - (100 / (1 + rs))

    # MACD
    ema12 = df["close"].ewm(span=12).mean()
    ema26 = df["close"].ewm(span=26).mean()
    df["MACD"] = ema12 - ema26
    df["Signal"] = df["MACD"].ewm(span=9).mean()

    # Ichimoku (ساده)
    high9 = df["high"].rolling(9).max()
    low9 = df["low"].rolling(9).min()
    df["Tenkan"] = (high9 + low9) / 2
    high26 = df["high"].rolling(26).max()
    low26 = df["low"].rolling(26).min()
    df["Kijun"] = (high26 + low26) / 2

    # ATR
    df["H-L"] = df["high"] - df["low"]
    df["H-C"] = abs(df["high"] - df["close"].shift())
    df["L-C"] = abs(df["low"] - df["close"].shift())
    df["TR"] = df[["H-L","H-C","L-C"]].max(axis=1)
    df["ATR14"] = df["TR"].rolling(14).mean()

    # فیبوناچی (آخرین 26 کندل)
    if len(df) >= 26:
        recent = df["close"].iloc[-26:]
        high = recent.max()
        low = recent.min()
        diff = high - low
        df["Fib23.6"] = high - 0.236 * diff
        df["Fib38.2"] = high - 0.382 * diff
        df["Fib50"] = high - 0.5 * diff
        df["Fib61.8"] = high - 0.618 * diff

    return df
# ─── تایم‌فریم‌ها ───
timeframes = ["5m", "15m", "1h", "4h"]

# ─── تحلیل سیگنال با ترکیب اندیکاتورها + فیبوناچی + ATR ───
def analyze(df):
    if df is None or len(df) < 50:
        return None

    last = df.iloc[-1]

    # شرایط ساده برای BUY و SELL (نمونه اولیه، قابل توسعه)
    buy_cond = (
        last["close"] > last["MA20"] and
        last["close"] > last["EMA20"] and
        last["RSI"] < 70 and
        last["MACD"] > last["Signal"] and
        last["close"] > last.get("Fib38.2", 0)  # مثال: بالاتر از فیبو 38.2
    )

    sell_cond = (
        last["close"] < last["MA20"] and
        last["close"] < last["EMA20"] and
        last["RSI"] > 30 and
        last["MACD"] < last["Signal"] and
        last["close"] < last.get("Fib61.8", 0)  # مثال: پایین‌تر از فیبو 61.8
    )

    if buy_cond:
        return "BUY"
    elif sell_cond:
        return "SELL"
    return None

# ─── تولید سیگنال تایید چند تایم‌فریم ───
def generate_signal(symbol):
    signals = []
    for tf in timeframes:
        df = get_ohlcv(symbol, tf)
        df = add_indicators(df)
        sig = analyze(df)
        if sig:
            signals.append(sig)

    # حداقل ۲ تایم‌فریم هم‌نظر
    if signals.count("BUY") >= 2:
        return "BUY"
    elif signals.count("SELL") >= 2:
        return "SELL"
    return None
# ─── اجرای ربات ───
def run_bot():
    symbols = get_top_symbols()
    logging.info(f"🔝 ارزهای انتخاب‌شده: {symbols[:10]} ...")

    while True:
        try:
            for symbol in symbols:
                signal = generate_signal(symbol)
                if signal:
                    msg = f"📊 سیگنال {signal} برای {symbol} (تأیید حداقل ۲ تایم‌فریم)"
                    send_telegram_message(msg)
                    logging.info(msg)
                time.sleep(2)  # جلوگیری از محدودیت API
        except Exception as e:
            logging.error(f"⚠️ خطای کلی در اجرای ربات: {e}")
        time.sleep(60)  # بررسی هر ۶۰ ثانیه

# ─── اجرای اصلی ───
if __name__ == "__main__":
    send_telegram_message("✅ ربات شروع شد (۶۰ ارز + ۴ تایم‌فریم + اندیکاتورها + فیبوناچی + ATR).")
    run_bot()