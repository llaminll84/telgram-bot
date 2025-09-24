import time
import os
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
api_key = os.getenv("API_KEY")
api_secret = os.getenv("API_SECRET")

exchange = ccxt.kucoin({
    "apiKey": api_key,
    "secret": api_secret,
    "enableRateLimit": True
})

# ─── تنظیمات ترید ───
symbols = ["BTC/USDT", "ETH/USDT", "BNB/USDT"]  # می‌تونی ارزها رو تغییر بدی
timeframe = "15m"
limit = 200
rsi_period = 14
ma_period = 20

# ─── گرفتن دیتا ───
def fetch_ohlcv(symbol, timeframe, limit=200):
    try:
        data = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
        df = pd.DataFrame(data, columns=["timestamp", "open", "high", "low", "close", "volume"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        return df
    except Exception as e:
        logging.error(f"❌ خطا در دریافت دیتا {symbol}: {e}")
        return None

# ─── محاسبه اندیکاتورها ───
def calculate_indicators(df):
    df["MA"] = df["close"].rolling(ma_period).mean()
    delta = df["close"].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(rsi_period).mean()
    avg_loss = loss.rolling(rsi_period).mean()
    rs = avg_gain / avg_loss
    df["RSI"] = 100 - (100 / (1 + rs))
    return df

# ─── سیگنال خرید/فروش ───
def generate_signal(df):
    if df is None or len(df) < ma_period:
        return None
    last_row = df.iloc[-1]
    signal = None
    if last_row["RSI"] < 30 and last_row["close"] > last_row["MA"]:
        signal = "BUY"
    elif last_row["RSI"] > 70 and last_row["close"] < last_row["MA"]:
        signal = "SELL"
    return signal
# ─── ارسال پیام به تلگرام ───
def send_telegram_message(message):
    try:
        bot.send_message(chat_id=CHAT_ID, text=message)
        logging.info(f"📩 پیام ارسال شد: {message}")
    except Exception as e:
        logging.error(f"❌ خطا در ارسال پیام تلگرام: {e}")

# ─── اجرای استراتژی ───
def run_bot():
    while True:
        try:
            for symbol in symbols:
                df = fetch_ohlcv(symbol, timeframe, limit)
                if df is not None:
                    df = calculate_indicators(df)
                    signal = generate_signal(df)
                    if signal:
                        msg = f"📊 سیگنال {signal} برای {symbol}\nقیمت: {df['close'].iloc[-1]:.2f}"
                        send_telegram_message(msg)
                        logging.info(msg)
                time.sleep(2)  # جلوگیری از محدودیت API
        except Exception as e:
            logging.error(f"⚠️ خطای کلی در اجرای ربات: {e}")
        time.sleep(60)  # هر ۱ دقیقه اجرا شود

# ─── اجرای اصلی ───
if __name__ == "__main__":
    send_telegram_message("✅ ربات ترید شروع به کار کرد.")
    run_bot()