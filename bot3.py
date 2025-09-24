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

# ─── اندیکاتورها + Fibonacci + ATR ───
def add_indicators(df):
    # MA
    df["MA20"] = df["close"].rolling(20).mean()
    df["EMA20"] = df["close"].ewm(span=20).mean()

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

    # ATR
    df["H-L"] = df["high"] - df["low"]
    df["H-C"] = abs(df["high"] - df["close"].shift())
    df["L-C"] = abs(df["low"] - df["close"].shift())
    df["TR"] = df[["H-L","H-C","L-C"]].max(axis=1)
    df["ATR14"] = df["TR"].rolling(14).mean()

    # Fibonacci (آخرین 26 کندل)
    if len(df) >= 26:
        recent = df["close"].iloc[-26:]
        high = recent.max()
        low = recent.min()
        diff = high - low
        df["Fib38.2"] = high - 0.382 * diff
        df["Fib61.8"] = high - 0.618 * diff

    return df
# ─── تایم‌فریم‌ها برای تأیید ───
TIMEFRAMES = ["15m", "1h"]  # تایم‌فریم اول و دوم

# ─── بررسی شرط‌های اندیکاتورها برای BUY/SELL ───
def check_conditions(last):
    conditions = {"buy": 0, "sell": 0}
    
    # شرط‌های BUY
    if last["close"] > last["MA20"] and last["close"] > last["EMA20"]:
        conditions["buy"] += 1
    if last["RSI"] < 70:
        conditions["buy"] += 1
    if last["MACD"] > last["Signal"]:
        conditions["buy"] += 1
    if last["close"] > last.get("Fib38.2", 0):
        conditions["buy"] += 1

    # شرط‌های SELL
    if last["close"] < last["MA20"] and last["close"] < last["EMA20"]:
        conditions["sell"] += 1
    if last["RSI"] > 30:
        conditions["sell"] += 1
    if last["MACD"] < last["Signal"]:
        conditions["sell"] += 1
    if last["close"] < last.get("Fib61.8", 0):
        conditions["sell"] += 1

    return conditions

# ─── تشخیص واگرایی ساده RSI/MACD ───
def check_divergence(df):
    if len(df) < 15:
        return None
    last = df.iloc[-1]
    prev = df.iloc[-5]
    
    # Bullish Divergence
    if last["close"] < prev["close"] and last["RSI"] > prev["RSI"]:
        return "Bullish Divergence"
    # Bearish Divergence
    if last["close"] > prev["close"] and last["RSI"] < prev["RSI"]:
        return "Bearish Divergence"
    return None

# ─── تحلیل یک تایم‌فریم ───
def analyze_timeframe(df):
    last = df.iloc[-1]
    conditions = check_conditions(last)
    divergence = check_divergence(df)
    
    # حداقل 3 شرط تایید
    if conditions["buy"] >= 3:
        return "BUY", last, divergence
    elif conditions["sell"] >= 3:
        return "SELL", last, divergence
    return None, last, divergence

# ─── تولید سیگنال با تأیید حداقل دو تایم‌فریم ───
def generate_signal(symbol):
    results = []
    lasts = []
    divergences = []

    for tf in TIMEFRAMES:
        df = get_ohlcv(symbol, tf)
        df = add_indicators(df)
        sig, last, div = analyze_timeframe(df)
        results.append(sig)
        lasts.append(last)
        divergences.append(div)

    # حداقل ۲ تایم‌فریم موافق
    if results.count("BUY") >= 2:
        entry = lasts[0]["close"]
        atr = lasts[0]["ATR14"] if "ATR14" in lasts[0] else 0
        sl = entry - atr * 1.5 if atr > 0 else entry * 0.98
        tp = entry + atr * 3 if atr > 0 else entry * 1.03
        div_msg = divergences[0] if divergences[0] else ""
        return {"signal": "BUY", "entry": entry, "sl": sl, "tp": tp, "divergence": div_msg}

    elif results.count("SELL") >= 2:
        entry = lasts[0]["close"]
        atr = lasts[0]["ATR14"] if "ATR14" in lasts[0] else 0
        sl = entry + atr * 1.5 if atr > 0 else entry * 1.02
        tp = entry - atr * 3 if atr > 0 else entry * 0.97
        div_msg = divergences[0] if divergences[0] else ""
        return {"signal": "SELL", "entry": entry, "sl": sl, "tp": tp, "divergence": div_msg}

    return None
# ─── اجرای ربات ───
def run_bot():
    symbols = get_top_symbols()
    logging.info(f"🔝 ارزهای انتخاب‌شده: {symbols[:10]} ...")

    while True:
        try:
            for symbol in symbols:
                sig = generate_signal(symbol)
                if sig:
                    div_text = f"\n⚡ واگرایی: {sig['divergence']}" if sig.get("divergence") else ""
                    msg = (
                        f"📊 سیگنال {sig['signal']} برای {symbol}\n"
                        f"Entry: {sig['entry']:.4f}\n"
                        f"Stop Loss: {sig['sl']:.4f}\n"
                        f"Take Profit: {sig['tp']:.4f}{div_text}"
                    )
                    send_telegram_message(msg)
                    logging.info(msg)
                time.sleep(2)  # جلوگیری از محدودیت API
        except Exception as e:
            logging.error(f"⚠️ خطای کلی در اجرای ربات: {e}")
        time.sleep(60)  # بررسی هر ۶۰ ثانیه

# ─── اجرای اصلی ───
if __name__ == "__main__":
    send_telegram_message("✅ ربات شروع شد (سیگنال با تایید دو تایم‌فریم، اندیکاتورهای کم‌خطا و واگرایی).")
    run_bot()