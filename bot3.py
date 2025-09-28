import time
import os
import logging
import ccxt
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from telegram import Bot
from keep_alive import keep_alive

# ─── فعال کردن سرور کوچک ───
keep_alive()

# ─── اطلاعات ربات تلگرام ───
TELEGRAM_TOKEN = os.getenv("BOT_TOKEN")
CHAT_ID = os.getenv("CHAT_ID")
bot = Bot(token=TELEGRAM_TOKEN)

# ─── تنظیمات لاگینگ ───
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# ─── اتصال به صرافی ───
exchange = ccxt.kucoin()

# ─── تایم‌فریم‌ها و سیمبل‌ها ───
timeframes = ['5m', '15m', '1h']
symbols = ['BTC/USDT', 'ETH/USDT', 'BNB/USDT']

# ─── اندیکاتورها ───
def calculate_indicators(df):
    if df is None or len(df) < 60:
        return df

    df['EMA9'] = df['close'].ewm(span=9).mean()
    df['EMA21'] = df['close'].ewm(span=21).mean()
    df['RSI'] = compute_rsi(df['close'], 14)
    df['ATR14'] = compute_atr(df, 14)
    df['trend'] = np.where(df['EMA9'] > df['EMA21'], 'bullish', 'bearish')
    return df

def compute_rsi(series, period=14):
    delta = series.diff()
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(period).mean()
    avg_loss = pd.Series(loss).rolling(period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def compute_atr(df, period=14):
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift())
    low_close = np.abs(df['low'] - df['close'].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return tr.rolling(period).mean()

# ─── دریافت دیتا ───
def get_klines(symbol, interval, limit=100):
    try:
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe=interval, limit=limit)
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        return df
    except Exception as e:
        logging.error(f"❌ خطا در دریافت داده {symbol} {interval}: {e}")
        return None

# ─── سیگنال گیری ───
def check_signal(df, interval):
    if df is None or len(df) < 60:
        return None

    price = df['close'].iloc[-1]
    prev_price = df['close'].iloc[-2]
    change = ((price - prev_price) / prev_price) * 100
    trend = df['trend'].iloc[-1]

    # candlestick patterns (شبیه ستاره‌ها)
    stars = [col for col in df.columns if isinstance(df[col].iloc[-1], (int, float)) and df[col].iloc[-1] == 1]

    atr = df['ATR14'].iloc[-1]
    rsi = df['RSI'].iloc[-1]
    ema9 = df['EMA9'].iloc[-1]
    ema21 = df['EMA21'].iloc[-1]

    signal_type = None
    entry = stop = tp = None

    # ---- لاگ دیباگ ----
    logging.info(f"{interval} | Δ={change:.2f}% | Trend={trend} | RSI={rsi:.1f} | Stars={len(stars)} | EMA9={ema9:.2f} | EMA21={ema21:.2f} | ATR={atr:.4f}")

    if (change >= 0.8 and trend == 'bullish' and len(stars) >= 2 
        and ema9 > ema21 and rsi > 45):
        signal_type = 'LONG'
        entry = price
        stop = price - 1.2 * atr
        tp = price + 1.8 * atr

    elif (change <= -0.8 and trend == 'bearish' and len(stars) >= 2 
          and ema9 < ema21 and rsi < 55):
        signal_type = 'SHORT'
        entry = price
        stop = price + 1.2 * atr
        tp = price - 1.8 * atr

    return {
        'type': signal_type,
        'entry': entry,
        'stop': stop,
        'tp': tp,
        'change': change,
        'trend': trend,
        'stars': stars,
        'RSI': rsi
    } if signal_type else None

# ─── پیام تلگرام ───
def send_telegram_alert(message):
    try:
        bot.send_message(chat_id=CHAT_ID, text=message, parse_mode="HTML")
    except Exception as e:
        logging.error(f"❌ خطا در ارسال تلگرام: {e}")

# ─── اجرای اصلی ───
def main():
    while True:
        alerts = []
        for symbol in symbols:
            tf_signals = []
            for interval in timeframes:
                df = get_klines(symbol, interval)
                df = calculate_indicators(df)
                sig = check_signal(df, interval)
                if sig:
                    tf_signals.append((interval, sig))

            if tf_signals:
                main_sig = tf_signals[0][1]
                valid = False
                for tf, sig in tf_signals:
                    if tf in ['15m', '1h'] and sig['type'] == main_sig['type']:
                        valid = True
                        break

                if valid:
                    alerts.append((symbol, tf_signals))

        for symbol, tf_signals in alerts:
            for tf, sig in tf_signals:
                message = (
                    f"🚨 سیگنال {sig['type']} در {symbol} | TF: {tf}\n"
                    f"💰 ورود: {sig['entry']:.2f}\n"
                    f"⛔ استاپ: {sig['stop']:.2f}\n"
                    f"🎯 تارگت: {sig['tp']:.2f}\n"
                    f"📊 RSI: {sig['RSI']:.1f} | تغییر: {sig['change']:.2f}% | Trend: {sig['trend']}\n"
                )
                send_telegram_alert(message)

        time.sleep(60)

if __name__ == "__main__":
    main()