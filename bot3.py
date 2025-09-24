import time
import os
import logging
import ccxt
import pandas as pd
import numpy as np
from ta.trend import EMAIndicator, MACD
from ta.momentum import RSIIndicator

# ─── تنظیمات لاگ ───
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# ─── اتصال به کوکوین ───
exchange = ccxt.kucoin()

# ─── گرفتن 80 ارز برتر بر اساس حجم 24 ساعته ───
def get_top_symbols(limit=80):
    markets = exchange.load_markets()
    symbols = sorted(
        markets.values(),
        key=lambda x: x['quoteVolume'],
        reverse=True
    )[:limit]
    return [s['symbol'] for s in symbols if '/USDT' in s['symbol']]

# ─── گرفتن دیتا ───
def fetch_ohlcv(symbol, timeframe='15m', limit=150):
    try:
        data = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
        df = pd.DataFrame(data, columns=['time','open','high','low','close','volume'])
        return df
    except Exception as e:
        logging.error(f"❌ خطا در گرفتن دیتا {symbol} - {timeframe}: {e}")
        return None
# ─── افزودن اندیکاتورها ───
def add_indicators(df):
    df['EMA20'] = EMAIndicator(df['close'], window=20).ema_indicator()
    macd = MACD(df['close'])
    df['MACD'] = macd.macd()
    df['Signal'] = macd.macd_signal()
    df['RSI'] = RSIIndicator(df['close'], window=14).rsi()
    return df

# ─── تشخیص کندل‌استیک ساده ───
def detect_candlestick(df):
    last = df.iloc[-1]
    prev = df.iloc[-2]
    pattern = None

    # Doji
    if abs(last['close'] - last['open']) <= (last['high'] - last['low'])*0.1:
        pattern = "Doji"
    # Hammer
    elif (last['close'] > last['open']) and ((last['close'] - last['low']) > 2*(last['high'] - last['close'])):
        pattern = "Hammer"
    # Engulfing
    elif (last['close'] > last['open']) and (prev['close'] < prev['open']) and (last['close'] > prev['open']) and (last['open'] < prev['close']):
        pattern = "Bullish Engulfing"
    elif (last['close'] < last['open']) and (prev['close'] > prev['open']) and (last['open'] > prev['close']) and (last['close'] < prev['open']):
        pattern = "Bearish Engulfing"

    return pattern

# ─── تشخیص خط روند ساده ───
def detect_trendline(prices, window=5, tol=0.01):
    highs, lows = [], []
    for i in range(window, len(prices)-window):
        if prices[i] == max(prices[i-window:i+window+1]):
            highs.append((i, prices[i]))
        if prices[i] == min(prices[i-window:i+window+1]):
            lows.append((i, prices[i]))

    trend_info = {"resistance": None, "support": None, "signal": None}
    last_idx = len(prices)-1
    last_price = prices[-1]

    if len(highs) >= 2:
        xh, yh = zip(*highs[-2:])
        a,b = np.polyfit(xh, yh,1)
        trend_info["resistance"] = (a,b)
        if abs(last_price - (a*last_idx + b))/(a*last_idx + b) < tol:
            trend_info["signal"] = "SELL (نزدیک مقاومت)"

    if len(lows) >= 2:
        xl, yl = zip(*lows[-2:])
        a,b = np.polyfit(xl, yl,1)
        trend_info["support"] = (a,b)
        if abs(last_price - (a*last_idx + b))/(a*last_idx + b) < tol:
            trend_info["signal"] = "BUY (نزدیک حمایت)"

    return trend_info
from telegram import Bot
TELEGRAM_TOKEN = os.getenv("BOT_TOKEN")
CHAT_ID = os.getenv("CHAT_ID")
bot = Bot(token=TELEGRAM_TOKEN)

def send_telegram(msg):
    try:
        bot.send_message(chat_id=CHAT_ID, text=msg)
    except Exception as e:
        logging.error(f"❌ خطا در ارسال تلگرام: {e}")

def generate_signal(df):
    last = df.iloc[-1]
    signals = []

    # شرط های ساده
    if last['RSI'] < 30:
        signals.append("BUY")
    if last['RSI'] > 70:
        signals.append("SELL")
    if last['MACD'] > last['Signal']:
        signals.append("BUY")
    if last['MACD'] < last['Signal']:
        signals.append("SELL")
    if last['close'] > last['EMA20']:
        signals.append("BUY")
    if last['close'] < last['EMA20']:
        signals.append("SELL")

    # کندل استیک
    pattern = detect_candlestick(df)
    if pattern in ["Hammer","Bullish Engulfing"]:
        signals.append("BUY")
    if pattern in ["Shooting Star","Bearish Engulfing"]:
        signals.append("SELL")

    # خط روند
    trend = detect_trendline(df['close'].values)
    if trend["signal"]:
        signals.append(trend["signal"])

    # اگر حداقل دو شرط برقرار بود سیگنال بده
    buy_count = sum([1 for s in signals if "BUY" in s])
    sell_count = sum([1 for s in signals if "SELL" in s])
    if buy_count >=2:
        return "BUY", pattern, trend
    if sell_count >=2:
        return "SELL", pattern, trend
    return None, pattern, trend

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

            logging.info(f"🔎 بررسی {symbol}")
            logging.info(f"کندل: {pattern}, خط روند: {trend['signal']}")

            if signal:
                entry = df['close'].iloc[-1]
                sl = entry * 0.98 if signal=="BUY" else entry * 1.02
                tp = entry * 1.03 if signal=="BUY" else entry * 0.97
                msg = (
                    f"📊 سیگنال {signal} برای {symbol}\n"
                    f"Entry: {entry:.4f}\nStop Loss: {sl:.4f}\nTake Profit: {tp:.4f}\n"
                    f"کندل: {pattern}\nخط روند: {trend['signal']}"
                )
                send_telegram(msg)
                logging.info(msg)

            time.sleep(3)  # فاصله بین هر ارز

        logging.info("⏱️ انتظار 5 دقیقه قبل از بررسی دوباره")
        time.sleep(300)  # بررسی دوباره همه ارزها هر 5 دقیقه

if __name__ == "__main__":
    send_telegram("✅ ربات شروع شد (80 ارز، اندیکاتور + کندل + خط روند)")
    run_bot()
