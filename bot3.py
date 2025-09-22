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

# ─── اطلاعات ربات ───
TELEGRAM_TOKEN = os.getenv("BOT_TOKEN")
CHAT_ID = os.getenv("CHAT_ID")
bot = Bot(token=TELEGRAM_TOKEN)

bot.send_message(chat_id=CHAT_ID, text="✅ ربات استارت شد و به درستی فعال است.")

# ─── تنظیمات لاگینگ ───
logging.basicConfig(level=logging.INFO)

# ─── اتصال به صرافی ───
exchange = ccxt.binance({
    "enableRateLimit": True
})

# ─── محاسبه اندیکاتورها ───
def ichimoku(df):
    high_prices = df['high']
    low_prices = df['low']

    nine_period_high = high_prices.rolling(window=9).max()
    nine_period_low = low_prices.rolling(window=9).min()
    df['Tenkan'] = (nine_period_high + nine_period_low) / 2

    period26_high = high_prices.rolling(window=26).max()
    period26_low = low_prices.rolling(window=26).min()
    df['Kijun'] = (period26_high + period26_low) / 2

    df['SenkouA'] = ((df['Tenkan'] + df['Kijun']) / 2).shift(26)

    period52_high = high_prices.rolling(window=52).max()
    period52_low = low_prices.rolling(window=52).min()
    df['SenkouB'] = ((period52_high + period52_low) / 2).shift(26)

    df['Chikou'] = df['close'].shift(-26)
    return df

def calculate_stoch_rsi(df, period=14, smoothK=3, smoothD=3):
    delta = df['close'].diff()
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)

    avg_gain = pd.Series(gain).rolling(window=period).mean()
    avg_loss = pd.Series(loss).rolling(window=period).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))

    stoch_rsi = (rsi - rsi.rolling(period).min()) / (rsi.rolling(period).max() - rsi.rolling(period).min())
    df['StochRSI'] = stoch_rsi.rolling(smoothK).mean()
    return df

def calculate_atr(df, period=14):
    df['H-L'] = df['high'] - df['low']
    df['H-C'] = abs(df['high'] - df['close'].shift())
    df['L-C'] = abs(df['low'] - df['close'].shift())
    df['TR'] = df[['H-L', 'H-C', 'L-C']].max(axis=1)
    df['ATR'] = df['TR'].rolling(window=period).mean()
    return df

def detect_candlestick_patterns(df):
    patterns = []
    o, h, l, c = df.iloc[-2][['open', 'high', 'low', 'close']], df.iloc[-1][['open', 'high', 'low', 'close']], df.iloc[-1]['low'], df.iloc[-1]['close']

    if df['close'].iloc[-2] < df['open'].iloc[-2] and df['close'].iloc[-1] > df['open'].iloc[-1] and df['close'].iloc[-1] > df['open'].iloc[-2]:
        patterns.append('Bullish Engulfing')
    if df['close'].iloc[-2] > df['open'].iloc[-2] and df['close'].iloc[-1] < df['open'].iloc[-1] and df['close'].iloc[-1] < df['open'].iloc[-2]:
        patterns.append('Bearish Engulfing')
    if (df['close'].iloc[-1] > df['open'].iloc[-1] and 
        (df['low'].iloc[-1] < min(df['close'].iloc[-2], df['open'].iloc[-2])) and 
        (df['close'].iloc[-1] > (df['open'].iloc[-2] + df['close'].iloc[-2]) / 2)):
        patterns.append('Morning Star')
    if (df['close'].iloc[-1] < df['open'].iloc[-1] and 
        (df['high'].iloc[-1] > max(df['close'].iloc[-2], df['open'].iloc[-2])) and 
        (df['close'].iloc[-1] < (df['open'].iloc[-2] + df['close'].iloc[-2]) / 2)):
        patterns.append('Evening Star')
    return patterns

def detect_order_block(df, period=20):
    order_blocks = []
    recent = df.tail(period)

    if recent['close'].iloc[-1] > recent['open'].iloc[-1] and recent['close'].max() > recent['close'].iloc[-2]:
        order_blocks.append("Bullish Order Block")
    if recent['close'].iloc[-1] < recent['open'].iloc[-1] and recent['close'].min() < recent['close'].iloc[-2]:
        order_blocks.append("Bearish Order Block")
    return order_blocks

def check_trend_strength(df):
    if df['close'].iloc[-1] > df['SenkouA'].iloc[-1] and df['close'].iloc[-1] > df['SenkouB'].iloc[-1]:
        if df['Tenkan'].iloc[-1] > df['Kijun'].iloc[-1]:
            return "strong_bullish"
        return "bullish"
    elif df['close'].iloc[-1] < df['SenkouA'].iloc[-1] and df['close'].iloc[-1] < df['SenkouB'].iloc[-1]:
        if df['Tenkan'].iloc[-1] < df['Kijun'].iloc[-1]:
            return "strong_bearish"
        return "bearish"
    return "neutral"

# ─── محاسبه فیبوناچی ───
def calculate_fibonacci(df, period=100):
    """ محاسبه سطوح فیبوناچی از آخرین سقف و کف """
    recent = df.tail(period)
    high = recent['high'].max()
    low = recent['low'].min()

    diff = high - low
    levels = {
        "0.236": high - diff * 0.236,
        "0.382": high - diff * 0.382,
        "0.5": high - diff * 0.5,
        "0.618": high - diff * 0.618,
        "0.786": high - diff * 0.786
    }
    return levels

def position_size(entry, stop, risk=0.01, capital=1000):
    risk_amount = capital * risk
    trade_risk = abs(entry - stop)
    size = risk_amount / trade_risk if trade_risk != 0 else 0
    return size
def generate_signal(df):
    trend = check_trend_strength(df)
    patterns = detect_candlestick_patterns(df)
    order_blocks = detect_order_block(df)
    fibonacci_levels = calculate_fibonacci(df)

    signal = None
    stop_loss = None
    take_profit = None

    atr = df['ATR'].iloc[-1]
    entry = df['close'].iloc[-1]

    # ─── شرایط ورود خرید ───
    if "Bullish Engulfing" in patterns or "Morning Star" in patterns or "Bullish Order Block" in order_blocks:
        if "bullish" in trend:
            stop_loss = entry - atr
            take_profit = entry + (atr * 2)
            signal = f"📈 خرید (Bullish) | Entry: {entry:.2f}, SL: {stop_loss:.2f}, TP: {take_profit:.2f}"

    # ─── شرایط ورود فروش ───
    elif "Bearish Engulfing" in patterns or "Evening Star" in patterns or "Bearish Order Block" in order_blocks:
        if "bearish" in trend:
            stop_loss = entry + atr
            take_profit = entry - (atr * 2)
            signal = f"📉 فروش (Bearish) | Entry: {entry:.2f}, SL: {stop_loss:.2f}, TP: {take_profit:.2f}"

    # ─── خروجی شامل فیبوناچی ───
    if signal:
        fibo_text = "\n📊 سطوح فیبوناچی:\n" + "\n".join([f"{k}: {v:.2f}" for k, v in fibonacci_levels.items()])
        signal += fibo_text

    return signal


def fetch_ohlcv(symbol="BTC/USDT", timeframe="1h", limit=200):
    df = pd.DataFrame(exchange.fetch_ohlcv(symbol, timeframe, limit=limit),
                      columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')

    # محاسبه اندیکاتورها
    df = ichimoku(df)
    df = calculate_stoch_rsi(df)
    df = calculate_atr(df)
    return df


def main():
    symbol = "BTC/USDT"
    timeframe = "1h"

    while True:
        try:
            df = fetch_ohlcv(symbol, timeframe)
            signal = generate_signal(df)

            if signal:
                logging.info(signal)
                bot.send_message(chat_id=CHAT_ID, text=signal)

            time.sleep(60 * 5)  # هر ۵ دقیقه
        except Exception as e:
            logging.error(f"❌ خطا: {e}")
            time.sleep(60)


if __name__ == "__main__":
    main()
