import time
import os
import ccxt
import pandas as pd
import numpy as np
from telegram import Bot
from keep_alive import keep_alive
from datetime import datetime

# ─── Keep-alive server
keep_alive()

# ─── Telegram bot setup
TELEGRAM_TOKEN = os.getenv("BOT_TOKEN")
CHAT_ID = os.getenv("CHAT_ID")
bot = Bot(token=TELEGRAM_TOKEN)

bot.send_message(chat_id=CHAT_ID, text="✅ Bot started successfully!")

exchange = ccxt.kucoin()

TOP_N = 80
TIMEFRAMES = ['5m','15m','30m','1h','4h']
last_alerts = {}  # To avoid repeated signals
SIGNAL_INTERVAL = 5 * 60  # 5 minutes

# ─── Risk management
ACCOUNT_BALANCE = 1000   # Example balance ($)
RISK_PER_TRADE = 0.01    # 1% risk per trade

def calculate_position_size(entry, stop):
    risk_amount = ACCOUNT_BALANCE * RISK_PER_TRADE
    risk_per_unit = abs(entry - stop)
    if risk_per_unit == 0:
        return 0
    position_size = risk_amount / risk_per_unit
    return round(position_size, 3)

# ─── Top symbols
def get_top_symbols():
    tickers = exchange.fetch_tickers()
    symbols = []
    for symbol, data in tickers.items():
        if symbol.endswith('/USDT'):
            symbols.append({
                'symbol': symbol,
                'volume': data['quoteVolume'],
                'change': data['percentage']
            })
    symbols.sort(key=lambda x: x['volume'], reverse=True)
    return symbols[:TOP_N]

# ─── OHLCV data
def get_ohlcv_df(symbol, timeframe):
    ohlcv = exchange.fetch_ohlcv(symbol, timeframe)
    df = pd.DataFrame(ohlcv, columns=['timestamp','open','high','low','close','volume'])
    return df

# ─── Indicators
def calculate_indicators(df):
    df['EMA9'] = df['close'].ewm(span=9, adjust=False).mean()
    df['EMA21'] = df['close'].ewm(span=21, adjust=False).mean()
    df['ATR'] = df['high'].combine(df['low'], max) - df['low'].combine(df['close'].shift(), min)
    # RSI
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / (loss + 1e-9)
    df['RSI'] = 100 - (100 / (1 + rs))
    return df

# ─── Candlestick patterns
def detect_candlestick_patterns(df):
    patterns = []
    open_, close = df['open'].iloc[-1], df['close'].iloc[-1]
    prev_open, prev_close = df['open'].iloc[-2], df['close'].iloc[-2]
    if prev_close < prev_open and close > open_ and close > prev_open and open_ < prev_close:
        patterns.append('Bullish Engulfing')
    if prev_close > prev_open and close < open_ and open_ > prev_close and close < prev_open:
        patterns.append('Bearish Engulfing')
    return patterns

# ─── Signal check
def check_signal(df, symbol, change):
    price = df['close'].iloc[-1]
    trend = 'neutral'
    if price > df['EMA21'].iloc[-1]:
        trend = 'bullish'
    elif price < df['EMA21'].iloc[-1]:
        trend = 'bearish'

    patterns = detect_candlestick_patterns(df)
    stars = []
    if df['volume'].iloc[-1] > df['volume'].rolling(20).mean().iloc[-1] * 1.5: stars.append('🔹')
    if df['RSI'].iloc[-1] < 30 or df['RSI'].iloc[-1] > 70: stars.append('🔹')
    if patterns: stars.append('🔹')

    signal_type = None
    entry = tp = stop = size = None

    if change >= 1 and trend == 'bullish' and len(stars) >= 2:
        signal_type = 'LONG'
        entry = price
        stop = price * 0.995
        tp = price * 1.01
    elif change <= -1 and trend == 'bearish' and len(stars) >= 2:
        signal_type = 'SHORT'
        entry = price
        stop = price * 1.005
        tp = price * 0.99

    if signal_type and entry and stop:
        size = calculate_position_size(entry, stop)

    # avoid repeated
    prev = last_alerts.get(symbol)
    if prev and prev["type"] == signal_type:
        return None

    if signal_type:
        last_alerts[symbol] = {"type": signal_type, "time": time.time()}

    return {
        'entry': entry,
        'tp': tp,
        'stop': stop,
        'type': signal_type,
        'patterns': patterns,
        'stars': stars,
        'size': size,
        'signal_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }

# ─── Main loop

def main():
    print("🚀 Multi-Coin & Multi-Timeframe Bot started")
    while True:
        try:
            top_symbols = get_top_symbols()

            for symbol_data in top_symbols:
                symbol = symbol_data['symbol']
                for tf in TIMEFRAMES:
                    try:
                        df = get_ohlcv_df(symbol, tf)
                        df = calculate_indicators(df)
                        signal = check_signal(df, symbol, symbol_data['change'])
                        if not signal: continue
                    except Exception as e:
                        print(f"[ERROR] {symbol} | TF: {tf} | {e}")
                        continue

                    msg = (
                        f"{signal['type']} Signal | {symbol} | TF: {tf}\n"
                        f"Entry: {signal['entry']:.4f}\n"
                        f"TP: {signal['tp']:.4f}\n"
                        f"Stop: {signal['stop']:.4f}\n"
                        f"Size: {signal['size']}\n"
                        f"Conditions: {''.join(signal['stars'])}\n"
                        f"Patterns: {signal['patterns']}\n"
                        f"Signal Time: {signal['signal_time']}"
                    )
                    bot.send_message(chat_id=CHAT_ID, text=msg)

            time.sleep(300)

        except Exception as e:
            print(f"⚠️ Error: {e}")
            time.sleep(30)

if __name__ == "__main__":
    main()
