import time
import os
import ccxt
import pandas as pd
import numpy as np
from telegram import Bot
from keep_alive import keep_alive

# ─── سرور کوچک
keep_alive()

# ─── تنظیمات تلگرام
TELEGRAM_TOKEN = os.getenv("BOT_TOKEN")
CHAT_ID = os.getenv("CHAT_ID")
bot = Bot(token=TELEGRAM_TOKEN)

# ─── صرافی
exchange = ccxt.kucoin()
TOP_N = 80
TIMEFRAMES = ['5m', '15m', '1h']
SIGNAL_INTERVAL = 2 * 60 * 60  # فاصله حداقل ۲ ساعت بین سیگنال‌ها

# ─── ذخیره آخرین زمان سیگنال برای هر ارز
last_signal_time = {}

# ─── ارسال پیام تلگرام
def send_telegram(msg):
    try:
        bot.send_message(chat_id=CHAT_ID, text=msg)
    except Exception as e:
        print(f"[Telegram Error] {e}")

# ─── گرفتن ۸۰ ارز برتر
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

# ─── گرفتن داده OHLCV
def get_ohlcv_df(symbol, timeframe):
    ohlcv = exchange.fetch_ohlcv(symbol, timeframe)
    df = pd.DataFrame(ohlcv, columns=['timestamp','open','high','low','close','volume'])
    return df

# ─── محاسبه اندیکاتورها
def calculate_indicators(df):
    df['EMA9'] = df['close'].ewm(span=9, adjust=False).mean()
    df['EMA21'] = df['close'].ewm(span=21, adjust=False).mean()
    df['MACD'] = df['close'].ewm(span=12, adjust=False).mean() - df['close'].ewm(span=26, adjust=False).mean()
    df['Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['BB_Mid'] = df['close'].rolling(20).mean()
    df['BB_Std'] = df['close'].rolling(20).std()
    df['BB_Upper'] = df['BB_Mid'] + 2 * df['BB_Std']
    df['BB_Lower'] = df['BB_Mid'] - 2 * df['BB_Std']
    df['ATR'] = df['high'].combine(df['low'], max) - df['low'].combine(df['close'].shift(), min)
    df['StochRSI'] = (df['close'] - df['close'].rolling(14).min()) / (df['close'].rolling(14).max() - df['close'].rolling(14).min())
    df['Tenkan'] = (df['high'].rolling(9).max() + df['low'].rolling(9).min()) / 2
    df['Kijun'] = (df['high'].rolling(26).max() + df['low'].rolling(26).min()) / 2
    df['SenkouA'] = ((df['Tenkan'] + df['Kijun'])/2).shift(26)
    df['SenkouB'] = ((df['high'].rolling(52).max() + df['low'].rolling(52).min())/2).shift(26)
    return df

# ─── شناسایی کندل‌ها
def detect_candlestick_patterns(df):
    patterns = []
    open_, close, high, low = df['open'].iloc[-1], df['close'].iloc[-1], df['high'].iloc[-1], df['low'].iloc[-1]
    prev_open, prev_close = df['open'].iloc[-2], df['close'].iloc[-2]

    if prev_close < prev_open and close > open_ and close > prev_open and open_ < prev_close:
        patterns.append('Bullish Engulfing')
    if prev_close > prev_open and close < open_ and open_ > prev_close and close < prev_open:
        patterns.append('Bearish Engulfing')
    if (close - low) > 2*(open_-low):
        patterns.append('Hammer')
    if (high-close) > 2*(high-open_):
        patterns.append('Hanging Man')
    if abs(close-open_)/(high-low+1e-9) < 0.1:
        patterns.append('Doji')
    return patterns

# ─── شناسایی Order Block
def detect_order_block(df):
    recent = df[-5:]
    blocks = []
    threshold = df['close'].std() * 1.5
    for i in range(len(recent)-1):
        if abs(recent['close'].iloc[i]-recent['open'].iloc[i]) > threshold:
            blocks.append((recent['low'].iloc[i], recent['high'].iloc[i]))
    return blocks

# ─── بررسی سیگنال با نمره‌دهی
def check_signal(df, symbol, change):
    price = df['close'].iloc[-1]
    score = 0
    conditions = []

    # روند
    trend = 'neutral'
    if price > df['SenkouA'].iloc[-1] and price > df['SenkouB'].iloc[-1]:
        trend = 'bullish'
    elif price < df['SenkouA'].iloc[-1] and price < df['SenkouB'].iloc[-1]:
        trend = 'bearish'

    # کندل
    patterns = detect_candlestick_patterns(df)
    if trend=='bullish' and any(p in patterns for p in ['Bullish Engulfing','Hammer','Morning Star']):
        score += 1
        conditions.append('*')
    if trend=='bearish' and any(p in patterns for p in ['Bearish Engulfing','Hanging Man','Evening Star']):
        score += 1
        conditions.append('*')

    # Order Block
    order_blocks = detect_order_block(df)
    if order_blocks:
        score += 1
        conditions.append('*')

    # حجم
    volume_check = df['volume'].iloc[-1] > df['volume'].rolling(20).mean().iloc[-1]*1.5
    if volume_check:
        score += 1
        conditions.append('*')

    # StochRSI
    stoch_rsi_check = df['StochRSI'].iloc[-1] > 0.8 if trend=='bearish' else df['StochRSI'].iloc[-1]<0.2
    if stoch_rsi_check:
        score += 1
        conditions.append('*')

    # ATR
    atr_check = df['ATR'].iloc[-1] > df['ATR'].rolling(14).mean().iloc[-1]
    if atr_check:
        score += 1
        conditions.append('*')

    # EMA+MACD
    if trend=='bullish' and df['close'].iloc[-1]>df['EMA21'].iloc[-1] and df['MACD'].iloc[-1]>df['Signal'].iloc[-1]:
        score += 1
        conditions.append('*')
    if trend=='bearish' and df['close'].iloc[-1]<df['EMA21'].iloc[-1] and df['MACD'].iloc[-1]<df['Signal'].iloc[-1]:
        score += 1
        conditions.append('*')

    # حد نصاب برای ارسال سیگنال
    if score >= 4:
        if trend=='bullish':
            entry = price
            tp = price * 1.01
            stop = price - df['ATR'].iloc[-1]
            signal_type = 'LONG'
        else:
            entry = price
            tp = price * 0.99
            stop = price + df['ATR'].iloc[-1]
            signal_type = 'SHORT'
        return {
            'entry': entry,
            'tp': tp,
            'stop': stop,
            'type': signal_type,
            'patterns': patterns,
            'order_blocks': order_blocks,
            'stars': ''.join(conditions)
        }
    return None

# ─── اجرای ربات
def main():
    print("🚀 ربات Multi-Coin & Multi-Timeframe با آلارم خودکار شروع شد")
    send_telegram("✅ ربات فعال شد و شروع به کار می‌کند. فاز گرم شدن ۵ دقیقه...")
    time.sleep(300)  # فاز گرم شدن ۵ دقیقه

    while True:
        try:
            top_symbols = get_top_symbols()
            alerts = []

            for symbol_data in top_symbols:
                symbol = symbol_data['symbol']
                # بررسی فاصله زمانی
                if symbol in last_signal_time and (time.time() - last_signal_time[symbol]) < SIGNAL_INTERVAL:
                    continue

                tf_signals = []
                for tf in TIMEFRAMES:
                    df = get_ohlcv_df(symbol, tf)
                    df = calculate_indicators(df)
                    signal = check_signal(df, symbol, symbol_data['change'])
                    if signal:
                        tf_signals.append(signal)

                # تایید حداقل دو تایم‌فریم
                if len(tf_signals) >= 2:
                    alerts.append((symbol, tf_signals))
                    last_signal_time[symbol] = time.time()

            # ارسال پیام تلگرام
            if alerts:
                msg = "🚨 Multi-Coin Alert 🚨\n"
                for symbol, sigs in alerts:
                    for s in sigs:
                        msg += (
                            f"{symbol}\n"
                            f"Type: {s['type']}\n"
                            f"Entry: {s['entry']:.4f}\n"
                            f"TP: {s['tp']:.4f}\n"
                            f"Stop: {s['stop']:.4f}\n"
                            f"Patterns: {s['patterns']}\n"
                            f"Order Blocks: {s['order_blocks']}\n"
                            f"Conditions: {s['stars']}\n\n"
                        )
                send_telegram(msg)

            print("⏳ صبر برای ۵ دقیقه بعدی ...\n")
            time.sleep(300)

        except Exception as e:
            print(f"⚠️ خطا: {e}")
            time.sleep(30)

if __name__ == "__main__":
    main()
