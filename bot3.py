import time
import os
import ccxt
import pandas as pd
import numpy as np
import datetime
from telegram import Bot
from keep_alive import keep_alive  # سرور کوچک برای جلوگیری از خوابیدن کانتینر

# ─── سرور کوچک
keep_alive()  

# ─── اطلاعات ربات تلگرام
TELEGRAM_TOKEN = os.getenv("BOT_TOKEN")
CHAT_ID = os.getenv("CHAT_ID")
bot = Bot(token=TELEGRAM_TOKEN)

bot.send_message(chat_id=CHAT_ID, text="✅ ربات با موفقیت راه‌اندازی شد!")

exchange = ccxt.kucoin()

TOP_N = 80
TIMEFRAMES = ['5m','15m','30m','1h','4h']
last_signal_time = {}
last_alerts = {}  # برای جلوگیری از سیگنال تکراری

SIGNAL_INTERVAL = 5 * 60  # 5 دقیقه فاصله بین سیگنال‌ها

# ─── مدیریت ریسک ───
ACCOUNT_BALANCE = 1000   # موجودی فرضی (دلار)
RISK_PER_TRADE = 0.01    # 1 درصد ریسک در هر معامله

def calculate_position_size(entry, stop):
    risk_amount = ACCOUNT_BALANCE * RISK_PER_TRADE
    risk_per_unit = abs(entry - stop)
    if risk_per_unit == 0:
        return 0
    position_size = risk_amount / risk_per_unit
    return round(position_size, 3)

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

# ─── اندیکاتورها
def calculate_indicators(df):
    if len(df) < 60:   # جلوگیری از خطا در دیتای کم
        return df

    df['EMA9'] = df['close'].ewm(span=9, adjust=False).mean()
    df['EMA21'] = df['close'].ewm(span=21, adjust=False).mean()
    df['MACD'] = df['close'].ewm(span=12, adjust=False).mean() - df['close'].ewm(span=26, adjust=False).mean()
    df['Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['BB_Mid'] = df['close'].rolling(20).mean()
    df['BB_Std'] = df['close'].rolling(20).std()
    df['BB_Upper'] = df['BB_Mid'] + 2 * df['BB_Std']
    df['BB_Lower'] = df['BB_Mid'] - 2 * df['BB_Std']

    # ATR
    df['TR'] = np.maximum.reduce([
        df['high'] - df['low'],
        abs(df['high'] - df['close'].shift()),
        abs(df['low'] - df['close'].shift())
    ])
    df['ATR14'] = df['TR'].rolling(14).mean()
    df['ATR'] = df['ATR14']

    # StochRSI
    df['StochRSI'] = (df['close'] - df['close'].rolling(14).min()) / (df['close'].rolling(14).max() - df['close'].rolling(14).min())

    # Ichimoku
    df['Tenkan'] = (df['high'].rolling(9).max() + df['low'].rolling(9).min()) / 2
    df['Kijun'] = (df['high'].rolling(26).max() + df['low'].rolling(26).min()) / 2
    df['SenkouA'] = ((df['Tenkan'] + df['Kijun']) / 2).shift(26)
    df['SenkouB'] = ((df['high'].rolling(52).max() + df['low'].rolling(52).min()) / 2).shift(26)

    # RSI
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / (loss + 1e-9)
    df['RSI'] = 100 - (100 / (1 + rs))

    # ADX
    df['+DM'] = np.where((df['high'].diff() > df['low'].diff()) & (df['high'].diff() > 0),
                         df['high'].diff(), 0)
    df['-DM'] = np.where((df['low'].diff() > df['high'].diff()) & (df['low'].diff() > 0),
                         df['low'].diff(), 0)
    df['+DI'] = 100 * (df['+DM'].ewm(alpha=1/14).mean() / df['ATR14'])
    df['-DI'] = 100 * (df['-DM'].ewm(alpha=1/14).mean() / df['ATR14'])
    df['DX'] = (abs(df['+DI'] - df['-DI']) / (df['+DI'] + df['-DI'] + 1e-9)) * 100
    df['ADX'] = df['DX'].ewm(alpha=1/14).mean()

    # SuperTrend
    factor = 3
    hl2 = (df['high'] + df['low']) / 2
    df['UpperBand'] = hl2 + (factor * df['ATR14'])
    df['LowerBand'] = hl2 - (factor * df['ATR14'])
    df['SuperTrend'] = np.where(df['close'] > df['UpperBand'], 1,
                                np.where(df['close'] < df['LowerBand'], -1, 0))
    return df

# ─── شناسایی کندل‌ها
def detect_candlestick_patterns(df):
    if len(df) < 3:
        return []
    patterns = []
    open_, close, high, low = df['open'].iloc[-1], df['close'].iloc[-1], df['high'].iloc[-1], df['low'].iloc[-1]
    prev_open, prev_close = df['open'].iloc[-2], df['close'].iloc[-2]
    p2_open, p2_close = df['open'].iloc[-3], df['close'].iloc[-3]

    if prev_close < prev_open and close > open_ and close > prev_open and open_ < prev_close:
        patterns.append('Bullish Engulfing')
    if prev_close > prev_open and close < open_ and open_ > prev_close and close < prev_open:
        patterns.append('Bearish Engulfing')
    if (close - low) > 2 * (open_ - low):
        patterns.append('Hammer')
    if (high - close) > 2 * (high - open_):
        patterns.append('Hanging Man')
    if abs(close - open_) / (high - low + 1e-9) < 0.1:
        patterns.append('Doji')

    return patterns

# ─── بررسی سیگنال و شروط
def check_signal(df, symbol, change):
    if len(df) < 60:
        return None

    price = df['close'].iloc[-1]
    trend = 'neutral'
    if price > df['SenkouA'].iloc[-1] and price > df['SenkouB'].iloc[-1]:
        trend = 'bullish'
    elif price < df['SenkouA'].iloc[-1] and price < df['SenkouB'].iloc[-1]:
        trend = 'bearish'

    patterns = detect_candlestick_patterns(df)

    # شروط
    stars = []
    if df['volume'].iloc[-1] > df['volume'].rolling(20).mean().iloc[-1] * 1.5: stars.append('🔹')
    if df['ATR'].iloc[-1] > df['ATR'].rolling(14).mean().iloc[-1]: stars.append('🔹')
    if df['ADX'].iloc[-1] > 25: stars.append("🔹")
    if patterns: stars.append('🔹')

    signal_type = None
    entry = tp = stop = size = None
    atr = df['ATR14'].iloc[-1]

    if (change >= 1 and trend == 'bullish' and len(stars) >= 3 
        and df['EMA9'].iloc[-1] > df['EMA21'].iloc[-1] 
        and df['RSI'].iloc[-1] > 50):
        signal_type = 'LONG'
        entry = price
        stop = price - 1.5 * atr
        tp = price + 2 * atr

    elif (change <= -1 and trend == 'bearish' and len(stars) >= 3 
          and df['EMA9'].iloc[-1] < df['EMA21'].iloc[-1] 
          and df['RSI'].iloc[-1] < 50):
        signal_type = 'SHORT'
        entry = price
        stop = price + 1.5 * atr
        tp = price - 2 * atr

    if signal_type and entry and stop:
        size = calculate_position_size(entry, stop)

    # جلوگیری از تکرار
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
        'size': size
    }

# ─── تابع اصلی ربات
def main():
    print("🚀 ربات Multi-Coin & Multi-Timeframe با آلارم خودکار شروع شد")
    while True:
        try:
            top_symbols = get_top_symbols()
            alerts = []

            for symbol_data in top_symbols:
                symbol = symbol_data['symbol']
                tf_signals = []

                for tf in TIMEFRAMES:
                    try:
                        df = get_ohlcv_df(symbol, tf)
                        df = calculate_indicators(df)
                        signal = check_signal(df, symbol, symbol_data['change'])
                        if not signal:
                            continue
                    except Exception as e:
                        print(f"[ERROR] {symbol} | TF: {tf} | {e}")
                        continue

                    # لاگ کامل همه ارزها
                    print(f"[LOG] {symbol} | TF: {tf} | Close: {df['close'].iloc[-1]:.4f} | "
                          f"Change: {symbol_data['change']:.2f}% | Signal: {signal['type']} | Stars: {''.join(signal['stars'])}")

                    tf_signals.append(signal)

                # سیگنال معتبر فقط وقتی که حداقل ۲ تایم‌فریم همسو باشن
                if len(tf_signals) >= 2 and all(s['type'] == tf_signals[0]['type'] for s in tf_signals):
                    alerts.append((symbol, tf_signals))
                    last_signal_time[symbol] = time.time()

            # ارسال پیام تلگرام
            if alerts:
                for symbol, sigs in alerts:
                    for s in sigs:
                        if not s['type']:
                            continue

                        color_emoji = "🟢" if s['type'] == "LONG" else "🔴"
                        now_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                        msg = (f"🚨 Multi-Coin Alert 🚨\n"
                               f"{color_emoji} {symbol}\n"
                               f"Type: {s['type']}\n"
                               f"Entry: {s['entry']:.4f}\n"
                               f"TP: {s['tp']:.4f}\n"
                               f"Stop: {s['stop']:.4f}\n"
                               f"Size: {s['size']}\n"
                               f"Patterns: {s['patterns']}\n"
                               f"Conditions: {''.join(s['stars'])}\n"
                               f"🕒 Time: {now_time}")

                        bot.send_message(chat_id=CHAT_ID, text=msg)

            print("⏳ صبر برای ۵ دقیقه بعدی ...\n")
            time.sleep(300)

        except Exception as e:
            print(f"⚠️ خطا: {e}")
            time.sleep(30)

if __name__ == "__main__":
    main()
