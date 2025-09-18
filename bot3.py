import time
import os
import ccxt
import pandas as pd
import numpy as np
from telegram import Bot
from keep_alive import keep_alive

# ─── فعال کردن سرور کوچک برای جلوگیری از خوابیدن کانتینر ───
keep_alive()

# ─── اطلاعات ربات تلگرام ───
TELEGRAM_TOKEN = os.getenv("BOT_TOKEN")
CHAT_ID = os.getenv("CHAT_ID")
bot = Bot(token=TELEGRAM_TOKEN)

# ─── پیام تست برای اطمینان از اتصال ───
bot.send_message(chat_id=CHAT_ID, text="✅ ربات با موفقیت راه‌اندازی شد!")

# ─── صرافی کوکوین ───
exchange = ccxt.kucoin()
TOP_N = 85
TIMEFRAMES = ['5m', '15m', '1h']

# گرفتن نمادهای برتر
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

# گرفتن دیتای OHLCV
def get_ohlcv_df(symbol, timeframe):
    ohlcv = exchange.fetch_ohlcv(symbol, timeframe)
    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    return df

# محاسبه اندیکاتورها + فیبوناچی
def calculate_indicators(df):
    # EMA
    df['EMA9'] = df['close'].ewm(span=9, adjust=False).mean()
    df['EMA21'] = df['close'].ewm(span=21, adjust=False).mean()

    # Bollinger Bands
    df['BB_Mid'] = df['close'].rolling(20).mean()
    df['BB_Std'] = df['close'].rolling(20).std()
    df['BB_Upper'] = df['BB_Mid'] + 2 * df['BB_Std']
    df['BB_Lower'] = df['BB_Mid'] - 2 * df['BB_Std']

    # ATR
    df['H-L'] = df['high'] - df['low']
    df['H-PC'] = abs(df['high'] - df['close'].shift())
    df['L-PC'] = abs(df['low'] - df['close'].shift())
    df['TR'] = df[['H-L', 'H-PC', 'L-PC']].max(axis=1)
    df['ATR'] = df['TR'].rolling(14).mean()

    # Stochastic RSI
    rsi_up = df['close'].diff().apply(lambda x: max(x, 0)).rolling(14).mean()
    rsi_down = df['close'].diff().abs().rolling(14).mean()
    df['RSI'] = rsi_up / rsi_down
    df['StochRSI'] = (df['RSI'] - df['RSI'].rolling(14).min()) / (
        df['RSI'].rolling(14).max() - df['RSI'].rolling(14).min()
    )

    # Ichimoku Cloud
    df['Tenkan'] = (df['high'].rolling(9).max() + df['low'].rolling(9).min()) / 2
    df['Kijun'] = (df['high'].rolling(26).max() + df['low'].rolling(26).min()) / 2
    df['SenkouA'] = ((df['Tenkan'] + df['Kijun']) / 2).shift(26)
    df['SenkouB'] = ((df['high'].rolling(52).max() + df['low'].rolling(52).min()) / 2).shift(26)

    # Order Blocks ساده (۵ کندل اخیر)
    df['OB_High'] = df['high'].rolling(5).max()
    df['OB_Low'] = df['low'].rolling(5).min()

    # حمایت/مقاومت ساده با Swing High/Low
    df['SwingHigh'] = df['high'][df['high'] == df['high'].rolling(5, center=True).max()]
    df['SwingLow'] = df['low'][df['low'] == df['low'].rolling(5, center=True).min()]

    # --- سطوح فیبوناچی اصلاحی ---
    if len(df) >= 50:
        recent_high = df['high'].iloc[-50:].max()
        recent_low = df['low'].iloc[-50:].min()
        diff = recent_high - recent_low
        df['Fib_0'] = recent_high
        df['Fib_236'] = recent_high - 0.236 * diff
        df['Fib_382'] = recent_high - 0.382 * diff
        df['Fib_5'] = recent_high - 0.5 * diff
        df['Fib_618'] = recent_high - 0.618 * diff
        df['Fib_786'] = recent_high - 0.786 * diff
        df['Fib_1'] = recent_low
    return df

# الگوهای شمعی
def detect_candlestick_patterns(df):
    patterns = []
    open_, close, high, low = df['open'].iloc[-1], df['close'].iloc[-1], df['high'].iloc[-1], df['low'].iloc[-1]
    prev_open, prev_close = df['open'].iloc[-2], df['close'].iloc[-2]

    if prev_close < prev_open and close > open_ and close > prev_open and open_ < prev_close:
        patterns.append('Bullish Engulfing')
    if prev_close > prev_open and close < open_ and open_ > prev_close and close < prev_open:
        patterns.append('Bearish Engulfing')
    if (close - low) > 2 * (open_ - low):
        patterns.append('Hammer')
    if (high - close) > 2 * (high - open_):
        patterns.append('Hanging Man')
    return patterns

# ستاپ‌های ساده
def detect_setups(df):
    setups = []
    if df['close'].iloc[-1] > df['close'][-21:-1].max() * 1.01:
        setups.append('Breakout Up')
    elif df['close'].iloc[-1] < df['close'][-21:-1].min() * 0.99:
        setups.append('Breakout Down')

    if df['close'].iloc[-1] > df['EMA21'].iloc[-1] and df['close'].iloc[-2] < df['EMA21'].iloc[-2]:
        setups.append('Pullback Up')
    elif df['close'].iloc[-1] < df['EMA21'].iloc[-1] and df['close'].iloc[-2] > df['EMA21'].iloc[-2]:
        setups.append('Pullback Down')

    if len(df) >= 4:
        if df['close'].iloc[-1] < df['close'].iloc[-3] and df['close'].iloc[-3] == df['close'].iloc[-2]:
            setups.append('Double Top')
        elif df['close'].iloc[-1] > df['close'].iloc[-3] and df['close'].iloc[-3] == df['close'].iloc[-2]:
            setups.append('Double Bottom')
    return setups

# بررسی سیگنال
def check_signal(df, symbol, change):
    price = df['close'].iloc[-1]
    trend = 'neutral'
    if price > df['EMA21'].iloc[-1]:
        trend = 'bullish'
    elif price < df['EMA21'].iloc[-1]:
        trend = 'bearish'

    # شرط حجم: کندل آخر > 1.5 × میانگین حجم 20 کندل قبلی
    if len(df) < 21:
        return None
    if df['volume'].iloc[-1] <= 1.5 * df['volume'].iloc[-21:-1].mean():
        return None

    patterns = detect_candlestick_patterns(df)
    setups = detect_setups(df)

    atr_check = df['ATR'].iloc[-1] > df['ATR'].rolling(14).mean().iloc[-1]
    stoch_check = df['StochRSI'].iloc[-1] > 0.8 if trend == 'bearish' else df['StochRSI'].iloc[-1] < 0.2
    ichimoku_check = (
        price > df['SenkouA'].iloc[-1] and price > df['SenkouB'].iloc[-1]
        if trend == 'bullish'
        else price < df['SenkouA'].iloc[-1] and price < df['SenkouB'].iloc[-1]
    )

    fib_check = True
    if 'Fib_618' in df.columns:
        # اگر قیمت نزدیک یکی از سطوح مهم فیبوناچی بود
        fib_levels = [df['Fib_236'].iloc[-1], df['Fib_382'].iloc[-1], df['Fib_5'].iloc[-1], df['Fib_618'].iloc[-1], df['Fib_786'].iloc[-1]]
        fib_check = any(abs(price - lvl) / price < 0.003 for lvl in fib_levels)

    if patterns and setups and atr_check and stoch_check and ichimoku_check and fib_check:
        signal_type = 'LONG' if trend == 'bullish' else 'SHORT'
        entry = price
        tp = price * 1.01 if signal_type == 'LONG' else price * 0.99
        stop = price * 0.995 if signal_type == 'LONG' else price * 1.005
        return {
            'entry': entry,
            'tp': tp,
            'stop': stop,
            'type': signal_type
        }
    return None

# حلقه اصلی
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
                    df = get_ohlcv_df(symbol, tf)
                    df = calculate_indicators(df)
                    signal = check_signal(df, symbol, symbol_data['change'])
                    if signal:
                        tf_signals.append(signal)

                if tf_signals:
                    longs = [s for s in tf_signals if s['type'] == 'LONG']
                    shorts = [s for s in tf_signals if s['type'] == 'SHORT']
                    if len(longs) >= 2:
                        alerts.append((symbol, longs[0]))
                    elif len(shorts) >= 2:
                        alerts.append((symbol, shorts[0]))

            if alerts:
                msg = "🚨 Multi-Coin Alert 🚨\n"
                for symbol, s in alerts:
                    msg += f"{symbol} → {s['type']} | Entry: {s['entry']:.4f} | TP: {s['tp']:.4f} | Stop: {s['stop']:.4f}\n"
                try:
                    bot.send_message(chat_id=CHAT_ID, text=msg)
                except Exception as e:
                    print(f"[Telegram Error] {e}")

            print("⏳ صبر برای ۵ دقیقه بعدی ...\n")
            time.sleep(300)
        except Exception as e:
            print(f"⚠️ خطا: {e}")
            time.sleep(30)

if __name__ == "__main__":
    main()
