import time
import os
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

bot.send_message(chat_id=CHAT_ID, text="✅ ربات با موفقیت راه‌اندازی شد!")

# ─── صرافی کوکوین ───
exchange = ccxt.kucoin()
TOP_N = 85
TIMEFRAMES = ['5m', '15m', '1h']


# ─── دریافت لیست برتر از لحاظ حجم ───
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


# ─── دریافت کندل‌های تاریخی ───
def get_ohlcv_df(symbol, timeframe):
    ohlcv = exchange.fetch_ohlcv(symbol, timeframe)
    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    return df.dropna()


# ─── محاسبه اندیکاتورها ───
def calculate_indicators(df):
    df['EMA9'] = df['close'].ewm(span=9, adjust=False).mean()
    df['EMA21'] = df['close'].ewm(span=21, adjust=False).mean()

    df['BB_Mid'] = df['close'].rolling(20).mean()
    df['BB_Std'] = df['close'].rolling(20).std()
    df['BB_Upper'] = df['BB_Mid'] + 2 * df['BB_Std']
    df['BB_Lower'] = df['BB_Mid'] - 2 * df['BB_Std']

    df['H-L'] = df['high'] - df['low']
    df['H-PC'] = abs(df['high'] - df['close'].shift())
    df['L-PC'] = abs(df['low'] - df['close'].shift())
    df['TR'] = df[['H-L', 'H-PC', 'L-PC']].max(axis=1)
    df['ATR'] = df['TR'].rolling(14).mean()

    rsi_up = df['close'].diff().apply(lambda x: max(x, 0)).rolling(14).mean()
    rsi_down = df['close'].diff().abs().rolling(14).mean()
    df['RSI'] = rsi_up / rsi_down
    df['StochRSI'] = (df['RSI'] - df['RSI'].rolling(14).min()) / (
        df['RSI'].rolling(14).max() - df['RSI'].rolling(14).min()
    )

    df['Tenkan'] = (df['high'].rolling(9).max() + df['low'].rolling(9).min()) / 2
    df['Kijun'] = (df['high'].rolling(26).max() + df['low'].rolling(26).min()) / 2
    df['SenkouA'] = ((df['Tenkan'] + df['Kijun']) / 2).shift(26)
    df['SenkouB'] = ((df['high'].rolling(52).max() + df['low'].rolling(52).min()) / 2).shift(26)

    recent = df.tail(50)
    high_price = recent['high'].max()
    low_price = recent['low'].min()
    diff = high_price - low_price
    for level, name in zip([0.236, 0.382, 0.5, 0.618, 0.786], ['Fib23', 'Fib38', 'Fib50', 'Fib61', 'Fib78']):
        df[name] = high_price - diff * level

    df['OB_High'] = df['high'].rolling(5).max()
    df['OB_Low'] = df['low'].rolling(5).min()

    df['SwingHigh'] = df['high'][df['high'] == df['high'].rolling(5, center=True).max()]
    df['SwingLow'] = df['low'][df['low'] == df['low'].rolling(5, center=True).min()]
    return df


# ─── تشخیص واگرایی RSI ───
def detect_rsi_divergence(df):
    if len(df) < 10:
        return None
    rsi = df['RSI']
    close = df['close']

    last_rsi_highs = rsi.tail(5).nlargest(2)
    last_price_highs = close.loc[last_rsi_highs.index]

    last_rsi_lows = rsi.tail(5).nsmallest(2)
    last_price_lows = close.loc[last_rsi_lows.index]

    bullish = last_price_lows.iloc[-1] < last_price_lows.iloc[0] and last_rsi_lows.iloc[-1] > last_rsi_lows.iloc[0]
    bearish = last_price_highs.iloc[-1] > last_price_highs.iloc[0] and last_rsi_highs.iloc[-1] < last_rsi_highs.iloc[0]

    if bullish:
        return 'bullish'
    elif bearish:
        return 'bearish'
    return None


# ─── تشخیص الگوهای کندلی ───
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
# ─── تشخیص الگوهای پرچم و مثلث ───
def detect_pattern_flags(df):
    flag_patterns = []
    if len(df) < 10:
        return flag_patterns
    recent = df.tail(10)
    highs = recent['high']
    lows = recent['low']
    closes = recent['close']

    if closes.iloc[-1] > closes.iloc[0] and highs.max() - lows.min() < 0.03 * closes.iloc[0]:
        flag_patterns.append('Bullish Flag')
    if closes.iloc[-1] < closes.iloc[0] and highs.max() - lows.min() < 0.03 * closes.iloc[0]:
        flag_patterns.append('Bearish Flag')
    if (highs.max() - highs.min()) < 0.03 * closes.iloc[0] and (lows.max() - lows.min()) < 0.03 * closes.iloc[0]:
        flag_patterns.append('Triangle / Wedge')
    return flag_patterns


# ─── تشخیص ستاپ‌ها ───
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


# ─── بررسی و ساخت سیگنال ───
def check_signal(df, symbol, change):
    if len(df) < 30:
        return None

    price = df['close'].iloc[-1]
    trend = 'neutral'
    if price > df['EMA21'].iloc[-1]:
        trend = 'bullish'
    elif price < df['EMA21'].iloc[-1]:
        trend = 'bearish'

    # شرط حجم
    if df['volume'].iloc[-1] <= 1.5 * df['volume'].iloc[-21:-1].mean():
        return None

    patterns = detect_candlestick_patterns(df)
    setups = detect_setups(df)
    divergence = detect_rsi_divergence(df)
    flag_patterns = detect_pattern_flags(df)

    atr_now = df['ATR'].iloc[-1]
    atr_avg = df['ATR'].rolling(14).mean().iloc[-1]
    atr_check = atr_now > atr_avg

    # StochRSI
    if trend == 'bullish':
        stoch_check = df['StochRSI'].iloc[-1] < 0.2
    else:
        stoch_check = df['StochRSI'].iloc[-1] > 0.8

    # ایچیموکو
    if trend == 'bullish':
        ichi_check = price > df['SenkouA'].iloc[-1] and price > df['SenkouB'].iloc[-1]
    else:
        ichi_check = price < df['SenkouA'].iloc[-1] and price < df['SenkouB'].iloc[-1]

    # شمارش شرط‌ها
    total_conditions = 6
    passed_conditions = 0
    if patterns:
        passed_conditions += 1
    if setups:
        passed_conditions += 1
    if atr_check:
        passed_conditions += 1
    if stoch_check:
        passed_conditions += 1
    if ichi_check:
        passed_conditions += 1
    if (trend == 'bullish' and divergence != 'bearish') or \
       (trend == 'bearish' and divergence != 'bullish'):
        passed_conditions += 1

    if passed_conditions >= 4:  # حداقل تعداد شرط لازم
        signal_type = 'LONG' if trend == 'bullish' else 'SHORT'
        atr_mult_stop = 1.5
        atr_mult_tp = 2.5

        if signal_type == 'LONG':
            stop = price - atr_mult_stop * atr_now
            tp = price + atr_mult_tp * atr_now
        else:
            stop = price + atr_mult_stop * atr_now
            tp = price - atr_mult_tp * atr_now

        rr = abs(tp - price) / abs(price - stop)
        if rr < 1.5:
            return None

        return {
            'entry': price,
            'tp': tp,
            'stop': stop,
            'type': signal_type,
            'patterns': flag_patterns,
            'passed': passed_conditions,
            'total': total_conditions
        }
    return None


# ─── حلقه اصلی ───
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
                    stars = "⭐" * s['passed']
                    msg += (
                        f"{symbol} → {s['type']}\n"
                        f"Entry: {s['entry']:.4f}\n"
                        f"TP: {s['tp']:.4f}\n"
                        f"Stop: {s['stop']:.4f}\n"
                        f"{stars} ({s['passed']}/{s['total']})\n"
                    )
                    if s['patterns']:
                        msg += f"🔹 الگوها: {', '.join(s['patterns'])}\n"
                    msg += "\n"
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
    
      # ─── پیام تستی ستاره‌ها ───
    test_passed = 3
    test_total = 6
    test_msg = (
        f"🚨 Test Multi-Coin Alert 🚨\n"
        f"SAMPLE/USDT → LONG\n"
        f"Entry: 650.0\n"
        f"TP: 660.0\n"
        f"Stop: 645.0\n"
        f"{'⭐'*test_passed} ({test_passed}/{test_total})"
    )
    print(test_msg)  # نمایش در لاگ
    bot.send_message(chat_id=CHAT_ID, text=test_msg)

    main()
