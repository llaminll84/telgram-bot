import time
import ccxt
import pandas as pd
import numpy as np
from telegram import Bot

# ─── اطلاعات ربات تلگرام ───
import os
from telegram import Bot

TELEGRAM_TOKEN = os.getenv("BOT_TOKEN")
CHAT_ID = os.getenv("CHAT_ID")
bot = Bot(token=TELEGRAM_TOKEN)


# صرافی کوکوین
exchange = ccxt.kucoin()

TOP_N = 50
TIMEFRAMES = ['5m', '15m', '1h']

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

def get_ohlcv_df(symbol, timeframe):
    ohlcv = exchange.fetch_ohlcv(symbol, timeframe)
    df = pd.DataFrame(ohlcv, columns=['timestamp','open','high','low','close','volume'])
    return df

def calculate_indicators(df):
    df['EMA9'] = df['close'].ewm(span=9, adjust=False).mean()
    df['EMA21'] = df['close'].ewm(span=21, adjust=False).mean()
    df['EMA12'] = df['close'].ewm(span=12, adjust=False).mean()
    df['EMA26'] = df['close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = df['EMA12'] - df['EMA26']
    df['Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['BB_Mid'] = df['close'].rolling(window=20).mean()
    df['BB_Std'] = df['close'].rolling(window=20).std()
    df['BB_Upper'] = df['BB_Mid'] + 2*df['BB_Std']
    df['BB_Lower'] = df['BB_Mid'] - 2*df['BB_Std']
    df['Tenkan'] = (df['high'].rolling(9).max() + df['low'].rolling(9).min()) / 2
    df['Kijun'] = (df['high'].rolling(26).max() + df['low'].rolling(26).min()) / 2
    df['SenkouA'] = ((df['Tenkan'] + df['Kijun']) / 2).shift(26)
    df['SenkouB'] = ((df['high'].rolling(52).max() + df['low'].rolling(52).min())/2).shift(26)
    return df

def detect_candlestick_patterns(df):
    patterns = []
    open_, close, high, low = df['open'].iloc[-1], df['close'].iloc[-1], df['high'].iloc[-1], df['low'].iloc[-1]
    prev_open, prev_close = df['open'].iloc[-2], df['close'].iloc[-2]
    
    if prev_close < prev_open and close > open_ and close > prev_open and open_ < prev_close:
        patterns.append('Bullish Engulfing')
    if prev_close > prev_open and close < open_ and open_ > prev_close and close < prev_open:
        patterns.append('Bearish Engulfing')
    if (close - low) > 2*(open_ - low):
        patterns.append('Hammer')
    if (high - close) > 2*(high - open_):
        patterns.append('Hanging Man')
    if abs(close - open_) / (high - low + 1e-9) < 0.1:
        patterns.append('Doji')
    return patterns

def detect_order_block(df):
    recent = df[-5:]
    blocks = []
    threshold = df['close'].std() * 1.5
    for i in range(len(recent)-1):
        if abs(recent['close'].iloc[i] - recent['open'].iloc[i]) > threshold:
            blocks.append((recent['low'].iloc[i], recent['high'].iloc[i]))
    return blocks

def check_signal(df, symbol, change):
    price = df['close'].iloc[-1]
    if price > df['SenkouA'].iloc[-1] and price > df['SenkouB'].iloc[-1]:
        trend = 'bullish'
    elif price < df['SenkouA'].iloc[-1] and price < df['SenkouB'].iloc[-1]:
        trend = 'bearish'
    else:
        trend = 'neutral'
    
    patterns = detect_candlestick_patterns(df)
    order_blocks = detect_order_block(df)
    volume_check = df['volume'].iloc[-1] > df['volume'].rolling(20).mean().iloc[-1] * 1.5
    
    if change >= 1 and trend=='bullish' and any(p in patterns for p in ['Bullish Engulfing','Hammer','Morning Star']) and volume_check:
        entry = price
        tp = price * 1.01
        stop = price * 0.995
        signal_type = 'LONG'
    elif change <= -1 and trend=='bearish' and any(p in patterns for p in ['Bearish Engulfing','Hanging Man','Evening Star']) and volume_check:
        entry = price
        tp = price * 0.99
        stop = price * 1.005
        signal_type = 'SHORT'
    else:
        return None
    
    return {
        'entry': entry,
        'tp': tp,
        'stop': stop,
        'type': signal_type,
        'patterns': patterns,
        'order_blocks': order_blocks
    }

def main():
    print("🚀 ربات Multi-Coin & Multi-Timeframe با آلارم خودکار شروع شد")
    while True:
        try:
            top_symbols = get_top_symbols()
            alerts = []
            for symbol_data in top_symbols:
                symbol = symbol_data['symbol']
                signal_count = 0
                tf_signals = []
                for tf in TIMEFRAMES:
                    df = get_ohlcv_df(symbol, tf)
                    df = calculate_indicators(df)
                    signal = check_signal(df, symbol, symbol_data['change'])
                    print(f"[CMD] {symbol} | TF: {tf} | Close: {df['close'].iloc[-1]:.4f} | Change: {symbol_data['change']:.2f}% | Patterns: {signal['patterns'] if signal else 'None'} | Order Blocks: {signal['order_blocks'] if signal else 'None'}")
                    if signal:
                        signal_count += 1
                        tf_signals.append(signal)
                # اگر حداقل ۲ تایم فریم همخوانی داشت → اضافه به آلارم
                if signal_count >= 1:
                    alerts.append((symbol, tf_signals))
            # ارسال آلارم تلگرام برای همه ارزها همزمان
            if alerts:
                msg = "🚨 Multi-Coin Alert 🚨\n"
                for symbol, sigs in alerts:
                    # میانگین قیمت‌ها برای تلگرام
                    entry = np.mean([s['entry'] for s in sigs])
                    tp = np.mean([s['tp'] for s in sigs])
                    stop = np.mean([s['stop'] for s in sigs])
                    msg += f"{symbol} → {sigs[0]['type']} | Entry: {entry:.4f} | TP: {tp:.4f} | Stop: {stop:.4f}\n"
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
