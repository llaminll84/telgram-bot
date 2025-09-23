import time
import os
import logging
import ccxt
import pandas as pd
import numpy as np
from telegram import Bot
from keep_alive import keep_alive
import requests

# ─── فعال کردن سرور کوچک ───
keep_alive()

# ─── اطلاعات ربات تلگرام ───
TELEGRAM_TOKEN = os.getenv("BOT_TOKEN")
CHAT_ID = os.getenv("CHAT_ID")
bot = Bot(token=TELEGRAM_TOKEN)
bot.send_message(chat_id=CHAT_ID, text="✅ ربات استارت شد و به درستی فعال است.")

# ─── تنظیمات لاگینگ ───
logging.basicConfig(level=logging.INFO)

# ─── اتصال به صرافی کوکوین ───
exchange = ccxt.kucoin({"enableRateLimit": True})

# ─── محاسبه ایچیموکو ───
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

# ─── محاسبه StochRSI ───
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

# ─── محاسبه ATR ───
def calculate_atr(df, period=14):
    df['H-L'] = df['high'] - df['low']
    df['H-C'] = abs(df['high'] - df['close'].shift())
    df['L-C'] = abs(df['low'] - df['close'].shift())
    df['TR'] = df[['H-L', 'H-C', 'L-C']].max(axis=1)
    df['ATR'] = df['TR'].rolling(window=period).mean()
    return df

# ─── تشخیص الگوهای کندلی ───
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
# ─── تشخیص اوردر بلاک ───
def detect_order_block(df, period=20):
    order_blocks = []
    recent = df.tail(period)

    if recent['close'].iloc[-1] > recent['open'].iloc[-1] and recent['close'].max() > recent['close'].iloc[-2]:
        order_blocks.append("Bullish Order Block")
    if recent['close'].iloc[-1] < recent['open'].iloc[-1] and recent['close'].min() < recent['close'].iloc[-2]:
        order_blocks.append("Bearish Order Block")
    return order_blocks

# ─── بررسی قدرت روند ───
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

# ─── محاسبه سطوح فیبوناچی ───
def calculate_fibonacci(df, period=100):
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

# ─── محاسبه سایز پوزیشن بر اساس ATR و ریسک درصدی ───
def position_size(entry, stop, risk=0.01, capital=1000):
    risk_amount = capital * risk
    trade_risk = abs(entry - stop)
    size = risk_amount / trade_risk if trade_risk != 0 else 0
    return size

# ─── تشخیص واگرایی (Bullish / Bearish Divergence) ───
def detect_divergence(df, lookback=14):
    if len(df) < lookback+1:
        return None
    recent = df.tail(lookback)
    # ساده‌ترین حالت: مقایسه lows و highs با RSI
    lows = recent['close'].min()
    highs = recent['close'].max()
    rsi = recent['StochRSI'].iloc[-lookback:]
    # Bullish Divergence
    if (recent['close'].iloc[-1] < recent['close'].iloc[-2]) and (rsi.iloc[-1] > rsi.iloc[-2]):
        return 'bullish_divergence'
    # Bearish Divergence
    if (recent['close'].iloc[-1] > recent['close'].iloc[-2]) and (rsi.iloc[-1] < rsi.iloc[-2]):
        return 'bearish_divergence'
    return None

# ─── گرفتن OHLCV و محاسبه تمام اندیکاتورها ───
def fetch_ohlcv(symbol="BTC/USDT", timeframe="1h", limit=200):
    df = pd.DataFrame(exchange.fetch_ohlcv(symbol, timeframe, limit=limit),
                      columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df = ichimoku(df)
    df = calculate_stoch_rsi(df)
    df = calculate_atr(df)
    return df

# ─── دریافت 80 ارز برتر کوکوین بر اساس حجم 24 ساعته ───
def get_top_80_kucoin_symbols():
    url = "https://api.kucoin.com/api/v1/market/allTickers"
    try:
        response = requests.get(url, timeout=10).json()
        tickers = response['data']['ticker']
        tickers_sorted = sorted(tickers, key=lambda x: float(x['volValue']), reverse=True)
        symbols = [t['symbol'] for t in tickers_sorted if t['symbol'].endswith('USDT')]
        return symbols[:80]
    except Exception as e:
        logging.error(f"❌ خطا در گرفتن نمادها: {e}")
        return []
# ─── تولید سیگنال و پیام چندخطی با تایم‌فریم‌های متعدد ───
def generate_multiframe_signal(symbol, timeframes=["5m","15m","1h","4h"]):
    messages = []
    signals_count = {"BUY":0, "SELL":0}
    frame_signals = {}

    for tf in timeframes:
        df = fetch_ohlcv(symbol, tf)
        msg = generate_signal(df, symbol=f"{symbol} ({tf})")
        frame_signals[tf] = msg
        if msg:
            if "BUY" in msg.splitlines()[1]:
                signals_count["BUY"] += 1
            elif "SELL" in msg.splitlines()[1]:
                signals_count["SELL"] += 1
        messages.append(msg)

    # تصمیم نهایی بر اساس تایید حداقل 2 تایم‌فریم
    final_signal = None
    if signals_count["BUY"] >= 2:
        final_signal = f"🔔 **سیگنال BUY** برای {symbol} تایید شده توسط {signals_count['BUY']} تایم‌فریم"
    elif signals_count["SELL"] >= 2:
        final_signal = f"🔔 **سیگنال SELL** برای {symbol} تایید شده توسط {signals_count['SELL']} تایم‌فریم"

    return final_signal, messages

# ─── حلقه اصلی ───
def main():
    timeframes = ["5m","15m","1h","4h"]
    symbols = get_top_80_kucoin_symbols()
    if not symbols:
        logging.error("❌ هیچ نمادی دریافت نشد.")
        return

    while True:
        for symbol in symbols:
            try:
                final_signal, frame_msgs = generate_multiframe_signal(symbol, timeframes)
                if final_signal:
                    logging.info(final_signal)
                    try:
                        bot.send_message(chat_id=CHAT_ID, text=final_signal)
                    except Exception as e:
                        logging.error(f"[Telegram Error] {e}")
                # ارسال پیام‌های تک‌تایم‌فریم در صورت نیاز (غیر ضروری)
                # for msg in frame_msgs:
                #     if msg:
                #         bot.send_message(chat_id=CHAT_ID, text=msg)
                time.sleep(1)  # فاصله بین نمادها
            except Exception as e:
                logging.error(f"❌ خطا در پردازش {symbol}: {e}")
        time.sleep(60*5)  # هر 5 دقیقه کل 80 ارز چک می‌شوند

if __name__ == "__main__":
    main()
