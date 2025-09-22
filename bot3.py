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
def generate_signal(df, symbol=None):
    """
    تولید سیگنال و متن پیام با:
    - شمارش ۵ شرط (روند، الگو، اوردر بلاک، حجم+StochRSI، ATR)
    - نمایش تعداد شروط تایید شده به صورت ستاره ⭐
    - ارسال جزئیات خط به خط (multiline)
    """
    trend = check_trend_strength(df)
    patterns = detect_candlestick_patterns(df)
    order_blocks = detect_order_block(df)
    fibonacci_levels = calculate_fibonacci(df)

    atr = df['ATR'].iloc[-1] if 'ATR' in df.columns else None
    entry = df['close'].iloc[-1]
    volume_mean = df['volume'].rolling(20).mean().iloc[-1] if len(df) >= 20 else df['volume'].mean()
    volume_check = df['volume'].iloc[-1] > volume_mean * 1.5
    stoch = df['StochRSI'].iloc[-1] if 'StochRSI' in df.columns else None
    stoch_buy = (stoch < 0.2) if (stoch is not None and not np.isnan(stoch)) else False
    stoch_sell = (stoch > 0.8) if (stoch is not None and not np.isnan(stoch)) else False
    atr_mean = df['ATR'].rolling(14).mean().iloc[-1] if 'ATR' in df.columns and len(df) >= 14 else None
    atr_check = False
    if (atr is not None) and (atr_mean is not None) and (not np.isnan(atr_mean)):
        atr_check = atr > atr_mean

    # ----- پنج شرط (برای خرید / فروش) -----
    cond_trend_buy = ('bullish' in trend)
    cond_trend_sell = ('bearish' in trend)

    cond_pattern_buy = any(p in patterns for p in ['Bullish Engulfing', 'Morning Star', 'Three White Soldiers'])
    cond_pattern_sell = any(p in patterns for p in ['Bearish Engulfing', 'Evening Star', 'Three Black Crows'])

    cond_order_buy = any('Bullish' in ob for ob in order_blocks) if order_blocks else False
    cond_order_sell = any('Bearish' in ob for ob in order_blocks) if order_blocks else False

    cond_vol_stoch_buy = volume_check and stoch_buy
    cond_vol_stoch_sell = volume_check and stoch_sell

    buy_conditions = [cond_trend_buy, cond_pattern_buy, cond_order_buy, cond_vol_stoch_buy, atr_check]
    sell_conditions = [cond_trend_sell, cond_pattern_sell, cond_order_sell, cond_vol_stoch_sell, atr_check]

    buy_count = sum(1 for c in buy_conditions if c)
    sell_count = sum(1 for c in sell_conditions if c)

    # اگر هیچ شرطی برقرار نبود، برگردان None (هیچ پیامی ارسال نشود)
    if buy_count == 0 and sell_count == 0:
        return None

    # تعیین جهت بر اساس بیشترین شروط تایید شده (در صورت تساوی: جهت با count بزرگ‌تر یا اولویت خرید)
    if buy_count >= sell_count:
        side = "BUY"
        conditions_met = buy_count
        chosen_conditions = buy_conditions
        chosen_pattern_flag = cond_pattern_buy
        chosen_order_flag = cond_order_buy
        chosen_vol_stoch_flag = cond_vol_stoch_buy
        chosen_trend_flag = cond_trend_buy
    else:
        side = "SELL"
        conditions_met = sell_count
        chosen_conditions = sell_conditions
        chosen_pattern_flag = cond_pattern_sell
        chosen_order_flag = cond_order_sell
        chosen_vol_stoch_flag = cond_vol_stoch_sell
        chosen_trend_flag = cond_trend_sell

    # محاسبه SL/TP/Size بر اساس ATR (همان منطق قبلی)
    if atr is None or np.isnan(atr):
        # اگر ATR در دسترس نبود از یک فاصله پیش‌فرض استفاده کن
        atr = (df['high'].iloc[-1] - df['low'].iloc[-1])
    if side == "BUY":
        stop = entry - atr * 1.5
        tp = entry + atr * 2
    else:
        stop = entry + atr * 1.5
        tp = entry - atr * 2

    size = position_size(entry, stop)  # تابع position_size قبلاً تعریف شده

    # ساخت پیام چندخطی با نمایش ستاره‌ها و وضعیت هر شرط
    stars = "⭐" * int(conditions_met)
    lines = []
    if symbol:
        lines.append(f"🔔 سیگنال برای {symbol}")
    lines.append(f"نوع سیگنال: {side}")
    lines.append(f"تعداد شروط تایید شده: {conditions_met}/5 {stars}")
    lines.append("")  # خط خالی برای خوانایی

    # نمایش وضعیت هر شرط بصورت جداگانه
    lines.append(f"1) فیلتر روند: {trend} {'✅' if chosen_trend_flag else '❌'}")
    lines.append(f"2) الگوهای کندلی: {patterns} {'✅' if chosen_pattern_flag else '❌'}")
    lines.append(f"3) اوردر بلاک: {order_blocks} {'✅' if chosen_order_flag else '❌'}")
    stoch_text = f"حجم={df['volume'].iloc[-1]:.2f}, StochRSI={stoch:.3f}" if stoch is not None else f"حجم={df['volume'].iloc[-1]:.2f}, StochRSI=N/A"
    lines.append(f"4) حجم + StochRSI: {stoch_text} {'✅' if chosen_vol_stoch_flag else '❌'}")
    atr_text = f"{atr:.6f}" if atr is not None else "N/A"
    lines.append(f"5) ATR check: {atr_text} {'✅' if atr_check else '❌'}")

    lines.append("")  # فاصله
    lines.append(f"Entry: {entry:.6f}")
    lines.append(f"Stop: {stop:.6f}")
    lines.append(f"TP: {tp:.6f}")
    lines.append(f"Size (units): {size:.6f}")

    # فیبوناچی
    lines.append("") 
    lines.append("📊 سطوح فیبوناچی:")
    for k, v in fibonacci_levels.items():
        lines.append(f"  {k}: {v:.6f}")

    # توضیح تکمیلی (اگر خواستی میشه در اینجا نکات اضافی هم قرار داد)
    message = "\n".join(lines)
    return message


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
            signal_text = generate_signal(df, symbol=symbol)

            if signal_text:
                logging.info(signal_text)
                try:
                    bot.send_message(chat_id=CHAT_ID, text=signal_text)
                except Exception as e:
                    logging.error(f"[Telegram Error] {e}")

            time.sleep(60 * 5)  # هر ۵ دقیقه
        except Exception as e:
            logging.error(f"❌ خطا: {e}")
            time.sleep(60)


if __name__ == "__main__":
    main()
