# bot3_fixed.py
import tim
import os
import logging
import ccxt
import pandas as pd
import numpy as np
from telegram import Bot

# ----------------- تنظیمات -----------------
TOP_LIMIT = 80                    # تعداد ارزها
TIMEFRAME = "5m"                  # تایم‌فریم بررسی
LOOP_INTERVAL = 300               # 5 دقیقه بین هر دور بررسی (ثانیه)
SLEEP_BETWEEN_SYMBOLS = 4         # ثانیه تاخیر بین هر نماد (برای جلوگیری از RateLimit)
SIGNAL_COOLDOWN_SECONDS = 30 * 60 # 30 دقیقه قبل از ارسال مجدد همان سیگنال برای یک نماد

# ----------------- لاگ و اتصال -----------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
exchange = ccxt.kucoin({"enableRateLimit": True})

TELEGRAM_TOKEN = os.getenv("BOT_TOKEN")
CHAT_ID = os.getenv("CHAT_ID")
bot = Bot(token=TELEGRAM_TOKEN)

def send_telegram(msg):
    try:
        bot.send_message(chat_id=CHAT_ID, text=msg)
    except Exception as e:
        logging.error(f"❌ خطا در ارسال تلگرام: {e}")

# ----------------- گرفتن 80 ارز برتر با ایمن‌سازی -----------------
def get_top_symbols(limit=TOP_LIMIT):
    try:
        tickers = exchange.fetch_tickers()  # mapping: symbol -> ticker
        items = []
        for sym, t in tickers.items():
            if not isinstance(sym, str) or "/USDT" not in sym:
                continue
            # سعی می‌کنیم quoteVolume یا baseVolume و انواع احتمالی را برداریم
            vol = 0.0
            if isinstance(t, dict):
                vol = t.get("quoteVolume") or t.get("baseVolume") or t.get("quote_volume") or t.get("base_volume") or 0.0
            try:
                vol = float(vol) if vol is not None else 0.0
            except:
                vol = 0.0
            items.append((sym, vol))
        # مرتب‌سازی بر اساس حجم نزولی
        items_sorted = sorted(items, key=lambda x: x[1], reverse=True)[:limit]
        symbols = [s for s, v in items_sorted]
        logging.info(f"🔝 انتخاب {len(symbols)} ارز برتر با بیشترین حجم (نمونه 10 اول): {symbols[:10]}")
        return symbols
    except Exception as e:
        logging.warning(f"⚠️ fetch_tickers() خطا داد: {e} — تلاش با load_markets() به عنوان جایگزین")
        try:
            markets = exchange.load_markets()
            markets_list = []
            for m in markets.values():
                sym = m.get("symbol")
                if not sym or "/USDT" not in sym:
                    continue
                vol = m.get("quoteVolume", 0) or m.get("baseVolume", 0) or 0
                try:
                    vol = float(vol)
                except:
                    vol = 0.0
                markets_list.append((sym, vol))
            markets_sorted = sorted(markets_list, key=lambda x: x[1], reverse=True)[:limit]
            symbols = [s for s, v in markets_sorted]
            logging.info(f"🔝 (fallback) انتخاب {len(symbols)} ارز برتر (نمونه 10 اول): {symbols[:10]}")
            return symbols
        except Exception as e2:
            logging.error(f"❌ نتوانستم لیست نمادها را دریافت کنم: {e2}")
            return []

# ----------------- گرفتن کندل‌ها -----------------
def fetch_ohlcv(symbol, timeframe=TIMEFRAME, limit=200):
    try:
        data = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
        df = pd.DataFrame(data, columns=["time","open","high","low","close","volume"])
        # تبدیل انواع به float
        df[['open','high','low','close','volume']] = df[['open','high','low','close','volume']].astype(float)
        return df
    except Exception as e:
        logging.error(f"❌ خطا در fetch_ohlcv({symbol}, {timeframe}): {e}")
        return None

# ----------------- اندیکاتورها (بدون کتابخانه ta) -----------------
def add_indicators(df):
    close = df['close']
    # EMA20
    df['EMA20'] = close.ewm(span=20, adjust=False).mean()
    # MACD
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    df['MACD'] = ema12 - ema26
    df['Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    # RSI
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / (avg_loss.replace(0, np.nan))
    df['RSI'] = 100 - (100 / (1 + rs))
    df['RSI'] = df['RSI'].fillna(50)  # جایگزین مقادیر ناموجود
    # ATR (برای SL/TP)
    df['H-L'] = df['high'] - df['low']
    df['H-PC'] = (df['high'] - df['close'].shift(1)).abs()
    df['L-PC'] = (df['low'] - df['close'].shift(1)).abs()
    df['TR'] = df[['H-L','H-PC','L-PC']].max(axis=1)
    df['ATR14'] = df['TR'].rolling(14).mean().fillna(method='bfill').fillna(0.0)
    return df

# ----------------- الگوهای کندلی ساده -----------------
def detect_candlestick(df):
    if len(df) < 2:
        return None
    last = df.iloc[-1]
    prev = df.iloc[-2]
    body = abs(last['close'] - last['open'])
    range_ = last['high'] - last['low']
    # Doji
    if range_ > 0 and body <= 0.1 * range_:
        return "Doji"
    # Hammer (ساده)
    if (last['close'] > last['open']) and ((last['close'] - last['low']) > 2 * (last['high'] - last['close'])):
        return "Hammer"
    # Bullish Engulfing
    if (last['close'] > last['open']) and (prev['close'] < prev['open']) and (last['close'] > prev['open']) and (last['open'] < prev['close']):
        return "Bullish Engulfing"
    # Bearish Engulfing
    if (last['close'] < last['open']) and (prev['close'] > prev['open']) and (last['open'] > prev['close']) and (last['close'] < prev['open']):
        return "Bearish Engulfing"
    return None

# ----------------- خط روند ساده (دو سوینگ) -----------------
def detect_trendline(prices, window=5, tol=0.01):
    highs, lows = [], []
    n = len(prices)
    if n < window*2 + 1:
        return {"resistance": None, "support": None, "signal": None}
    for i in range(window, n-window):
        segment = prices[i-window:i+window+1]
        if prices[i] == max(segment):
            highs.append((i, prices[i]))
        if prices[i] == min(segment):
            lows.append((i, prices[i]))
    trend_info = {"resistance": None, "support": None, "signal": None}
    last_idx = n - 1
    last_price = prices[-1]
    try:
        if len(highs) >= 2:
            xh, yh = zip(*highs[-2:])
            a,b = np.polyfit(xh, yh, 1)
            trend_info["resistance"] = (a,b)
            expected = a*last_idx + b
            if expected != 0 and abs(last_price - expected) / abs(expected) < tol:
                trend_info["signal"] = "SELL (نزدیک مقاومت)"
        if len(lows) >= 2:
            xl, yl = zip(*lows[-2:])
            a,b = np.polyfit(xl, yl, 1)
            trend_info["support"] = (a,b)
            expected = a*last_idx + b
            if expected != 0 and abs(last_price - expected) / abs(expected) < tol:
                trend_info["signal"] = "BUY (نزدیک حمایت)"
    except Exception as e:
        logging.debug(f"⚠️ خطا در محاسبه trendline: {e}")
    return trend_info

# ----------------- منطق سیگنال (حداقل 2 شرط) -----------------
EMA_THRESHOLD = 0.002     # 0.2% فاصله از EMA برای تأیید قوی‌تر
MACD_THRESHOLD = 1e-8     # کمی فاصله برای تشخیص کراس
RSI_BUY_MAX = 60
RSI_SELL_MIN = 40

def analyze_df(symbol, df):
    if df is None or df.empty:
        return None
    if len(df) < 20:
        return None
    last = df.iloc[-1]
    signals = []
    reasons = []

    # EMA condition
    try:
        if last['close'] > last['EMA20'] * (1 + EMA_THRESHOLD):
            signals.append("BUY")
            reasons.append("EMA")
        elif last['close'] < last['EMA20'] * (1 - EMA_THRESHOLD):
            signals.append("SELL")
            reasons.append("EMA")
    except Exception:
        pass

    # RSI
    try:
        if last['RSI'] < RSI_BUY_MAX:
            signals.append("BUY")
            reasons.append("RSI")
        elif last['RSI'] > RSI_SELL_MIN:
            signals.append("SELL")
            reasons.append("RSI")
    except Exception:
        pass

    # MACD
    try:
        diff = last['MACD'] - last['Signal']
        if diff > MACD_THRESHOLD:
            signals.append("BUY"); reasons.append("MACD")
        elif diff < -MACD_THRESHOLD:
            signals.append("SELL"); reasons.append("MACD")
    except Exception:
        pass

    # Candlestick
    pattern = detect_candlestick(df)
    if pattern in ["Hammer", "Bullish Engulfing"]:
        signals.append("BUY"); reasons.append(f"Candle:{pattern}")
    if pattern in ["Bearish Engulfing"]:
        signals.append("SELL"); reasons.append(f"Candle:{pattern}")

    # Trendline
    trend = detect_trendline(df['close'].values)
    if trend.get("signal"):
        if "BUY" in trend["signal"]:
            signals.append("BUY"); reasons.append("TrendSupport")
        elif "SELL" in trend["signal"]:
            signals.append("SELL"); reasons.append("TrendResist")

    # count
    buy_count = sum(1 for s in signals if "BUY" in s)
    sell_count = sum(1 for s in signals if "SELL" in s)

    # decide (min 2 conditions)
    if buy_count >= 2:
        entry = float(last['close'])
        atr = float(last.get('ATR14', 0.0)) if 'ATR14' in last else 0.0
        if atr > 0:
            sl = entry - atr * 1.5
            tp = entry + atr * 3
        else:
            sl = entry * 0.985
            tp = entry * 1.03
        return {"symbol": symbol, "signal": "BUY", "entry": entry, "sl": sl, "tp": tp, "pattern": pattern, "reasons": reasons, "trend": trend}
    if sell_count >= 2:
        entry = float(last['close'])
        atr = float(last.get('ATR14', 0.0)) if 'ATR14' in last else 0.0
        if atr > 0:
            sl = entry + atr * 1.5
            tp = entry - atr * 3
        else:
            sl = entry * 1.015
            tp = entry * 0.97
        return {"symbol": symbol, "signal": "SELL", "entry": entry, "sl": sl, "tp": tp, "pattern": pattern, "reasons": reasons, "trend": trend}

    return None

# ----------------- run_bot (حلقه اصلی) -----------------
last_sent = {}  # symbol -> {"signal": "BUY"/"SELL", "time": timestamp}

def run_bot():
    logging.info("ربات در حال اجراست")
    while True:
        try:
            symbols = get_top_symbols(TOP_LIMIT)
            if not symbols:
                logging.warning("لیست نمادها خالی است — صبر می‌کنم و دوباره تلاش می‌کنم")
                time.sleep(60)
                continue

            logging.info(f"شروع بررسی {len(symbols)} نماد (نمونه 10 اول: {symbols[:10]})")
            for symbol in symbols:
                start_t = time.time()
                try:
                    df = fetch_ohlcv(symbol, timeframe=TIMEFRAME, limit=200)
                    if df is None or df.empty:
                        logging.debug(f"دیتای {symbol} خالی است")
                        time.sleep(SLEEP_BETWEEN_SYMBOLS)
                        continue

                    df = add_indicators(df)
                    result = analyze_df(symbol, df)

                    logging.info(f"🔎 بررسی {symbol} — آخرین قیمت: {df['close'].iloc[-1]:.6f}")

                    if result:
                        now = time.time()
                        prev = last_sent.get(symbol)
                        # اگر قبلاً هم همین سیگنال رو فرستاده‌ایم و هنوز در cooldown هستیم، رد شود
                        if prev and prev.get("signal") == result["signal"] and (now - prev.get("time", 0)) < SIGNAL_COOLDOWN_SECONDS:
                            logging.info(f"⏳ {symbol} → سیگنال {result['signal']} قبلاً ارسال شده و در بازه cooldown است. رد شد.")
                        else:
                            # ارسال پیام
                            msg = (
                                f"📊 سیگنال {result['signal']} برای {symbol}\n"
                                f"Entry: {result['entry']:.6f}\n"
                                f"Stop Loss: {result['sl']:.6f}\n"
                                f"Take Profit: {result['tp']:.6f}\n"
                                f"الگوی کندلی: {result.get('pattern')}\n"
                                f"دلایل: {', '.join(result.get('reasons', []))}\n"
                                f"خط روند: {result.get('trend', {}).get('signal')}"
                            )
                            send_telegram(msg)
                            logging.info(f"✅ ارسال سیگنال {result['signal']} برای {symbol} — دلایل: {result.get('reasons')}")
                            last_sent[symbol] = {"signal": result["signal"], "time": now}
                    else:
                        logging.debug(f"{symbol} سیگنال معتبر نداشت")
                except Exception as e_sym:
                    logging.error(f"❌ خطا در پردازش {symbol}: {e_sym}")
                # فاصله بین هر نماد برای جلوگیری از RateLimit
                elapsed = time.time() - start_t
                to_sleep = max(0, SLEEP_BETWEEN_SYMBOLS - elapsed)
                time.sleep(to_sleep)

            logging.info(f"⏱️ تمام نمادها بررسی شد — صبر برای {LOOP_INTERVAL} ثانیه تا دور بعدی")
            time.sleep(LOOP_INTERVAL)

        except Exception as e:
            logging.error(f"⚠️ خطای کلی در حلقه run_bot: {e}")
            time.sleep(60)

# ----------------- شروع فقط یک‌بار پیام -----------------
if __name__ == "__main__":
    send_telegram("✅ ربات فعال شد و شروع به کار می‌کند (80 ارز، هر 5 دقیقه بررسی).")
    logging.info("✅ ربات فعال شد و شروع به کار می‌کند.")
    run_bot()
