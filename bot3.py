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

exchange = ccxt.kucoin()

# ─── ====== CONFIG: اینها رو میتونی تغییر بدی ======
TOP_N = 80
TIMEFRAMES = ['5m','15m','30m','1h','4h']
LOW_TF_TO_REQUIRE_HIGH_CONFIRM = ['5m']   # تایم‌فریم‌های پایین که نیاز به تائید 1h/4h دارن
HIGH_TFS = ['1h','4h']
SIGNALS_PER_CYCLE = 5        # ابتدا 1 یا 2 بذار؛ بعدا میتونی بالا ببری
MIN_SCORE = 12.0              # حداقل امتیازی که کاندید باید داشته باشه تا Consider بشه
MIN_SCORE_HIGH_CONFIRM = 10.0 # اگر تایم‌فریم پایین باشه، تاییدیه high-tf باید این حد رو داشته باشه
SEND_DELAY_BETWEEN_MSGS = 1.0 # ثانیه بین ارسال پیام‌ها
SIGNAL_INTERVAL = 5 * 60     # cooldown برای هر نماد (ثانیه)
# ─── ================================================

last_signal_time = {}   # cooldown per symbol
last_alerts = {}        # جلوگیری از تکرار فوری

# ─── مدیریت ریسک ───
ACCOUNT_BALANCE = 1000.0   # موجودی فرضی (دلار)
RISK_PER_TRADE = 0.01      # 1 درصد ریسک در هر معامله

def calculate_position_size(entry, stop):
    try:
        risk_amount = ACCOUNT_BALANCE * RISK_PER_TRADE
        risk_per_unit = abs(entry - stop)
        if risk_per_unit == 0:
            return 0
        position_size = risk_amount / risk_per_unit
        return round(position_size, 3)
    except Exception:
        return 0

# ─── گرفتن ۸۰ ارز برتر (دفاعی در برابر داده ناقص)
def get_top_symbols():
    try:
        tickers = exchange.fetch_tickers()
    except Exception as e:
        print(f"[ERROR] fetch_tickers failed: {e}")
        return []
    symbols = []
    for symbol, data in tickers.items():
        try:
            if symbol.endswith('/USDT'):
                vol = data.get('quoteVolume') if isinstance(data, dict) else data['quoteVolume']
                ch = data.get('percentage') if isinstance(data, dict) else data['percentage']
                symbols.append({
                    'symbol': symbol,
                    'volume': vol if vol is not None else 0,
                    'change': ch if ch is not None else 0
                })
        except Exception:
            continue
    symbols.sort(key=lambda x: x['volume'], reverse=True)
    return symbols[:TOP_N]

# ─── گرفتن داده OHLCV (دفاعی)
def get_ohlcv_df(symbol, timeframe, limit=200):
    try:
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
        df = pd.DataFrame(ohlcv, columns=['timestamp','open','high','low','close','volume'])
        return df
    except Exception as e:
        print(f"[ERROR] fetch_ohlcv {symbol} {timeframe}: {e}")
        return pd.DataFrame()

# ─── اندیکاتورهای تکمیلی: Pivot, OBV, VWAP, Fibonacci

def calculate_pivot_points(df):
    try:
        df['Pivot'] = (df['high'] + df['low'] + df['close']) / 3
        df['R1'] = 2*df['Pivot'] - df['low']
        df['S1'] = 2*df['Pivot'] - df['high']
        df['R2'] = df['Pivot'] + (df['high'] - df['low'])
        df['S2'] = df['Pivot'] - (df['high'] - df['low'])
        df['R3'] = df['high'] + 2*(df['Pivot']-df['low'])
        df['S3'] = df['low'] - 2*(df['high']-df['Pivot'])
    except Exception:
        # در صورت بروز خطا، ستون‌ها اضافه نمیشن اما تابع ادامه پیدا میکنه
        pass
    return df


def calculate_obv(df):
    try:
        obv = [0]
        for i in range(1, len(df)):
            if df['close'].iloc[i] > df['close'].iloc[i-1]:
                obv.append(obv[-1] + df['volume'].iloc[i])
            elif df['close'].iloc[i] < df['close'].iloc[i-1]:
                obv.append(obv[-1] - df['volume'].iloc[i])
            else:
                obv.append(obv[-1])
        df['OBV'] = obv
    except Exception:
        df['OBV'] = np.nan
    return df


def calculate_vwap(df):
    try:
        typical = (df['high'] + df['low'] + df['close']) / 3.0
        cum_vol_price = (typical * df['volume']).cumsum()
        cum_vol = df['volume'].cumsum()
        df['VWAP'] = cum_vol_price / (cum_vol.replace(0, np.nan))
    except Exception:
        df['VWAP'] = np.nan
    return df


def calculate_fibonacci(df, lookback=20):
    try:
        swing_high = df['high'].iloc[-lookback:].max()
        swing_low = df['low'].iloc[-lookback:].min()
        diff = swing_high - swing_low if swing_high != swing_low else 0.0

        levels = {
            'fib_0': swing_high,
            'fib_0.236': swing_high - 0.236 * diff,
            'fib_0.382': swing_high - 0.382 * diff,
            'fib_0.5': swing_high - 0.5 * diff,
            'fib_0.618': swing_high - 0.618 * diff,
            'fib_0.786': swing_high - 0.786 * diff,
            'fib_1': swing_low,
            'fib_1.618': swing_high + 0.618 * diff,
            'fib_2.618': swing_high + 1.618 * diff,
        }
        for k, v in levels.items():
            df[k] = v
    except Exception:
        # اگر داده کافی نیست، ستون‌ها را با NaN پر کن
        df['fib_0'] = np.nan
        df['fib_0.236'] = np.nan
        df['fib_0.382'] = np.nan
        df['fib_0.5'] = np.nan
        df['fib_0.618'] = np.nan
        df['fib_0.786'] = np.nan
        df['fib_1'] = np.nan
        df['fib_1.618'] = np.nan
        df['fib_2.618'] = np.nan
    return df

# ─── اندیکاتورها (همون نسخهٔ قبلی)
def calculate_indicators(df):
    if df is None or len(df) < 60:
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
    df['StochRSI'] = (df['close'] - df['close'].rolling(14).min()) / (df['close'].rolling(14).max() - df['close'].rolling(14).min() + 1e-9)

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

    # ADX (ساده)
    df['+DM'] = np.where((df['high'].diff() > df['low'].diff()) & (df['high'].diff() > 0),
                         df['high'].diff(), 0)
    df['-DM'] = np.where((df['low'].diff() > df['high'].diff()) & (df['low'].diff() > 0),
                         df['low'].diff(), 0)
    atr14 = df['ATR14'].replace(0, np.nan)
    df['+DI'] = 100 * (df['+DM'].ewm(alpha=1/14).mean() / (atr14))
    df['-DI'] = 100 * (df['-DM'].ewm(alpha=1/14).mean() / (atr14))
    df['DX'] = (abs(df['+DI'] - df['-DI']) / (df['+DI'] + df['-DI'] + 1e-9)) * 100
    df['ADX'] = df['DX'].ewm(alpha=1/14).mean()

    # SuperTrend (ساده)
    factor = 3
    hl2 = (df['high'] + df['low']) / 2
    df['UpperBand'] = hl2 + (factor * df['ATR14'])
    df['LowerBand'] = hl2 - (factor * df['ATR14'])
    df['SuperTrend'] = np.where(df['close'] > df['UpperBand'], 1,
                                np.where(df['close'] < df['LowerBand'], -1, 0))

    # ─── اضافه کردن اندیکاتورهای جدید بدون تغییر ساختار
    df = calculate_pivot_points(df)
    df = calculate_obv(df)
    df = calculate_vwap(df)
    df = calculate_fibonacci(df)

    return df

# ─── شناسایی کندل‌ها
def detect_candlestick_patterns(df):
    if df is None or len(df) < 3:
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

# ===== تابع امتیازدهی =====
def compute_signal_score(sig, df, intrabar_change):
    try:
        stars_count = len(sig.get('stars', []))
        adx = float(df['ADX'].iloc[-1]) if 'ADX' in df.columns and not pd.isna(df['ADX'].iloc[-1]) else 0.0
        vol_mean = float(df['volume'].rolling(20).mean().iloc[-1]) if 'volume' in df.columns else np.nan
        vol_rel = 1.0
        if not pd.isna(vol_mean) and vol_mean > 0:
            vol_rel = float(df['volume'].iloc[-1]) / vol_mean
        rsi = float(df['RSI'].iloc[-1]) if 'RSI' in df.columns and not pd.isna(df['RSI'].iloc[-1]) else 50.0
        ema9 = float(df['EMA9'].iloc[-1]) if 'EMA9' in df.columns and not pd.isna(df['EMA9'].iloc[-1]) else 0.0
        ema21 = float(df['EMA21'].iloc[-1]) if 'EMA21' in df.columns and not pd.isna(df['EMA21'].iloc[-1]) else 1.0
        ema_diff = (ema9 - ema21) / (ema21 if ema21 != 0 else 1e-9)

        score = 0.0
        score += stars_count * 8.0
        score += min(adx, 50.0)
        score += min(abs(intrabar_change) * 5, 25.0)
        if vol_rel > 1.0:
            score += (vol_rel - 1.0) * 10.0
        if sig.get('type') == 'LONG':
            score += max(0.0, ema_diff * 100.0)
        elif sig.get('type') == 'SHORT':
            score += max(0.0, -ema_diff * 100.0)
        if rsi > 80:
            score -= (rsi - 80) * 0.8
        if rsi < 20:
            score -= (20 - rsi) * 0.8

        return round(score, 2)
    except Exception:
        return 0.0

# ===== چک سیگنال (الگوی قبلی با خروجی score) =====
def check_signal(df, symbol, change):
    try:
        if df is None or len(df) < 60:
            return None

        # بررسی وجود فیلدهای مورد نیاز
        needed = ['EMA9','EMA21','ATR14','RSI','ADX','volume']
        if any(col not in df.columns or pd.isna(df[col].iloc[-1]) for col in needed):
            return None

        price = df['close'].iloc[-1]
        prev_price = df['close'].iloc[-2]
        intrabar_change = ((price - prev_price) / prev_price) * 100.0

        trend = 'neutral'
        if not pd.isna(df['SenkouA'].iloc[-1]) and not pd.isna(df['SenkouB'].iloc[-1]):
            if price > df['SenkouA'].iloc[-1] and price > df['SenkouB'].iloc[-1]:
                trend = 'bullish'
            elif price < df['SenkouA'].iloc[-1] and price < df['SenkouB'].iloc[-1]:
                trend = 'bearish'

        patterns = detect_candlestick_patterns(df)

        # ستاره‌ها (شُل‌تر)
        stars = []
        vol_mean = df['volume'].rolling(20).mean().iloc[-1]
        if not pd.isna(vol_mean) and df['volume'].iloc[-1] > vol_mean * 1.2:
            stars.append('🔹')
        if df['ATR'].iloc[-1] > df['ATR'].rolling(14).mean().iloc[-1]:
            stars.append('🔹')
        if df['ADX'].iloc[-1] > 20:
            stars.append('🔹')
        if patterns:
            stars.append('🔹')

        signal_type = None
        entry = tp = stop = size = None
        atr = df['ATR14'].iloc[-1]

        # لاگ مختصر برای دیباگ
        print(f"[LOG] {symbol} | intrabarΔ={intrabar_change:.3f}% | 24hΔ={change:.2f}% | Trend={trend} | RSI={df['RSI'].iloc[-1]:.1f} | Stars={len(stars)}")

        # شروط ورود (ملایم)
        if (intrabar_change >= 0.2 and change >= 0.2 and trend == 'bullish' and len(stars) >= 2
            and df['EMA9'].iloc[-1] > df['EMA21'].iloc[-1] and df['RSI'].iloc[-1] > 48):
            signal_type = 'LONG'
            entry = price
            stop = price - 1.2 * atr
            tp = price + 1.8 * atr

        elif (intrabar_change <= -0.2 and change <= -0.2 and trend == 'bearish' and len(stars) >= 2
              and df['EMA9'].iloc[-1] < df['EMA21'].iloc[-1] and df['RSI'].iloc[-1] < 52):
            signal_type = 'SHORT'
            entry = price
            stop = price + 1.2 * atr
            tp = price - 1.8 * atr

        if signal_type and entry and stop:
            size = calculate_position_size(entry, stop)

        if not signal_type:
            return None

        temp_sig = {
            'entry': entry, 'tp': tp, 'stop': stop, 'type': signal_type,
            'patterns': patterns, 'stars': stars, 'size': size
        }
        score = compute_signal_score(temp_sig, df, intrabar_change)
        temp_sig['score'] = score

        # strength label (اختیاری برای پیام)
        if score >= (MIN_SCORE * 2):
            temp_sig['strength'] = 'strong'
        elif score >= MIN_SCORE:
            temp_sig['strength'] = 'normal'
        else:
            temp_sig['strength'] = 'weak'

        # برای جلوگیری از ارسال آنیِ دوباره، یک رکورد موقت در last_alerts بگذار
        prev = last_alerts.get(symbol)
        if prev and prev.get("type") == signal_type and (time.time() - prev.get("time", 0) < SIGNAL_INTERVAL):
            return None
        last_alerts[symbol] = {"type": signal_type, "time": time.time()}

        return temp_sig

    except Exception as e:
        print(f"[ERROR] check_signal {symbol}: {e}")
        return None

# ===== main() با جمع‌آوری همهٔ کاندیدها، فیلتر، رتبه‌بندی، و قواعد تایید تایم‌فریم پایین =====
def main():
    print("🚀 ربات شروع شد — priority + tf-confirmation فعال")
    while True:
        try:
            top_symbols = get_top_symbols()
            candidates = []

            for symbol_data in top_symbols:
                symbol = symbol_data['symbol']
                for tf in TIMEFRAMES:
                    try:
                        df = get_ohlcv_df(symbol, tf)
                        if df is None or df.empty:
                            continue
                        df = calculate_indicators(df)
                        sig = check_signal(df, symbol, symbol_data['change'])
                        if sig:
                            candidates.append({
                                'symbol': symbol,
                                'tf': tf,
                                'signal': sig,
                                'score': sig.get('score', 0.0)
                            })
                    except Exception as e:
                        print(f"[ERROR] {symbol} | TF: {tf} | {e}")
                        continue

            print(f"[INFO] Found {len(candidates)} candidates this cycle.")

            # فیلتر بر اساس MIN_SCORE (اگر خواستی میتونی این خطو حذف کنی تا همه رو نگه داره)
            filtered = [c for c in candidates if c['score'] >= MIN_SCORE]
            print(f"[INFO] {len(filtered)} candidates passed MIN_SCORE >= {MIN_SCORE}")

            # مرتب‌سازی بر اساس score نزولی
            filtered.sort(key=lambda x: x['score'], reverse=True)

            # انتخاب نهایی با رعایت محدودیت‌ها و تایید TF پایین
            selected = []
            used_symbols = set()
            for c in filtered:
                if len(selected) >= SIGNALS_PER_CYCLE:
                    break
                sym = c['symbol']
                tf = c['tf']
                sig = c['signal']
                # cooldown per symbol
                last_t = last_signal_time.get(sym, 0)
                if time.time() - last_t < SIGNAL_INTERVAL:
                    continue
                if sym in used_symbols:
                    continue

                # اگر تایم‌فریم از لیست تایم‌فریم‌های پایین بود، نیاز به تأیید high-tf داریم
                if tf in LOW_TF_TO_REQUIRE_HIGH_CONFIRM:
                    # آیا توی filtered یک تایید کننده high-tf برای همین نماد وجود داره؟
                    confirmed = False
                    for c2 in filtered:
                        if c2['symbol'] == sym and c2['tf'] in HIGH_TFS and c2['signal']['type'] == sig['type'] and c2['score'] >= MIN_SCORE_HIGH_CONFIRM:
                            confirmed = True
                            break
                    if not confirmed:
                        # رد کن (نیاز به تایید high-tf)
                        # ولی اگر می‌خوای خیلی انعطاف بدی میتونی خط زیرو کامنت کنی تا low-tf بدون تایید هم بفرسته
                        # print(f"[INFO] {sym} {tf} skipped: no high-tf confirmation")
                        continue

                # انتخاب
                selected.append(c)
                used_symbols.add(sym)
                last_signal_time[sym] = time.time()

            print(f"[INFO] Selected {len(selected)} signals to send (max {SIGNALS_PER_CYCLE}).")

            # ارسال پیام‌ها با فاصله کوتاه
            if selected:
                for c in selected:
                    s = c['signal']
                    sym = c['symbol']
                    tf = c['tf']
                    color_emoji = "🟢" if s['type'] == "LONG" else "🔴"
                    strength_tag = " 🔥" if s.get('strength') == 'strong' else (" ⭐" if s.get('strength') == 'normal' else "")
                    now_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    msg = (f"🚨 Multi-Coin Alert 🚨\n"
                           f"{color_emoji} {sym} | TF: {tf}{strength_tag}\n"
                           f"Type: {s['type']}\n"
                           f"Entry: {s['entry']:.4f}\n"
                           f"TP: {s['tp']:.4f}\n"
                           f"Stop: {s['stop']:.4f}\n"
                           f"Size: {s['size']}\n"
                           f"Score: {s.get('score', 0.0)}\n"
                           f"Patterns: {s['patterns']}\n"
                           f"Conditions: {''.join(s['stars'])}\n"
                           f"🕒 Time: {now_time}")
                    try:
                        bot.send_message(chat_id=CHAT_ID, text=msg)
                        print(f"[SENT] {sym} | TF:{tf} | Score:{s.get('score',0)}")
                    except Exception as e:
                        print(f"[ERROR] sending telegram {sym}: {e}")
                    time.sleep(SEND_DELAY_BETWEEN_MSGS)

            # صبر تا چرخه بعدی
            time.sleep(300)

        except Exception as e:
            print(f"⚠️ خطا در main: {e}")
            time.sleep(30)

if __name__ == "__main__":
    main()
