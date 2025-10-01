import time
import os
import ccxt
import pandas as pd
import numpy as np
import datetime
from collections import deque
from telegram import Bot
from keep_alive import keep_alive  # سرور کوچک برای جلوگیری از خوابیدن کانتینر

# ─── سرور کوچک
keep_alive()

# ─── اطلاعات ربات تلگرام
TELEGRAM_TOKEN = os.getenv("BOT_TOKEN")
CHAT_ID = os.getenv("CHAT_ID")
bot = Bot(token=TELEGRAM_TOKEN)

# ─── صرافی (rate limit فعال)
exchange = ccxt.kucoin({
    'enableRateLimit': True
})

# ─── ====== CONFIG: اینها رو میتونی تغییر بدی ======
TOP_N = 200
TIMEFRAMES = ['5m','15m','30m','1h','4h']
LOW_TF_TO_REQUIRE_HIGH_CONFIRM = ['5m','15m']   # تایم‌فریم‌های پایین که نیاز به تائید 1h/4h دارن
HIGH_TFS = ['1h','4h']
SIGNALS_PER_CYCLE = 3
MIN_SCORE = 13.0
MIN_SCORE_HIGH_CONFIRM = 10.0
SEND_DELAY_BETWEEN_MSGS = 1.0
SIGNAL_INTERVAL = 5 * 60

# دقت اضافه: سخت‌گیری‌های اختیاری
REQUIRE_DIVERGENCE = True          # اگر True، برای تایم پایین نیاز به واگرایی یا تایید TF بالاتر داریم
DIVERGENCE_LOOKBACK = 60
DIVERGENCE_ORDER = 3               # پارامتر تشخیص swing
VOLUME_SPIKE_FACTOR = 3.0          # نسبت به baseline که به عنوان اسپایک در نظر گرفته میشه
VOLUME_BASELINE_ALPHA = 0.15       # EMA alpha برای بروزرسانی baseline حجم
VOLUME_MIN_ABS = 100.0             # حداقل حجم مطلق برای شنیدن اسپایک (پایین بذار اگر میخوای ارزای کوچیکم رصد شه)
ANOMALY_COOLDOWN = 60 * 60         # یک ساعت
# ───────────────────────────────────────────────────

last_signal_time = {}
last_alerts = {}

# برای ذخیرهٔ baseline حجم و جلوگیری از اعلان مکرر
volume_baseline = {}
volume_last_alert = {}

# ─── مدیریت ریسک ───
ACCOUNT_BALANCE = 1000.0
RISK_PER_TRADE = 0.01

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

# ─── توابع کمکی امن برای فراخوانی API با retry ساده
def safe_fetch_tickers():
    for i in range(3):
        try:
            return exchange.fetch_tickers()
        except Exception as e:
            print(f"[WARN] fetch_tickers failed (retry {i+1}): {e}")
            time.sleep(1 + i * 2)
    return {}

def safe_fetch_ohlcv(symbol, timeframe, limit=200):
    for i in range(3):
        try:
            return exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
        except Exception as e:
            print(f"[WARN] fetch_ohlcv {symbol} {timeframe} failed (retry {i+1}): {e}")
            time.sleep(1 + i * 2)
    return None
# ─── گرفتن TOP symbols با اطلاعات حجم 24h
def get_top_symbols():
    tickers = safe_fetch_tickers()
    symbols = []
    for symbol, data in tickers.items():
        try:
            if symbol.endswith('/USDT'):
                # quoteVolume ممکنه داخل dict باشه
                vol = data.get('quoteVolume') if isinstance(data, dict) else data['quoteVolume']
                ch = data.get('percentage') if isinstance(data, dict) else data['percentage']
                symbols.append({'symbol': symbol, 'volume': vol if vol is not None else 0.0, 'change': ch if ch is not None else 0.0})
        except Exception:
            continue
    symbols.sort(key=lambda x: x['volume'] or 0.0, reverse=True)
    return symbols[:TOP_N]

# ─── گرفتن داده OHLCV به صورت DataFrame
def get_ohlcv_df(symbol, timeframe, limit=200):
    try:
        ohlcv = safe_fetch_ohlcv(symbol, timeframe, limit)
        if not ohlcv:
            return pd.DataFrame()
        df = pd.DataFrame(ohlcv, columns=['timestamp','open','high','low','close','volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        return df
    except Exception as e:
        print(f"[ERROR] fetch_ohlcv {symbol} {timeframe}: {e}")
        return pd.DataFrame()

# ─── اندیکاتورها (نسخهٔ قبلی + EMA50/EMA200 و MACD_HIST)
def calculate_indicators(df):
    if df is None or len(df) < 60:
        return df

    df['EMA9'] = df['close'].ewm(span=9, adjust=False).mean()
    df['EMA21'] = df['close'].ewm(span=21, adjust=False).mean()
    df['EMA50'] = df['close'].ewm(span=50, adjust=False).mean()
    df['EMA200'] = df['close'].ewm(span=200, adjust=False).mean()

    df['MACD'] = df['close'].ewm(span=12, adjust=False).mean() - df['close'].ewm(span=26, adjust=False).mean()
    df['Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_HIST'] = df['MACD'] - df['Signal']

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
    df['+DM'] = np.where((df['high'].diff() > df['low'].diff()) & (df['high'].diff() > 0), df['high'].diff(), 0)
    df['-DM'] = np.where((df['low'].diff() > df['high'].diff()) & (df['low'].diff() > 0), df['low'].diff(), 0)
    atr14 = df['ATR14'].replace(0, np.nan)
    df['+DI'] = 100 * (df['+DM'].ewm(alpha=1/14).mean() / (atr14))
    df['-DI'] = 100 * (df['-DM'].ewm(alpha=1/14).mean() / (atr14))
    df['DX'] = (abs(df['+DI'] - df['-DI']) / (df['+DI'] + df['-DI'] + 1e-9)) * 100
    df['ADX'] = df['DX'].ewm(alpha=1/14).mean()

    # SuperTrend
    factor = 3
    hl2 = (df['high'] + df['low']) / 2
    df['UpperBand'] = hl2 + (factor * df['ATR14'])
    df['LowerBand'] = hl2 - (factor * df['ATR14'])
    df['SuperTrend'] = np.where(df['close'] > df['UpperBand'], 1, np.where(df['close'] < df['LowerBand'], -1, 0))

    # اندیکاتورهای جدید: Pivot, OBV, VWAP, Fibonacci
    df = calculate_pivot_points(df)
    df = calculate_obv(df)
    df = calculate_vwap(df)
    df = calculate_fibonacci(df)

    # پاکسازی مقادیر Inf/NaN
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    return df
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
        for k in ['fib_0','fib_0.236','fib_0.382','fib_0.5','fib_0.618','fib_0.786','fib_1','fib_1.618','fib_2.618']:
            df[k] = np.nan
    return df

# ─── شناسایی کندل‌ها
def detect_candlestick_patterns(df):
    if df is None or len(df) < 3:
        return []
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
    if abs(close - open_) / (high - low + 1e-9) < 0.1:
        patterns.append('Doji')

    return patterns
# ===== واگرایی خودکار (RSI/MACD) =====

def find_local_extrema(series, order=3, kind='min'):
    # ساده و دقیق برای پیدا کردن swing points
    idx = []
    N = len(series)
    for i in range(order, N - order):
        window = series.iloc[i - order: i + order + 1]
        if kind == 'min' and series.iloc[i] == window.min():
            idx.append(i)
        if kind == 'max' and series.iloc[i] == window.max():
            idx.append(i)
    return idx


def detect_divergence(df, indicator='RSI', lookback=DIVERGENCE_LOOKBACK, order=DIVERGENCE_ORDER):
    try:
        if df is None or len(df) < lookback:
            return None
        price = df['close'].iloc[-lookback:]
        ind = df[indicator].iloc[-lookback:]

        # پیدا کردن دو سوئینگ آخر
        lows = find_local_extrema(price, order=order, kind='min')
        highs = find_local_extrema(price, order=order, kind='max')
        # انتقال ایندکس‌ها به ایندکس‌های اصلی
        lows = [i + (len(df) - lookback) for i in lows]
        highs = [i + (len(df) - lookback) for i in highs]

        # بررسی واگرایی صعودی: دو کف اخیر قیمت کف پایین‌تر بزند و اندیکاتور کف بالاتر
        if len(lows) >= 2:
            i1, i2 = lows[-2], lows[-1]
            p1, p2 = df['close'].iloc[i1], df['close'].iloc[i2]
            ind1, ind2 = df[indicator].iloc[i1], df[indicator].iloc[i2]
            if p2 < p1 and ind2 > ind1:
                return {'type': 'bullish', 'indicator': indicator, 'p1_idx': i1, 'p2_idx': i2}

        # بررسی واگرایی نزولی: دو سقف اخیر قیمت سقف بالاتر و اندیکاتور سقف پایین‌تر
        if len(highs) >= 2:
            i1, i2 = highs[-2], highs[-1]
            p1, p2 = df['close'].iloc[i1], df['close'].iloc[i2]
            ind1, ind2 = df[indicator].iloc[i1], df[indicator].iloc[i2]
            if p2 > p1 and ind2 < ind1:
                return {'type': 'bearish', 'indicator': indicator, 'p1_idx': i1, 'p2_idx': i2}

        return None
    except Exception as e:
        print(f"[WARN] detect_divergence failed: {e}")
        return None
# ===== بررسی تایم‌فریم بالاتر برای تأیید =====

def confirm_high_tf(symbol, tf_low, required_type, high_tfs=HIGH_TFS):
    # برای افزایش دقت، تایم‌فریم بالاتر را فراخوانی کرده و بررسی می‌کنیم
    try:
        for htf in high_tfs:
            ohlcv = safe_fetch_ohlcv(symbol, htf, limit=200)
            if not ohlcv:
                continue
            df_ht = pd.DataFrame(ohlcv, columns=['timestamp','open','high','low','close','volume'])
            df_ht = calculate_indicators(df_ht)
            if df_ht is None or len(df_ht) < 60:
                continue
            price = df_ht['close'].iloc[-1]
            ema9 = df_ht['EMA9'].iloc[-1]
            ema21 = df_ht['EMA21'].iloc[-1]
            macd = df_ht['MACD'].iloc[-1]
            signal = df_ht['Signal'].iloc[-1]
            rsi = df_ht['RSI'].iloc[-1]
            ema200 = df_ht['EMA200'].iloc[-1] if 'EMA200' in df_ht.columns else None

            if required_type == 'LONG':
                cond = (ema9 > ema21) and (macd > signal) and (rsi > 48)
                if ema200 is not None:
                    cond = cond and (price > ema200)
                if cond:
                    return True
            else:  # SHORT
                cond = (ema9 < ema21) and (macd < signal) and (rsi < 52)
                if ema200 is not None:
                    cond = cond and (price < ema200)
                if cond:
                    return True
            # کوتاه کردن تماس‌ها در برابر ریت‌لیمیت
            time.sleep(exchange.rateLimit / 1000)
    except Exception as e:
        print(f"[WARN] confirm_high_tf failed for {symbol}: {e}")
    return False

# ===== محاسبهٔ اسپایک حجم با baseline ساده (EMA) =====

def update_volume_baseline(symbol, current_vol):
    if symbol not in volume_baseline:
        volume_baseline[symbol] = float(current_vol or 0.0)
        return volume_baseline[symbol]
    prev = volume_baseline[symbol]
    new = VOLUME_BASELINE_ALPHA * float(current_vol or 0.0) + (1 - VOLUME_BASELINE_ALPHA) * prev
    volume_baseline[symbol] = new
    return new


def detect_volume_spike(symbol, current_vol):
    """
    نسخهٔ دقیق‌تر تشخیص اسپایک حجم:
    1) یک چک سریع نسبت به baseline EMA انجام میده (lightweight).
    2) اگر ratio بالا بود، یک بررسی دقیق‌تر با دادهٔ 1h انجام میده: z-score آخرین کندل حجمی نسبت به 24 کندلِ قبلی و نسبت last24/prev24.
    3) از cooldown برای جلوگیری از اعلان‌های مکرر استفاده میشه.
    """
    try:
        if current_vol is None:
            return False

        # baseline سریع
        baseline = volume_baseline.get(symbol)
        if baseline is None:
            update_volume_baseline(symbol, current_vol)
            return False
        ratio = float(current_vol) / (baseline + 1e-9)

        # شرط سریع اولیه
        if not (current_vol >= VOLUME_MIN_ABS and ratio >= VOLUME_SPIKE_FACTOR):
            # اگر اسپایک نبود، به‌روز‌رسانی baseline و خروج
            update_volume_baseline(symbol, current_vol)
            return False

        # cooldown
        last = volume_last_alert.get(symbol, 0)
        if time.time() - last < ANOMALY_COOLDOWN:
            return False

        # بررسی دقیق‌تر با OHLCV 1h (آخرین 48 ساعت)
        try:
            ohlcv = safe_fetch_ohlcv(symbol, '1h', limit=48)
            if not ohlcv:
                # اگر نتونستیم دیتای دقیق بگیریم، باز هم می‌تونیم براساس ratio اعتماد کنیم
                volume_last_alert[symbol] = time.time()
                return True

            df_1h = pd.DataFrame(ohlcv, columns=['ts','open','high','low','close','volume'])
            vols = df_1h['volume'].astype(float)

            if len(vols) >= 24:
                last24 = vols[-24:].sum()
                prev24 = vols[-48:-24].sum() if len(vols) >= 48 else None
                # z-score بر اساس 24 کندل اخیر (بدون آخرین کندل)
                window = vols[-25:-1] if len(vols) >= 25 else vols[:-1]
                mean = window.mean() if len(window) > 0 else vols.mean()
                std = window.std(ddof=0) if len(window) > 0 else vols.std(ddof=0)
                z = (vols.iloc[-1] - mean) / (std + 1e-9)

                # قضاوت نهایی: نسبت last24/prev24 و z-score باید بزرگ باشن
                if prev24 and prev24 > 0 and last24 / (prev24 + 1e-9) >= VOLUME_SPIKE_FACTOR and z >= VOLUME_ZSCORE_THRESH:
                    volume_last_alert[symbol] = time.time()
                    return True
                else:
                    # اگر شرایط دقیق برقرار نبود، baseline را به‌روزرسانی کن و رد کن
                    update_volume_baseline(symbol, current_vol)
                    return False
            else:
                # اگر داده کافی نیست، fallback به ratio
                volume_last_alert[symbol] = time.time()
                return True
        except Exception as e:
            print(f"[WARN] detailed volume check failed for {symbol}: {e}")
            # fallback: اگر چک دقیق نشد، اما ratio بالا بود، اعلام کن (قابل تنظیم)
            volume_last_alert[symbol] = time.time()
            return True

    except Exception as e:
        print(f"[WARN] detect_volume_spike error {symbol}: {e}")
        return False
# ===== تابع امتیازدهی (همون قبلی بدون تغییر منطقی) =====
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

        stars = []
        vol_mean = df['volume'].rolling(20).mean().iloc[-1]
        if not pd.isna(vol_mean) and df['volume'].iloc[-1] > vol_mean * 1.5:
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

        # شروط ورود (سفت‌تر)
        if (intrabar_change >= 0.2 and change >= 0.2 and trend == 'bullish' and len(stars) >= 2
            and df['EMA9'].iloc[-1] > df['EMA21'].iloc[-1] and df['RSI'].iloc[-1] > 50):
            signal_type = 'LONG'
            entry = price
            stop = price - 1.2 * atr
            tp = price + 1.8 * atr

        elif (intrabar_change <= -0.2 and change <= -0.2 and trend == 'bearish' and len(stars) >= 2
              and df['EMA9'].iloc[-1] < df['EMA21'].iloc[-1] and df['RSI'].iloc[-1] < 50):
            signal_type = 'SHORT'
            entry = price
            stop = price + 1.2 * atr
            tp = price - 1.8 * atr

        if signal_type and entry and stop:
            size = calculate_position_size(entry, stop)

        if not signal_type:
            return None

        temp_sig = {'entry': entry, 'tp': tp, 'stop': stop, 'type': signal_type, 'patterns': patterns, 'stars': stars, 'size': size}
        score = compute_signal_score(temp_sig, df, intrabar_change)
        temp_sig['score'] = score

        # اضافه کردن واگرایی و تایید TF بالا به سیگنال برای later filtering
        div_rsi = detect_divergence(df, indicator='RSI')
        div_macd = detect_divergence(df, indicator='MACD')
        temp_sig['divergence'] = {'RSI': div_rsi, 'MACD': div_macd}

        # strength label
        if score >= (MIN_SCORE * 2):
            temp_sig['strength'] = 'strong'
        elif score >= MIN_SCORE:
            temp_sig['strength'] = 'normal'
        else:
            temp_sig['strength'] = 'weak'

        prev = last_alerts.get(symbol)
        if prev and prev.get('type') == signal_type and (time.time() - prev.get('time', 0) < SIGNAL_INTERVAL):
            return None
        last_alerts[symbol] = {'type': signal_type, 'time': time.time()}

        return temp_sig

    except Exception as e:
        print(f"[ERROR] check_signal {symbol}: {e}")
        return None
# ===== main loop: جمع‌آوری کاندیدها + فیلتر پیشرفته + اسپایک حجم =====
def main():
    print("🚀 ربات شروع شد — با تایید TF بالا و واگرایی")
    while True:
        try:
            top_symbols = get_top_symbols()
            candidates = []

            # مرحلهٔ اول: بررسی نمادها و جمع‌آوری کاندیدها
            for symbol_data in top_symbols:
                symbol = symbol_data['symbol']
                change = symbol_data.get('change', 0.0)
                # بروزرسانی baseline حجم سریع با مقدار 24h از tickers
                current_vol = symbol_data.get('volume', 0.0)
                # detect volume spikes (lightweight)
                spike = detect_volume_spike(symbol, current_vol)

                for tf in TIMEFRAMES:
                    try:
                        df = get_ohlcv_df(symbol, tf)
                        if df is None or df.empty:
                            continue
                        df = calculate_indicators(df)
                        sig = check_signal(df, symbol, change)
                        if sig:
                            # ذخیره‌ی چند فیلد اضافه برای بررسی بعدی
                            candidates.append({'symbol': symbol, 'tf': tf, 'signal': sig, 'score': sig.get('score', 0.0), 'df': df, 'volume_spike': spike})
                    except Exception as e:
                        print(f"[ERROR] {symbol} | TF: {tf} | {e}")
                        continue

            print(f"[INFO] Found {len(candidates)} raw candidates this cycle.")

            # فیلتر اولیه بر اساس MIN_SCORE
            filtered = [c for c in candidates if c['score'] >= MIN_SCORE]
            print(f"[INFO] {len(filtered)} candidates passed MIN_SCORE >= {MIN_SCORE}")

            # مرتب‌سازی
            filtered.sort(key=lambda x: x['score'], reverse=True)

            # فیلتر دقیق‌تر: واگرایی یا تایید TF بالا برای تایم پایین
            final_candidates = []
            used_symbols = set()
            for c in filtered:
                if len(final_candidates) >= SIGNALS_PER_CYCLE:
                    break
                sym = c['symbol']
                tf = c['tf']
                sig = c['signal']
                df = c['df']

                # cooldown per symbol
                last_t = last_signal_time.get(sym, 0)
                if time.time() - last_t < SIGNAL_INTERVAL:
                    continue
                if sym in used_symbols:
                    continue

                # اگر تایم پایینه، نیاز به تایید داریم
                if tf in LOW_TF_TO_REQUIRE_HIGH_CONFIRM:
                    confirmed = confirm_high_tf(sym, tf, sig['type'])
                    div_ok = (sig.get('divergence', {})['RSI'] is not None) or (sig.get('divergence', {})['MACD'] is not None)
                    if REQUIRE_DIVERGENCE:
                        if not (confirmed or div_ok):
                            # رد کن چون نه واگرایی هست نه تایید TF بالا
                            continue
                    else:
                        if not confirmed:
                            continue

                # اضافه کردن نهایی
                final_candidates.append(c)
                used_symbols.add(sym)
                last_signal_time[sym] = time.time()

            print(f"[INFO] Selected {len(final_candidates)} signals to send (max {SIGNALS_PER_CYCLE}).")

            # ارسال پیام‌ها — اگر حجم غیرعادی بوده و سیگنال هست، علامت ویژه میزنیم
            if final_candidates:
                for c in final_candidates:
                    s = c['signal']
                    sym = c['symbol']
                    tf = c['tf']
                    spike = c.get('volume_spike', False)
                    color_emoji = "🟢" if s['type'] == "LONG" else "🔴"
                    strength_tag = " 🔥" if s.get('strength') == 'strong' else (" ⭐" if s.get('strength') == 'normal' else "")
                    vol_tag = " 📈VOLSpike" if spike else ""
                    now_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    msg = (f"🚨 Multi-Coin Alert 🚨\n"
                           f"{color_emoji} {sym} | TF: {tf}{strength_tag}{vol_tag}\n"
                           f"Type: {s['type']}\n"
                           f"Entry: {s['entry']:.6f}\n"
                           f"TP: {s['tp']:.6f}\n"
                           f"Stop: {s['stop']:.6f}\n"
                           f"Size: {s['size']}\n"
                           f"Score: {s.get('score', 0.0)}\n"
                           f"Patterns: {s['patterns']}\n"
                           f"Divergence: {s.get('divergence')}\n"
                           f"🕒 Time: {now_time}")
                    try:
                        bot.send_message(chat_id=CHAT_ID, text=msg)
                        print(f"[SENT] {sym} | TF:{tf} | Score:{s.get('score',0)} | VOLSPIKE={spike}")
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
