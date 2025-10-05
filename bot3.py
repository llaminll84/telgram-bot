# paste this entire file, replace your current bot file
import time
import os
import ccxt
import pandas as pd
import numpy as np
import datetime
import logging
from collections import deque
from telegram import Bot
from keep_alive import keep_alive  # سرور کوچک برای جلوگیری از خوابیدن کانتینر

# ─── تنظیمات لاگ
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')

# ─── سرور کوچک
keep_alive()

# ─── اطلاعات ربات تلگرام (در صورت نبودن، ارسال تلگرام غیرفعال می‌شود)
TELEGRAM_TOKEN = os.getenv("BOT_TOKEN")
CHAT_ID = os.getenv("CHAT_ID")
if TELEGRAM_TOKEN and CHAT_ID:
    bot = Bot(token=TELEGRAM_TOKEN)
else:
    bot = None
    logging.warning("BOT_TOKEN یا CHAT_ID تنظیم نشده — ارسال پیام تلگرام غیرفعال است.")

# ─── صرافی (rate limit فعال)
exchange = ccxt.kucoin({'enableRateLimit': True})

# ─── ====== CONFIG: اینها رو میتونی تغییر بدی ======
TOP_N = 200
TIMEFRAMES = ['5m','15m','30m','1h','4h']
LOW_TF_TO_REQUIRE_HIGH_CONFIRM = ['5m','15m']
HIGH_TFS = ['1h','4h']
SIGNALS_PER_CYCLE = 2
MIN_SCORE = 20.0
MIN_SCORE_HIGH_CONFIRM = 13.0
SEND_DELAY_BETWEEN_MSGS = 1.0
SIGNAL_INTERVAL = 5 * 60

# دقت اضافه: سخت‌گیری‌های اختیاری
REQUIRE_DIVERGENCE = True
DIVERGENCE_LOOKBACK = 60
DIVERGENCE_ORDER = 3
VOLUME_SPIKE_FACTOR = 2.0
VOLUME_BASELINE_ALPHA = 0.15
VOLUME_MIN_ABS = 50.0
ANOMALY_COOLDOWN = 60 * 60
VOLUME_ZSCORE_THRESH = 2.0
FAST_FILTER_CHANGE = 0.25

# فیلترهای جدید برای دقت بالا
CANDLE_BODY_MIN = 0.45
CLOSE_POS_THRESHOLD = 0.65
VOLUME_MULTIPLIER_CONFIRM = 1.3
VOLATILITY_HIGH_THRESHOLD = 0.008
REQUIRED_CONFIRMS_LOWVOL = 2
REQUIRED_CONFIRMS_HIGHVOL = 3
MONITOR_AFTER_SEND = False
MONITOR_CANDLES = 6
# ───────────────────────────────────────────────────

last_signal_time = {}
last_alerts = {}
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

# ─── توابع امن fetch
def safe_fetch_tickers():
    for i in range(3):
        try:
            res = exchange.fetch_tickers()
            try:
                time.sleep(exchange.rateLimit / 1000)
            except Exception:
                pass
            return res
        except Exception as e:
            logging.warning(f"fetch_tickers failed (retry {i+1}): {e}")
            time.sleep(1 + i * 2)
    return {}

def safe_fetch_ohlcv(symbol, timeframe, limit=200):
    for i in range(3):
        try:
            data = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
            try:
                time.sleep(exchange.rateLimit / 1000)
            except Exception:
                pass
            return data
        except Exception as e:
            logging.warning(f"fetch_ohlcv {symbol} {timeframe} failed (retry {i+1}): {e}")
            time.sleep(1 + i * 2)
    return None

# ─── گرفتن TOP symbols با اطلاعات حجم 24h (مقاوم‌تر)
def get_top_symbols():
    tickers = safe_fetch_tickers()
    symbols = []
    for symbol, data in tickers.items():
        try:
            if not isinstance(data, dict):
                continue
            if not symbol.endswith('/USDT'):
                continue
            vol = data.get('quoteVolume') or data.get('baseVolume') or 0.0
            ch = data.get('percentage') or data.get('change') or 0.0
            symbols.append({'symbol': symbol, 'volume': float(vol or 0.0), 'change': float(ch or 0.0)})
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
        logging.error(f"fetch_ohlcv {symbol} {timeframe}: {e}")
        return pd.DataFrame()

# ─── اندیکاتورها
def calculate_indicators(df):
    if df is None or len(df) < 60:
        return df

    df = df.copy()
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
    df['+DI'] = 100 * (pd.Series(df['+DM']).ewm(alpha=1/14).mean() / (atr14))
    df['-DI'] = 100 * (pd.Series(df['-DM']).ewm(alpha=1/14).mean() / (atr14))
    df['DX'] = (abs(df['+DI'] - df['-DI']) / (df['+DI'] + df['-DI'] + 1e-9)) * 100
    df['ADX'] = df['DX'].ewm(alpha=1/14).mean()

    # SuperTrend (ساده)
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

    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    return df

# ─── اندیکاتورهای تکمیلی
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
    prev_open, prev_close = df['open'].iloc[-2], df['close'].iloc[-2']

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
        lows = find_local_extrema(price, order=order, kind='min')
        highs = find_local_extrema(price, order=order, kind='max')
        lows = [i + (len(df) - lookback) for i in lows]
        highs = [i + (len(df) - lookback) for i in highs]
        if len(lows) >= 2:
            i1, i2 = lows[-2], lows[-1]
            p1, p2 = df['close'].iloc[i1], df['close'].iloc[i2]
            ind1, ind2 = df[indicator].iloc[i1], df[indicator].iloc[i2]
            if p2 < p1 and ind2 > ind1:
                return {'type': 'bullish', 'indicator': indicator, 'p1_idx': i1, 'p2_idx': i2}
        if len(highs) >= 2:
            i1, i2 = highs[-2], highs[-1]
            p1, p2 = df['close'].iloc[i1], df['close'].iloc[i2]
            ind1, ind2 = df[indicator].iloc[i1], df[indicator].iloc[i2]
            if p2 > p1 and ind2 < ind1:
                return {'type': 'bearish', 'indicator': indicator, 'p1_idx': i1, 'p2_idx': i2}
        return None
    except Exception as e:
        logging.warning(f"detect_divergence failed: {e}")
        return None

# ===== بررسی تایم‌فریم بالاتر برای تأیید =====
def confirm_high_tf(symbol, tf_low, required_type, high_tfs=HIGH_TFS):
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
            else:
                cond = (ema9 < ema21) and (macd < signal) and (rsi < 52)
                if ema200 is not None:
                    cond = cond and (price < ema200)
                if cond:
                    return True
            try:
                time.sleep(exchange.rateLimit / 1000)
            except Exception:
                pass
    except Exception as e:
        logging.warning(f"confirm_high_tf failed for {symbol}: {e}")
    return False

# ===== محاسبهٔ اسپایک حجم با baseline ساده (EMA)
def update_volume_baseline(symbol, current_vol):
    if symbol not in volume_baseline:
        volume_baseline[symbol] = float(current_vol or 0.0)
        return volume_baseline[symbol]
    prev = volume_baseline[symbol]
    new = VOLUME_BASELINE_ALPHA * float(current_vol or 0.0) + (1 - VOLUME_BASELINE_ALPHA) * prev
    volume_baseline[symbol] = new
    return new

def detect_volume_spike_light(symbol, current_vol):
    try:
        if current_vol is None:
            return False
        baseline = volume_baseline.get(symbol)
        if baseline is None:
            update_volume_baseline(symbol, current_vol)
            return False
        ratio = float(current_vol) / (baseline + 1e-9)
        if not (current_vol >= VOLUME_MIN_ABS and ratio >= VOLUME_SPIKE_FACTOR):
            update_volume_baseline(symbol, current_vol)
            return False
        last = volume_last_alert.get(symbol, 0)
        if time.time() - last < ANOMALY_COOLDOWN:
            return False
        volume_last_alert[symbol] = time.time()
        return True
    except Exception as e:
        logging.warning(f"detect_volume_spike_light error {symbol}: {e}")
        return False

def detect_volume_spike_detailed(symbol, current_vol):
    try:
        if current_vol is None:
            return False
        baseline = volume_baseline.get(symbol)
        if baseline is None:
            update_volume_baseline(symbol, current_vol)
            return False
        ratio = float(current_vol) / (baseline + 1e-9)
        if not (current_vol >= VOLUME_MIN_ABS and ratio >= VOLUME_SPIKE_FACTOR):
            update_volume_baseline(symbol, current_vol)
            return False
        last = volume_last_alert.get(symbol, 0)
        if time.time() - last < ANOMALY_COOLDOWN:
            return False
        ohlcv = safe_fetch_ohlcv(symbol, '1h', limit=48)
        if not ohlcv:
            volume_last_alert[symbol] = time.time()
            return True
        df_1h = pd.DataFrame(ohlcv, columns=['ts','open','high','low','close','volume'])
        vols = df_1h['volume'].astype(float)
        if len(vols) >= 24:
            last24 = vols[-24:].sum()
            prev24 = vols[-48:-24].sum() if len(vols) >= 48 else None
            window = vols[-25:-1] if len(vols) >= 25 else vols[:-1]
            mean = window.mean() if len(window) > 0 else vols.mean()
            std = window.std(ddof=0) if len(window) > 0 else vols.std(ddof=0)
            z = (vols.iloc[-1] - mean) / (std + 1e-9)
            if prev24 and prev24 > 0 and last24 / (prev24 + 1e-9) >= VOLUME_SPIKE_FACTOR and z >= VOLUME_ZSCORE_THRESH:
                volume_last_alert[symbol] = time.time()
                return True
            else:
                update_volume_baseline(symbol, current_vol)
                return False
        else:
            volume_last_alert[symbol] = time.time()
            return True
    except Exception as e:
        logging.warning(f"detailed volume check failed for {symbol}: {e}")
        volume_last_alert[symbol] = time.time()
        return True

# === توابع کمکی جدید برای دقت بالا ===
def candle_strength_metric(df):
    try:
        last = df.iloc[-1]
        body = abs(last['close'] - last['open'])
        rng = (last['high'] - last['low']) + 1e-9
        body_ratio = body / rng
        close_pos = (last['close'] - last['low']) / (rng + 1e-9)
        if last['close'] > last['open']:
            score = 0.0
            if body_ratio >= CANDLE_BODY_MIN and close_pos >= CLOSE_POS_THRESHOLD:
                score = 1.0
            elif body_ratio >= 0.3 and close_pos >= 0.55:
                score = 0.6
            elif body_ratio >= 0.2:
                score = 0.3
            return score, 'bull'
        elif last['close'] < last['open']:
            score = 0.0
            if body_ratio >= CANDLE_BODY_MIN and close_pos <= (1 - CLOSE_POS_THRESHOLD):
                score = 1.0
            elif body_ratio >= 0.3 and close_pos <= 0.45:
                score = 0.6
            elif body_ratio >= 0.2:
                score = 0.3
            return score, 'bear'
        else:
            return 0.0, 'neutral'
    except Exception:
        return 0.0, 'neutral'

def volume_confirmation(df):
    try:
        vol = df['volume'].iloc[-1]
        mean20 = df['volume'].rolling(20).mean().iloc[-1]
        if pd.isna(mean20) or mean20 == 0:
            return False, 1.0
        rel = vol / (mean20 + 1e-9)
        return (vol >= VOLUME_MIN_ABS and rel >= VOLUME_MULTIPLIER_CONFIRM), rel
    except Exception:
        return False, 1.0

def atr_volatility(df):
    try:
        atr = df['ATR14'].iloc[-1]
        price = df['close'].iloc[-1]
        if price == 0:
            return 0.0
        return (atr / price)
    except Exception:
        return 0.0

# compute_signal_score (تقویت‌شده)
def compute_signal_score(sig, df, intrabar_change):
    try:
        base = 0.0
        stars_count = len(sig.get('stars', []))
        base += stars_count * 6.0
        adx = float(df['ADX'].iloc[-1]) if 'ADX' in df.columns and not pd.isna(df['ADX'].iloc[-1]) else 0.0
        base += min(adx, 50.0) * 0.6
        base += min(abs(intrabar_change) * 6.0, 25.0)
        vol_mean = float(df['volume'].rolling(20).mean().iloc[-1]) if 'volume' in df.columns else np.nan
        vol_rel = 1.0
        if not pd.isna(vol_mean) and vol_mean > 0:
            vol_rel = float(df['volume'].iloc[-1]) / vol_mean
            if vol_rel > 1.0:
                base += (vol_rel - 1.0) * 8.0
        macd_hist = float(df['MACD_HIST'].iloc[-1]) if 'MACD_HIST' in df.columns and not pd.isna(df['MACD_HIST'].iloc[-1]) else 0.0
        base += min(abs(macd_hist) * 8.0, 20.0)
        ema9 = float(df['EMA9'].iloc[-1]) if 'EMA9' in df.columns and not pd.isna(df['EMA9'].iloc[-1]) else 0.0
        ema21 = float(df['EMA21'].iloc[-1]) if 'EMA21' in df.columns and not pd.isna(df['EMA21'].iloc[-1]) else 1.0
        ema_diff = (ema9 - ema21) / (ema21 if ema21 != 0 else 1e-9)
        if sig.get('type') == 'LONG':
            base += max(0.0, ema_diff * 120.0)
        else:
            base += max(0.0, -ema_diff * 120.0)
        cs, ctype = candle_strength_metric(df)
        base += cs * 10.0
        if 'VWAP' in df.columns and not pd.isna(df['VWAP'].iloc[-1]):
            if (sig.get('type') == 'LONG' and df['close'].iloc[-1] > df['VWAP'].iloc[-1]) or \
               (sig.get('type') == 'SHORT' and df['close'].iloc[-1] < df['VWAP'].iloc[-1]):
                base += 4.0
        rsi = float(df['RSI'].iloc[-1]) if 'RSI' in df.columns and not pd.isna(df['RSI'].iloc[-1]) else 50.0
        if rsi > 85:
            base -= (rsi - 85) * 1.0
        if rsi < 15:
            base -= (15 - rsi) * 1.0
        return round(base, 2)
    except Exception:
        return 0.0

# ===== بازنویسی check_signal با منطق قوی تر و لاگ دقیق =====
def check_signal(df, symbol, change):
    try:
        if df is None or len(df) < 60:
            return None
        needed = ['EMA9','EMA21','ATR14','RSI','ADX','volume']
        if any(col not in df.columns or pd.isna(df[col].iloc[-1]) for col in needed):
            return None
        price = df['close'].iloc[-1]
        prev_price = df['close'].iloc[-2]
        intrabar_change = ((price - prev_price) / (prev_price + 1e-9)) * 100.0
        trend = 'neutral'
        try:
            if not pd.isna(df['SenkouA'].iloc[-1]) and not pd.isna(df['SenkouB'].iloc[-1]):
                if price > df['SenkouA'].iloc[-1] and price > df['SenkouB'].iloc[-1]:
                    trend = 'bullish'
                elif price < df['SenkouA'].iloc[-1] and price < df['SenkouB'].iloc[-1]:
                    trend = 'bearish'
        except Exception:
            trend = 'neutral'
        patterns = detect_candlestick_patterns(df)
        confs = {}
        ema9 = df['EMA9'].iloc[-1]; ema21 = df['EMA21'].iloc[-1]
        ema_cross_long = (ema9 > ema21)
        ema_cross_short = (ema9 < ema21)
        confs['ema_cross'] = ema_cross_long or ema_cross_short
        rsi = df['RSI'].iloc[-1]
        confs['rsi_long'] = rsi > 55
        confs['rsi_short'] = rsi < 45
        macd_hist = df['MACD_HIST'].iloc[-1] if 'MACD_HIST' in df.columns else (df['MACD'].iloc[-1] - df['Signal'].iloc[-1])
        confs['macd_long'] = macd_hist > 0
        confs['macd_short'] = macd_hist < 0
        cs_score, cs_type = candle_strength_metric(df)
        confs['candle_strong'] = cs_score >= 0.6
        vol_ok, vol_rel = volume_confirmation(df)
        confs['volume_spike'] = vol_ok
        confs['above_vwap'] = False
        if 'VWAP' in df.columns and not pd.isna(df['VWAP'].iloc[-1]):
            confs['above_vwap'] = df['close'].iloc[-1] > df['VWAP'].iloc[-1]
        vol_ratio = atr_volatility(df)
        required_confirms = REQUIRED_CONFIRMS_HIGHVOL if vol_ratio >= VOLATILITY_HIGH_THRESHOLD else REQUIRED_CONFIRMS_LOWVOL
        candidate_type = None
        if ema_cross_long and (trend == 'bullish' or rsi > 50):
            candidate_type = 'LONG'
        elif ema_cross_short and (trend == 'bearish' or rsi < 50):
            candidate_type = 'SHORT'
        else:
            return None
        confirms = 0
        confirm_reasons = []
        if candidate_type == 'LONG':
            for k in ['rsi_long','macd_long','candle_strong','volume_spike','above_vwap']:
                if confs.get(k):
                    confirms += 1
                    confirm_reasons.append(k)
        else:
            for k in ['rsi_short','macd_short','candle_strong','volume_spike','above_vwap']:
                if confs.get(k):
                    confirms += 1
                    confirm_reasons.append(k)
        if confirms < required_confirms:
            logging.debug(f"{symbol} skipped: confirms={confirms} < required {required_confirms} | reasons={confirm_reasons}")
            return None
        atr = df['ATR14'].iloc[-1] if 'ATR14' in df.columns else 0.0
        if atr <= 0 or pd.isna(atr):
            return None
        entry = price
        if candidate_type == 'LONG':
            stop = price - 1.0 * atr
            tp = price + 1.5 * atr
        else:
            stop = price + 1.0 * atr
            tp = price - 1.5 * atr
        size = calculate_position_size(entry, stop)
        temp_sig = {'entry': entry, 'tp': tp, 'stop': stop, 'type': candidate_type,
                    'patterns': patterns, 'stars': confirm_reasons, 'size': size}
        score = compute_signal_score(temp_sig, df, intrabar_change)
        temp_sig['score'] = score
        temp_sig['confirm_reasons'] = confirm_reasons
        temp_sig['volatility'] = vol_ratio
        div_rsi = detect_divergence(df, indicator='RSI')
        div_macd = detect_divergence(df, indicator='MACD')
        temp_sig['divergence'] = {'RSI': div_rsi, 'MACD': div_macd}
        if score >= (MIN_SCORE * 1.8):
            temp_sig['strength'] = 'strong'
        elif score >= MIN_SCORE:
            temp_sig['strength'] = 'normal'
        else:
            temp_sig['strength'] = 'weak'
        if temp_sig['score'] < MIN_SCORE:
            logging.debug(f"{symbol} score {temp_sig['score']} below MIN_SCORE {MIN_SCORE}")
            return None
        prev = last_alerts.get(symbol)
        if prev and prev.get('type') == candidate_type and (time.time() - prev.get('time', 0) < SIGNAL_INTERVAL):
            return None
        last_alerts[symbol] = {'type': candidate_type, 'time': time.time()}
        logging.info(f"[SIG_CAND] {symbol} | Type={candidate_type} | Score={score} | Confirms={confirms}/{required_confirms} | Reasons={confirm_reasons} | Vol={vol_ratio:.4f}")
        return temp_sig
    except Exception as e:
        logging.error(f"check_signal {symbol}: {e}")
        return None

# ===== main loop =====
def main():
    logging.info("🚀 ربات شروع شد — نسخهٔ با دقت بالا")
    while True:
        try:
            top_symbols = get_top_symbols()
            candidates = []
            for symbol_data in top_symbols:
                symbol = symbol_data['symbol']
                change = symbol_data.get('change', 0.0)
                current_vol = symbol_data.get('volume', 0.0)
                if abs(change) < FAST_FILTER_CHANGE:
                    continue
                high_confirmed = False
                for htf in HIGH_TFS:
                    df_ht = get_ohlcv_df(symbol, htf)
                    if df_ht is None or df_ht.empty:
                        continue
                    df_ht = calculate_indicators(df_ht)
                    sig_ht = check_signal(df_ht, symbol, change)
                    if sig_ht:
                        candidates.append({'symbol': symbol, 'tf': htf, 'signal': sig_ht, 'score': sig_ht.get('score', 0.0), 'df': df_ht, '24h_volume': current_vol})
                        high_confirmed = True
                if not high_confirmed:
                    continue
                for tf in [t for t in TIMEFRAMES if t not in HIGH_TFS]:
                    df = get_ohlcv_df(symbol, tf)
                    if df is None or df.empty:
                        continue
                    df = calculate_indicators(df)
                    sig = check_signal(df, symbol, change)
                    if sig:
                        candidates.append({'symbol': symbol, 'tf': tf, 'signal': sig, 'score': sig.get('score', 0.0), 'df': df, '24h_volume': current_vol})
            logging.info(f"[INFO] Found {len(candidates)} raw candidates this cycle.")
            filtered = [c for c in candidates if c['score'] >= MIN_SCORE]
            logging.info(f"[INFO] {len(filtered)} candidates passed MIN_SCORE >= {MIN_SCORE}")
            filtered.sort(key=lambda x: x['score'], reverse=True)
            final_candidates = []
            used_symbols = set()
            for c in filtered:
                if len(final_candidates) >= SIGNALS_PER_CYCLE:
                    break
                sym = c['symbol']
                tf = c['tf']
                sig = c['signal']
                df = c['df']
                last_t = last_signal_time.get(sym, 0)
                if time.time() - last_t < SIGNAL_INTERVAL:
                    continue
                if sym in used_symbols:
                    continue
                if tf in LOW_TF_TO_REQUIRE_HIGH_CONFIRM:
                    confirmed = confirm_high_tf(sym, tf, sig['type'])
                    div_ok = (sig.get('divergence', {}) is not None and (sig.get('divergence', {}).get('RSI') or sig.get('divergence', {}).get('MACD')))
                    if REQUIRE_DIVERGENCE:
                        if not (confirmed or div_ok):
                            continue
                    else:
                        if not confirmed:
                            continue
                final_candidates.append(c)
                used_symbols.add(sym)
                last_signal_time[sym] = time.time()
            logging.info(f"[INFO] Selected {len(final_candidates)} signals to send (max {SIGNALS_PER_CYCLE}).")
            if final_candidates:
                for c in final_candidates:
                    sym = c['symbol']
                    vol24 = c.get('24h_volume', 0.0)
                    spike = detect_volume_spike_detailed(sym, vol24)
                    c['volume_spike'] = spike
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
                           f"Confirms: {s.get('confirm_reasons')}\n"
                           f"Volatility: {s.get('volatility'):.4f}\n"
                           f"Divergence: {s.get('divergence')}\n"
                           f"🕒 Time: {now_time}")
                    try:
                        if bot:
                            bot.send_message(chat_id=CHAT_ID, text=msg)
                        logging.info(f"[SENT] {sym} | TF:{tf} | Score:{s.get('score',0)} | VOLSPIKE={spike}")
                    except Exception as e:
                        logging.error(f"sending telegram {sym}: {e}")
                    if MONITOR_AFTER_SEND:
                        try:
                            monitor_signal_outcome(sym, tf, s['entry'], s['tp'], s['stop'], MONITOR_CANDLES)
                        except Exception as e:
                            logging.warning(f"monitoring failed for {sym}: {e}")
                    time.sleep(SEND_DELAY_BETWEEN_MSGS)
            time.sleep(SIGNAL_INTERVAL)
        except Exception as e:
            logging.error(f"خطا در main: {e}")
            time.sleep(30)

# مانیتور کوتاه (اختیاری)
def monitor_signal_outcome(symbol, timeframe, entry, tp, stop, look_candles=6):
    try:
        ohlcv = safe_fetch_ohlcv(symbol, timeframe, limit=look_candles)
        if not ohlcv:
            return None
        dfm = pd.DataFrame(ohlcv, columns=['timestamp','open','high','low','close','volume'])
        hit_tp = False
        hit_stop = False
        for i in range(len(dfm)):
            high = dfm['high'].iloc[i]
            low = dfm['low'].iloc[i]
            if tp and high >= tp:
                hit_tp = True
                break
            if stop and low <= stop:
                hit_stop = True
                break
        logging.info(f"[MONITOR] {symbol} outcome -> TP:{hit_tp} STOP:{hit_stop}")
        return {'tp': hit_tp, 'stop': hit_stop}
    except Exception as e:
        logging.warning(f"monitor_signal_outcome failed: {e}")
        return None

if __name__ == "__main__":
    main()
