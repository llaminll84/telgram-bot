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
exchange = ccxt.kucoin({
    'enableRateLimit': True
})

# ─── ====== CONFIG: اینها رو میتونی تغییر بدی ======
TOP_N = 200
TIMEFRAMES = ['5m','15m','30m','1h','4h']
LOW_TF_TO_REQUIRE_HIGH_CONFIRM = ['5m','15m']   # تایم‌فریم‌های پایین که نیاز به تائید 1h/4h دارن
HIGH_TFS = ['1h','4h']
SIGNALS_PER_CYCLE = 2          # کمتر برای دقت بالاتر
MIN_SCORE = 20.0               # حداقل امتیاز لازم برای ارسال
MIN_SCORE_HIGH_CONFIRM = 13.0
SEND_DELAY_BETWEEN_MSGS = 1.0
SIGNAL_INTERVAL = 5 * 60      # cooldown پایه

# دقت اضافه: سخت‌گیری‌های اختیاری
REQUIRE_DIVERGENCE = True          # اگر True، برای تایم پایین نیاز به واگرایی یا تایید TF بالاتر داریم
DIVERGENCE_LOOKBACK = 60
DIVERGENCE_ORDER = 3               # پارامتر تشخیص swing
VOLUME_SPIKE_FACTOR = 2.0          # نسبت به baseline که به عنوان اسپایک در نظر گرفته میشه
VOLUME_BASELINE_ALPHA = 0.15       # EMA alpha برای بروزرسانی baseline حجم
VOLUME_MIN_ABS = 50.0              # حداقل حجم مطلق
ANOMALY_COOLDOWN = 60 * 60         # یک ساعت
VOLUME_ZSCORE_THRESH = 2.0
FAST_FILTER_CHANGE = 0.25          # درصد تغییر 24h سریع برای اسکن عمیق‌تر

# فیلترهای جدید برای دقت بالا
CANDLE_BODY_MIN = 0.45            # نسبت بدنه کندل به رنج (>= => قوی)
CLOSE_POS_THRESHOLD = 0.65        # بسته شدن در بالای 65% از رنج کندل
VOLUME_MULTIPLIER_CONFIRM = 1.3   # نسبت حجم برای تأیید
VOLATILITY_HIGH_THRESHOLD = 0.008 # ATR14/price نسبت برای تشخیص نوسان بالا (قابل تنظیم)
REQUIRED_CONFIRMS_LOWVOL = 2      # تاییدهای لازم در بازار کم نوسان
REQUIRED_CONFIRMS_HIGHVOL = 3     # تاییدهای لازم در بازار پرنوسان
MONITOR_AFTER_SEND = False        # اگر True، پس از ارسال سیگنال بررسی کوتاه انجام میشه (پیشرفته)
MONITOR_CANDLES = 6               # تعداد کندل برای مانیتورینگ (در صورت فعال)
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

# ─── توابع امن fetch (همان قبلی)
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

# ─── get_top_symbols، get_ohlcv_df، calculate_indicators و بقیه توابع همان‌هایی هستند که خودت داشتی (کپی دقیق)
# برای اختصار از همون توابع شما استفاده می‌کنیم - اگر تغییرات پایه خواستی بگو تا آن‌ها را جدا بازبینی کنم.

# --- (کپی توابع get_top_symbols, get_ohlcv_df, calculate_indicators, calculate_pivot_points, calculate_obv, calculate_vwap, calculate_fibonacci, detect_candlestick_patterns, find_local_extrema, detect_divergence, confirm_high_tf, volume baseline functions) ---
# برای جلوگیری از تکرار، فرض می‌گیریم این توابع دقیقا همان نسخهٔ قبلی شما هستند و در فایل حضور دارند.
# (در صورت نیاز من میتوانم کل فایل را با همه توابع بازنویسی کنم — ولی چون خودت همین الان همه رو دادی، من تغییرات فقط در بخش‌های پایین اعمال کردم.)

# === توابع جدید کمکی برای دقت بالاتر ===

def candle_strength_metric(df):
    """برمی‌گرداند مقدار 0..1 که نشان‌دهندهٔ قدرت کندل آخر است و نوع کندل ('bull','bear','neutral')"""
    try:
        last = df.iloc[-1]
        body = abs(last['close'] - last['open'])
        rng = (last['high'] - last['low']) + 1e-9
        body_ratio = body / rng
        close_pos = (last['close'] - last['low']) / (rng + 1e-9)

        if last['close'] > last['open']:
            # bullish candle
            score = 0.0
            if body_ratio >= CANDLE_BODY_MIN and close_pos >= CLOSE_POS_THRESHOLD:
                score = 1.0
            elif body_ratio >= 0.3 and close_pos >= 0.55:
                score = 0.6
            elif body_ratio >= 0.2:
                score = 0.3
            return score, 'bull'
        elif last['close'] < last['open']:
            # bearish candle
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

# بازنویسی compute_signal_score برای اضافه کردن فاکتورهای جدید
def compute_signal_score(sig, df, intrabar_change):
    try:
        base = 0.0
        stars_count = len(sig.get('stars', []))
        base += stars_count * 6.0

        adx = float(df['ADX'].iloc[-1]) if 'ADX' in df.columns and not pd.isna(df['ADX'].iloc[-1]) else 0.0
        base += min(adx, 50.0) * 0.6

        # intrabar change اهمیت داره
        base += min(abs(intrabar_change) * 6.0, 25.0)

        vol_mean = float(df['volume'].rolling(20).mean().iloc[-1]) if 'volume' in df.columns else np.nan
        vol_rel = 1.0
        if not pd.isna(vol_mean) and vol_mean > 0:
            vol_rel = float(df['volume'].iloc[-1]) / vol_mean
            if vol_rel > 1.0:
                base += (vol_rel - 1.0) * 8.0

        # MACD histogram magnitude
        macd_hist = float(df['MACD_HIST'].iloc[-1]) if 'MACD_HIST' in df.columns and not pd.isna(df['MACD_HIST'].iloc[-1]) else 0.0
        base += min(abs(macd_hist) * 8.0, 20.0)

        # EMA separation
        ema9 = float(df['EMA9'].iloc[-1]) if 'EMA9' in df.columns and not pd.isna(df['EMA9'].iloc[-1]) else 0.0
        ema21 = float(df['EMA21'].iloc[-1]) if 'EMA21' in df.columns and not pd.isna(df['EMA21'].iloc[-1]) else 1.0
        ema_diff = (ema9 - ema21) / (ema21 if ema21 != 0 else 1e-9)
        if sig.get('type') == 'LONG':
            base += max(0.0, ema_diff * 120.0)
        else:
            base += max(0.0, -ema_diff * 120.0)

        # Candle strength
        cs, ctype = candle_strength_metric(df)
        base += cs * 10.0

        # VWAP
        if 'VWAP' in df.columns and not pd.isna(df['VWAP'].iloc[-1]):
            if (sig.get('type') == 'LONG' and df['close'].iloc[-1] > df['VWAP'].iloc[-1]) or \
               (sig.get('type') == 'SHORT' and df['close'].iloc[-1] < df['VWAP'].iloc[-1]):
                base += 4.0

        # RSI extremes penalty
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

        # بررسی ستون‌های مورد نیاز
        needed = ['EMA9','EMA21','ATR14','RSI','ADX','volume']
        if any(col not in df.columns or pd.isna(df[col].iloc[-1]) for col in needed):
            return None

        price = df['close'].iloc[-1]
        prev_price = df['close'].iloc[-2]
        intrabar_change = ((price - prev_price) / (prev_price + 1e-9)) * 100.0

        # trend by ichimoku
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

        # confirmations
        confs = {}
        # EMA cross
        ema9 = df['EMA9'].iloc[-1]; ema21 = df['EMA21'].iloc[-1]
        ema_cross_long = (ema9 > ema21)
        ema_cross_short = (ema9 < ema21)
        confs['ema_cross'] = ema_cross_long or ema_cross_short

        # RSI confirm
        rsi = df['RSI'].iloc[-1]
        confs['rsi_long'] = rsi > 55
        confs['rsi_short'] = rsi < 45

        # MACD hist
        macd_hist = df['MACD_HIST'].iloc[-1] if 'MACD_HIST' in df.columns else (df['MACD'].iloc[-1] - df['Signal'].iloc[-1])
        confs['macd_long'] = macd_hist > 0
        confs['macd_short'] = macd_hist < 0

        # candle strength
        cs_score, cs_type = candle_strength_metric(df)
        confs['candle_strong'] = cs_score >= 0.6

        # volume confirm
        vol_ok, vol_rel = volume_confirmation(df)
        confs['volume_spike'] = vol_ok

        # vwap
        confs['above_vwap'] = False
        if 'VWAP' in df.columns and not pd.isna(df['VWAP'].iloc[-1]):
            confs['above_vwap'] = df['close'].iloc[-1] > df['VWAP'].iloc[-1]

        # calculate volatility
        vol_ratio = atr_volatility(df)
        required_confirms = REQUIRED_CONFIRMS_HIGHVOL if vol_ratio >= VOLATILITY_HIGH_THRESHOLD else REQUIRED_CONFIRMS_LOWVOL

        # Determine candidate type (LONG/SHORT) based on EMA and trend
        candidate_type = None
        if ema_cross_long and (trend == 'bullish' or rsi > 50):
            candidate_type = 'LONG'
        elif ema_cross_short and (trend == 'bearish' or rsi < 50):
            candidate_type = 'SHORT'
        else:
            return None  # no clear directional bias

        # Count confirmations relevant to candidate_type
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

        # require minimum confirmations (dynamic)
        if confirms < required_confirms:
            logging.debug(f"{symbol} skipped: confirms={confirms} < required {required_confirms} | reasons={confirm_reasons}")
            return None

        # Now compute stop/tp based on ATR
        atr = df['ATR14'].iloc[-1] if 'ATR14' in df.columns else (df['ATR'].iloc[-1] if 'ATR' in df.columns else 0.0)
        if atr <= 0 or pd.isna(atr):
            return None

        # Set entry/stop/tp with ATR-based sizing (conservative multipliers)
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

        # strength label (سخت‌تر)
        if score >= (MIN_SCORE * 1.8):
            temp_sig['strength'] = 'strong'
        elif score >= MIN_SCORE:
            temp_sig['strength'] = 'normal'
        else:
            temp_sig['strength'] = 'weak'

        # final check on score
        if temp_sig['score'] < MIN_SCORE:
            logging.debug(f"{symbol} score {temp_sig['score']} below MIN_SCORE {MIN_SCORE}")
            return None

        # prevent immediate re-signal
        prev = last_alerts.get(symbol)
        if prev and prev.get('type') == candidate_type and (time.time() - prev.get('time', 0) < SIGNAL_INTERVAL):
            return None
        last_alerts[symbol] = {'type': candidate_type, 'time': time.time()}

        # full log
        logging.info(f"[SIG_CAND] {symbol} | Type={candidate_type} | Score={score} | Confirms={confirms}/{required_confirms} | Reasons={confirm_reasons} | Vol={vol_ratio:.4f}")

        return temp_sig

    except Exception as e:
        logging.error(f"check_signal {symbol}: {e}")
        return None

# ===== main loop (نگه داشتم، با چند بهبود کوچک) =====
def main():
    logging.info("🚀 ربات شروع شد — نسخهٔ با دقت بالا")
    while True:
        try:
            top_symbols = get_top_symbols()
            candidates = []

            # مرحلهٔ اول: بررسی نمادها و جمع‌آوری کاندیدها (ابتدا تایم‌فریم‌های بلندتر را بررسی می‌کنیم)
            for symbol_data in top_symbols:
                symbol = symbol_data['symbol']
                change = symbol_data.get('change', 0.0)
                current_vol = symbol_data.get('volume', 0.0)

                # فیلتر سریع براساس 24h change (اگر کمِ خیلی، صرفه‌نظر کن)
                if abs(change) < FAST_FILTER_CHANGE:
                    continue

                # ابتدا تایم‌فریم‌های بلندتر را بررسی کن (پایه تایید)
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

                # اگر تایم بالا تأیید نکرد، رد کن (حفظ سخت‌گیری برای وین‌ریت)
                if not high_confirmed:
                    continue

                # حالا تایم‌های پایین‌تر را بررسی کن (فقط در صورت تأیید TF بالا)
                for tf in [t for t in TIMEFRAMES if t not in HIGH_TFS]:
                    df = get_ohlcv_df(symbol, tf)
                    if df is None or df.empty:
                        continue
                    df = calculate_indicators(df)
                    sig = check_signal(df, symbol, change)
                    if sig:
                        candidates.append({'symbol': symbol, 'tf': tf, 'signal': sig, 'score': sig.get('score', 0.0), 'df': df, '24h_volume': current_vol})

            logging.info(f"[INFO] Found {len(candidates)} raw candidates this cycle.")

            # فیلتر اولیه بر اساس MIN_SCORE
            filtered = [c for c in candidates if c['score'] >= MIN_SCORE]
            logging.info(f"[INFO] {len(filtered)} candidates passed MIN_SCORE >= {MIN_SCORE}")

            # مرتب‌سازی
            filtered.sort(key=lambda x: x['score'], reverse=True)

            # فیلتر دقیق‌تر: واگرایی/تأیید TF پایین و cooldown
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

                # اگر تایم پایینه، نیاز به تایید داریم (اگر REQUIRE_DIVERGENCE True)
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

            # بررسی اسپایک حجم (heavy) فقط برای shortlisted
            if final_candidates:
                for c in final_candidates:
                    sym = c['symbol']
                    vol24 = c.get('24h_volume', 0.0)
                    spike = detect_volume_spike_detailed(sym, vol24)
                    c['volume_spike'] = spike

            # ارسال پیام‌ها
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

                    # optional short monitor (off by default)
                    if MONITOR_AFTER_SEND:
                        try:
                            monitor_signal_outcome(sym, tf, s['entry'], s['tp'], s['stop'], MONITOR_CANDLES)
                        except Exception as e:
                            logging.warning(f"monitoring failed for {sym}: {e}")

                    time.sleep(SEND_DELAY_BETWEEN_MSGS)

            # صبر تا چرخه بعدی
            time.sleep(SIGNAL_INTERVAL)

        except Exception as e:
            logging.error(f"خطا در main: {e}")
            time.sleep(30)

# ===== (اختیاری) تابع مانیتور کوتاه برای بررسی نتیجهٔ سیگنال ارسالی =====
def monitor_signal_outcome(symbol, timeframe, entry, tp, stop, look_candles=6):
    """
    اگر MONITOR_AFTER_SEND True شود، این تابع کندل‌های بعدی را می‌خواند و بررسی می‌کند
    آیا TP یا Stop در بازهٔ زمانی کوتاه خورده یا نه. استفاده برای جمع‌آوری داده و بهبود آتی.
    """
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
