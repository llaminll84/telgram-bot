import time
import os
import logging
import ccxt
import pandas as pd
import numpy as np
from telegram import Bot

# ----------------- تنظیمات -----------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

# سرور keep_alive اگر لازم داری
try:
    from keep_alive import keep_alive
    keep_alive()
except Exception:
    logging.debug("keep_alive module not found or failed — ادامه بدون سرویس نگهدارنده")

TELEGRAM_TOKEN = os.getenv("BOT_TOKEN")
CHAT_ID = os.getenv("CHAT_ID")

if not TELEGRAM_TOKEN or not CHAT_ID:
    logging.warning("BOT_TOKEN یا CHAT_ID تنظیم نشده‌اند. ارسال تلگرام غیرفعال است.")
    bot = None
else:
    bot = Bot(token=TELEGRAM_TOKEN)

# پیکربندی صرافی
exchange = ccxt.kucoin({'enableRateLimit': True})
try:
    exchange.load_markets()
except Exception as e:
    logging.warning(f"load_markets failed: {e}")

TOP_N = 80
TIMEFRAMES = ['5m', '15m', '1h']
CANDLES_LIMIT = 200
ALERT_COOLDOWN = 60*60
MIN_VOLUME_USDT = 50_000
CONFIRM_REQUIRED_TFS = 2
MAX_MSG_PART = 4000

last_alert_time = {}

# ----------------- کمک‌تابع‌ها -----------------
def safe_send_telegram(text):
    if not bot:
        logging.info(text)
        return
    try:
        if len(text) <= MAX_MSG_PART:
            bot.send_message(chat_id=CHAT_ID, text=text)
        else:
            for i in range(0, len(text), MAX_MSG_PART):
                bot.send_message(chat_id=CHAT_ID, text=text[i:i+MAX_MSG_PART])
    except Exception as e:
        logging.exception(f"[Telegram Error] {e}")

def get_top_symbols():
    try:
        tickers = exchange.fetch_tickers()
    except Exception as e:
        logging.exception(f"fetch_tickers failed: {e}")
        return []
    symbols = []
    for symbol, data in tickers.items():
        if not symbol.endswith('/USDT'):
            continue
        if 'PERP' in symbol or 'SWAP' in symbol:
            continue
        vol = data.get('quoteVolume') or data.get('volume') or 0
        change = data.get('percentage') or 0.0
        symbols.append({'symbol': symbol, 'volume': float(vol), 'change': float(change)})
    symbols.sort(key=lambda x: x['volume'], reverse=True)
    return symbols[:TOP_N]

def get_ohlcv_df(symbol, timeframe, limit=CANDLES_LIMIT):
    try:
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
    except Exception as e:
        logging.exception(f"fetch_ohlcv failed for {symbol} {timeframe}: {e}")
        return None
    if not ohlcv or len(ohlcv) < 80:
        logging.debug(f"insufficient ohlcv for {symbol} {timeframe}: {len(ohlcv) if ohlcv else 0}")
        return None
    df = pd.DataFrame(ohlcv, columns=['timestamp','open','high','low','close','volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    df[['open','high','low','close','volume']] = df[['open','high','low','close','volume']].astype(float)
    return df

def calculate_indicators(df):
    if df is None or df.shape[0] < 80:
        return df
    df = df.copy()
    df['EMA9'] = df['close'].ewm(span=9, adjust=False).mean()
    df['EMA21'] = df['close'].ewm(span=21, adjust=False).mean()
    ema12 = df['close'].ewm(span=12, adjust=False).mean()
    ema26 = df['close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = ema12 - ema26
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['BB_Mid'] = df['close'].rolling(20).mean()
    df['BB_Std'] = df['close'].rolling(20).std()
    df['BB_Upper'] = df['BB_Mid'] + 2*df['BB_Std']
    df['BB_Lower'] = df['BB_Mid'] - 2*df['BB_Std']
    tr1 = df['high'] - df['low']
    tr2 = (df['high'] - df['close'].shift()).abs()
    tr3 = (df['low'] - df['close'].shift()).abs()
    df['TR'] = pd.concat([tr1,tr2,tr3], axis=1).max(axis=1)
    df['ATR'] = df['TR'].rolling(14).mean()
    delta = df['close'].diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    roll_up = up.ewm(span=14, adjust=False).mean()
    roll_down = down.ewm(span=14, adjust=False).mean()
    rs = roll_up / roll_down.replace(0, np.nan)
    df['RSI'] = 100 - (100 / (1 + rs))
    rsi_min = df['RSI'].rolling(14).min()
    rsi_max = df['RSI'].rolling(14).max()
    df['StochRSI'] = (df['RSI'] - rsi_min) / (rsi_max - rsi_min + 1e-9)
    df['Tenkan'] = (df['high'].rolling(9).max() + df['low'].rolling(9).min()) / 2
    df['Kijun'] = (df['high'].rolling(26).max() + df['low'].rolling(26).min()) / 2
    df['SenkouA'] = ((df['Tenkan'] + df['Kijun'])/2).shift(26)
    df['SenkouB'] = ((df['high'].rolling(52).max() + df['low'].rolling(52).min())/2).shift(26)
    df['Chikou'] = df['close'].shift(-26)
    return df

def detect_candlestick_patterns(df):
    patterns = []
    if df is None or df.shape[0]<3:
        return patterns
    last = df.iloc[-1]
    prev = df.iloc[-2]
    prev2 = df.iloc[-3]
    open_, close, high, low = last['open'], last['close'], last['high'], last['low']
    prev_open, prev_close = prev['open'], prev['close']
    prev2_open, prev2_close = prev2['open'], prev2['close']
    if prev_close<prev_open and close>open_ and close>prev_open and open_<prev_close:
        patterns.append('Bullish Engulfing')
    if prev_close>prev_open and close<open_ and open_>prev_close and close<prev_open:
        patterns.append('Bearish Engulfing')
    if (close-low)>2*abs(open_-low):
        patterns.append('Hammer')
    if (high-close)>2*abs(high-open_):
        patterns.append('Hanging Man')
    if abs(close-open_)/(high-low+1e-9)<0.15:
        patterns.append('Doji')
    if (prev2_close<prev2_open) and (abs(prev['close']-prev['open'])<(prev2_open-prev2_close)*0.5) and (close>open_ and close>prev2_open):
        patterns.append('Morning Star')
    if (prev2_close>prev2_open) and (abs(prev['close']-prev['open'])<(prev2_close-prev2_open)*0.5) and (close<open_ and close<prev2_open):
        patterns.append('Evening Star')
    return patterns

def detect_order_block(df):
    blocks=[]
    if df is None or df.shape[0]<30:
        return blocks
    vol_mean=df['volume'].rolling(20).mean()
    close_std=df['close'].std()
    for i in range(2,min(len(df)-1,60)):
        body=abs(df['close'].iloc[-i]-df['open'].iloc[-i])
        if df['volume'].iloc[-i]>(vol_mean.iloc[-i] or 0)*2 and body>(close_std*1.2):
            blocks.append((float(df['low'].iloc[-i]),float(df['high'].iloc[-i])))
    return blocks

def check_spread(symbol):
    try:
        ob=exchange.fetch_order_book(symbol, limit=5)
        if not ob or not ob.get('bids') or not ob.get('asks'):
            return None
        best_bid=ob['bids'][0][0]
        best_ask=ob['asks'][0][0]
        mid=(best_bid+best_ask)/2
        spread_pct=(best_ask-best_bid)/(mid+1e-9)
        return float(spread_pct)
    except Exception:
        return None

def check_signal(df,symbol,change):
    try:
        if df is None or df.shape[0]<80:
            return None
        price=df['close'].iloc[-1]
        spread=check_spread(symbol)
        if spread is not None and spread>0.01:
            logging.debug(f"{symbol} skipped due to high spread: {spread:.4f}")
            return None
        senkou_a=df['SenkouA'].iloc[-1]
        senkou_b=df['SenkouB'].iloc[-1]
        trend='neutral'
        if not np.isnan(senkou_a) and not np.isnan(senkou_b):
            if price>senkou_a and price>senkou_b:
                trend='bullish'
            elif price<senkou_a and price<senkou_b:
                trend='bearish'
        patterns=detect_candlestick_patterns(df)
        order_blocks=detect_order_block(df)
        vol=df['volume'].iloc[-1]
        vol_mean_20=df['volume'].rolling(20).mean().iloc[-1] or 0
        volume_check=vol_mean_20>0 and vol>vol_mean_20*1.5
        stoch=df['StochRSI'].iloc[-1] if not np.isnan(df['StochRSI'].iloc[-1]) else 0.5
        stoch_check=False
        if trend=='bullish':
            stoch_check=stoch<0.35
        elif trend=='bearish':
            stoch_check=stoch>0.65
        atr=df['ATR'].iloc[-1] if not np.isnan(df['ATR'].iloc[-1]) else 0
        atr_mean=df['ATR'].rolling(14).mean().iloc[-1] if not np.isnan(df['ATR'].rolling(14).mean().iloc[-1]) else 0
        atr_check=atr_mean>0 and atr>atr_mean*0.8
        change_filter=abs(change)>=0.8
        if change>=1 and trend=='bullish' and any(p in patterns for p in ['Bullish Engulfing','Hammer','Morning Star']) and volume_check and stoch_check and atr_check and change_filter:
            entry=price
            tp=price*1.015
            stop=price*(1-max(0.003,0.5*(atr/(price+1e-9))))
            signal_type='LONG'
        elif change<=-1 and trend=='bearish' and any(p in patterns for p in ['Bearish Engulfing','Hanging Man','Evening Star']) and volume_check and stoch_check and atr_check and change_filter:
            entry=price
            tp=price*0.985
            stop=price*(1+max(0.003,0.5*(atr/(price+1e-9))))
            signal_type='SHORT'
        else:
            return None
        return {
            'entry':float(entry),
            'tp':float(tp),
            'stop':float(stop),
            'type':signal_type,
            'patterns':patterns,
            'order_blocks':order_blocks,
            'spread':spread,
            'volume':vol
        }
    except Exception as e:
        logging.exception(f"check_signal error for {symbol}: {e}")
        return None

# ----------------- حلقه اصلی -----------------
def main():
    logging.info("🚀 ربات بهینه‌شده شروع شد")
    while True:
        try:
            top_symbols=get_top_symbols()
            alerts=[]
            for symbol_data in top_symbols:
                symbol=symbol_data['symbol']
                change=symbol_data.get('change',0)
                last=last_alert_time.get(symbol)
                if last and (time.time()-last)<ALERT_COOLDOWN:
                    logging.debug(f"{symbol} in cooldown, skipping.")
                    continue
                tf_signals=[]
                for tf in TIMEFRAMES:
                    df=get_ohlcv_df(symbol,tf)
                    if df is None:
                        continue
                    df=calculate_indicators(df)
                    signal=check_signal(df,symbol,change)
                    logging.info(f"[{symbol} {tf}] Close: {df['close'].iloc[-1]:.6f} | change: {change:.2f}% | signal: {bool(signal)}")
                    if signal:
                        tf_signals.append((tf,signal))
                if len(tf_signals)>=CONFIRM_REQUIRED_TFS:
                    alerts.append((symbol,tf_signals))
                    last_alert_time[symbol]=time.time()
            if alerts:
                msg="🚨 Multi-Coin Alert 🚨\n\n"
                for symbol,sigs in alerts:
                    msg+=f"{symbol}\n"
                    for tf,s in sigs:
                        msg+=(f"TF: {tf} | Type: {s['type']} | Entry: {s['entry']:.6f} | TP: {s['tp']:.6f} | Stop: {s['stop']:.6f}\n"
                              f"Patterns: {s['patterns']} | OrderBlocks: {s['order_blocks']} | Vol: {int(s['volume'])} | Spread: {s['spread']}\n\n")
                    msg+="-"*30+"\n"
                safe_send_telegram(msg)
            logging.info("⏳ sleep 300s")
            time.sleep(300)
        except Exception as e:
            logging.exception(f"Unhandled loop error: {e}")
            time.sleep(30)

if __name__=="__main__":
    main()
