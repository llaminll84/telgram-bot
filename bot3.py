import time
import os
import ccxt
import pandas as pd
from telegram import Bot
from keep_alive import keep_alive  # اگر روی هاست استفاده می‌کنی

# ── روشن نگه داشتن کانتینر (در صورت نیاز)
keep_alive()

# ── اطلاعات ربات تلگرام
TELEGRAM_TOKEN = os.getenv("BOT_TOKEN")
CHAT_ID = os.getenv("CHAT_ID")
bot = Bot(token=TELEGRAM_TOKEN)

bot.send_message(chat_id=CHAT_ID, text="✅ ربات با موفقیت راه‌اندازی شد!")

# ── صرافی
exchange = ccxt.kucoin()

TOP_N = 5        # فقط 5 کوین برای تست
TIMEFRAMES = ['5m']  # فقط یک تایم‌فریم برای سادگی


def get_top_symbols():
    tickers = exchange.fetch_tickers()
    symbols = []
    for s, d in tickers.items():
        if s.endswith('/USDT'):
            symbols.append({"symbol": s, "volume": d["quoteVolume"], "change": d["percentage"]})
    symbols.sort(key=lambda x: x["volume"], reverse=True)
    return symbols[:TOP_N]


def get_ohlcv_df(symbol, tf):
    ohlcv = exchange.fetch_ohlcv(symbol, tf)
    return pd.DataFrame(ohlcv, columns=["ts", "open", "high", "low", "close", "volume"])


# تابع سیگنال خیلی ساده
def check_signal(df, symbol):
    price = df["close"].iloc[-1]
    return {
        "entry": price,
        "tp": price * 1.01,
        "stop": price * 0.995,
        "type": "TEST",
        "patterns": [],
        "order_blocks": []
    }


def main():
    print("🚀 ربات ساده شروع شد")
    while True:
        try:
            top = get_top_symbols()
            alerts = []
            for coin in top:
                symbol = coin["symbol"]
                for tf in TIMEFRAMES:
                    df = get_ohlcv_df(symbol, tf)
                    if len(df) < 5:
                        continue
                    signal = check_signal(df, symbol)
                    print(f"[CMD] {symbol} | Close: {df['close'].iloc[-1]:.4f} | Signal: {signal}")
                    alerts.append((symbol, signal))

            if alerts:
                msg = "🚨 TEST ALERT 🚨\n"
                for sym, sig in alerts:
                    msg += f"{sym} → {sig['type']} | Entry: {sig['entry']:.4f}\n"
                try:
                    bot.send_message(chat_id=CHAT_ID, text=msg)
                except Exception as e:
                    print("[Telegram Error]", e)

            print("⏳ صبر برای ۳۰ ثانیه بعدی...\n")
            time.sleep(30)
        except Exception as e:
            print("⚠️ خطا:", e)
            time.sleep(15)


if __name__ == "__main__":
    main()
