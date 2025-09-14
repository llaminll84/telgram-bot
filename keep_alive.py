# keep_alive.py
import os
from flask import Flask
from threading import Thread

app = Flask('')

@app.route('/')
def home():
    return "✅ ربات فعال است!"

def run():
    # پورت را از محیط دریافت کن، اگر نبود 8080 پیش‌فرض
    port = int(os.environ.get("PORT", 8080))
    app.run(host='0.0.0.0', port=port)

def keep_alive():
    t = Thread(target=run)
    t.start()
