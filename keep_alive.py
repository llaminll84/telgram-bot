import os
from flask import Flask
from threading import Thread

app = Flask('')

@app.route('/')
def home():
    return "✅ ربات فعال است!"

def run():
    port = int(os.environ.get("PORT", 8000))
    app.run(host='0.0.0.0', port=port)
