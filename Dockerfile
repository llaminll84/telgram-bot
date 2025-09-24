# Base image سبک با Python 3.11
FROM python:3.11.7-slim
FROM public.ecr.aws/docker/library/python:3.11.7-slim


# محیط کاری
WORKDIR /app

# غیر فعال کردن cache و buffering
ENV PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# نصب پکیج‌های سیستمی مورد نیاز
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        libffi-dev \
        && apt-get clean && rm -rf /var/lib/apt/lists/*

# کپی فایل‌ها
COPY requirements.txt .
COPY bot3.py keep_alive.py ./

# نصب پکیج‌ها
RUN pip install --no-cache-dir -r requirements.txt

# اجرای ربات
CMD ["python", "bot3.py"]
