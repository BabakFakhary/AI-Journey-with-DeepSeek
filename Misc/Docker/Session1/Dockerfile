# استفاده از image پایه Python (سبک)
FROM python:3.9-slim

# تنظیم محیط برای Python
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV ENVIRONMENT=production

# ایجاد کاربر غیر-root برای امنیت
RUN useradd --create-home --shell /bin/bash appuser
WORKDIR /home/appuser/app

# کپی فایل requirements و نصب dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# کپی کد برنامه
COPY . .

# تغییر مالکیت فایل‌ها به کاربر appuser
RUN chown -R appuser:appuser /home/appuser/app

# تغییر کاربر به appuser
USER appuser

# باز کردن پورت 5000 برای دسترسی
EXPOSE 5000

# دستور اجرای برنامه
CMD ["python", "app.py"]