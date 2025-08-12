#                                                                            به نام خدا
# نصب کتابخانه‌های لازم
# pip install fastapi uvicorn
from fastapi import FastAPI
# ---------------------------------------
# تست اجرا لوکال
# python -m uvicorn main:app --reload
# http://127.0.0.1:8000/docs
#----------------------------------------

app = FastAPI()

@app.get("/api/hello")
def read_hello():
    return {"message": "Hello from FastAPI"}