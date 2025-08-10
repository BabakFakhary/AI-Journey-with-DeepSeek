#                                                                به نام خدا
# install: pip install --upgrade arabic-reshaper
import arabic_reshaper
#-------------------------------------------------------
# install: pip install python-bidi
from bidi.algorithm import get_display
#-------------------------------------------------------
import numpy as np

# ساخت دو ماتریس ۳x۳
A = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
B = np.array([[9, 8, 7], [6, 5, 4], [3, 2, 1]])

# ضرب ماتریس‌ها
C = np.dot(A, B)
print("حاصل ضرب ماتریس‌ها:\n", C)

# محاسبه معکوس ماتریس A (اگر وجود داشته باشد)
try:
    A_inv = np.linalg.inv(A)
    print(get_display(arabic_reshaper.reshape("\n معکوس ماتریس A:\n", A_inv)))
except:
    print(get_display(arabic_reshaper.reshape("\n ماتریس A معکوس‌ پذیر نیست!")))


# دستورات Git
#--------------------------------------------------------
# git add Week1/Matrix_Operations.py
# git commit -m "اضافه کردن تمرین ضرب ماتریس‌ها"
# git push    