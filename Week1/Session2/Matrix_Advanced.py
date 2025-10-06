#                                                                             به نام خدا
# install: pip install --upgrade arabic-reshaper
import arabic_reshaper
#-------------------------------------------------------
# install: pip install python-bidi
from bidi.algorithm import get_display
#-------------------------------------------------------
import numpy as np

# ساخت یک ماتریس ۲x۲
M = np.array([[2, 3], [1, 4]])

# محاسبه دترمینان
det_M = np.linalg.det(M)
print(get_display(arabic_reshaper.reshape("دترمینان ماتریس M:")), det_M)

# ساخت ماتریس همانی ۳x۳
I = np.eye(3)
print(get_display(arabic_reshaper.reshape("\nماتریس همانی ۳x۳:\n")), I)

# بررسی معکوس‌پذیری
try:
    inv_M = np.linalg.inv(M)
    print(get_display(arabic_reshaper.reshape("\nمعکوس ماتریس M:\n")), inv_M)
except:
    print(get_display(arabic_reshaper.reshape("\nماتریس M معکوس‌ پذیر نیست!")))