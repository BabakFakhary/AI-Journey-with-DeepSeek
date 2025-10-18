#                                                                به نام خدا
# install: pip install --upgrade arabic-reshaper
import arabic_reshaper
#-------------------------------------------------------
# install: pip install python-bidi
from bidi.algorithm import get_display
#-------------------------------------------------------
# pip install speechrecognition pyaudio pyttsx3
import speech_recognition as sr
import pyttsx3
import time
import datetime
import webbrowser
import os

# تنظیمات نمایش فارسی
def fa(text):
    return get_display(arabic_reshaper.reshape(text))

# =================================================================================
# Voice Assistant نسخه پیشرفته‌تر با قابلیت‌های بیشتر
# =================================================================================

class AdvancedVoiceAssistant:
    def __init__(self):
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()
        self.tts_engine = pyttsx3.init()
        
        # تنظیمات صدا
        self.setup_voice()
        
        # دستورات قابل تشخیص
        self.commands = {
            'time': ['ساعت چنده', 'ساعت چند است', 'time'],
            'date': ['تاریخ امروز چیه', 'امروز چندمه'],
            'search': ['جستجو کن', 'سرچ کن', 'search'],
            'calculator': ['ماشین حساب', 'حساب کن'],
            'weather': ['هوا چطوره', 'آب و هوا'],
            'news': ['اخبار', 'خبر'],
            'exit': ['خداحافظ', 'بای', 'خروج']
        }
    
    def setup_voice(self):
        """تنظیمات کیفیت صدا"""
        self.tts_engine.setProperty('rate', 160)
        self.tts_engine.setProperty('volume', 0.8)
        
        # انتخاب بهترین صدا
        voices = self.tts_engine.getProperty('voices')
        if voices:
            self.tts_engine.setProperty('voice', voices[1].id)  # صدای زنانه
    
    def listen_continuous(self, timeout=5):
        """گوش دادن پیوسته با timeout"""
        try:
            with self.microphone as source:
                print(fa("🎤 آماده گوش دادن..."))
                self.recognizer.adjust_for_ambient_noise(source, duration=1)
                audio = self.recognizer.listen(source, timeout=timeout, phrase_time_limit=10)
            
            text = self.recognizer.recognize_google(audio, language='fa-IR')
            return text.lower()
            
        except sr.WaitTimeoutError:
            return None
        except sr.UnknownValueError:
            return "unknown"
        except Exception as e:
            print(fa(f"خطا: {e}"))
            return None
    
    def speak(self, text):
        """صحبت کردن با تاخیر طبیعی"""
        print(fa(f"🤖: {text}"))
        self.tts_engine.say(text)
        self.tts_engine.runAndWait()
    
    def get_time(self):
        """دریافت زمان فعلی"""
        now = datetime.datetime.now()
        return fa(f"ساعت {now.hour} و {now.minute} دقیقه است")
    
    def get_date(self):
        """دریافت تاریخ فعلی"""
        today = datetime.datetime.now()
        persian_months = {
            1: 'فروردین', 2: 'اردیبهشت', 3: 'خرداد',
            4: 'تیر', 5: 'مرداد', 6: 'شهریور',
            7: 'مهر', 8: 'آبان', 9: 'آذر',
            10: 'دی', 11: 'بهمن', 12: 'اسفند'
        }
        return fa(f"امروز {today.day} {persian_months[today.month]} {today.year} است")
    
    def search_web(self, query):
        """جستجو در وب"""
        webbrowser.open(f"https://www.google.com/search?q={query}")
        return fa(f"در حال جستجو برای {query}")
    
    def process_command(self, text):
        """پردازش هوشمند دستورات"""
        if not text or text == "unknown":
            return "متوجه نشدم، لطفا تکرار کنید"
        
        # تشخیص نوع دستور
        if any(cmd in text for cmd in self.commands['time']):
            return self.get_time()
        
        elif any(cmd in text for cmd in self.commands['date']):
            return self.get_date()
        
        elif any(cmd in text for cmd in self.commands['search']):
            query = text.replace('جستجو کن', '').replace('سرچ کن', '').strip()
            return self.search_web(query)
        
        elif any(cmd in text for cmd in self.commands['exit']):
            return "exit"
        
        elif 'چطوری' in text or 'حالت' in text:
            return "من خوبم ممنون! چطور میتونم کمک کنم؟"
        
        elif 'اسم' in text and 'تو' in text:
            return "من دستیار صوتی شما هستم. شما میتونید اسم من رو انتخاب کنید!"
        
        else:
            return f"دستور '{text}' را متوجه شدم اما هنوز این قابلیت را ندارم"
    
    def run(self):
        """اجرای اصلی دستیار"""
        self.speak("سلام! من دستیار صوتی شما هستم. برای خروج بگویید خداحافظ")
        
        while True:
            command = self.listen_continuous()
            
            if command:
                response = self.process_command(command)
                
                if response == "exit":
                    self.speak("خداحافظ! موفق باشید")
                    break
                else:
                    self.speak(response)

# اجرای دستیار
assistant = AdvancedVoiceAssistant()
assistant.run()