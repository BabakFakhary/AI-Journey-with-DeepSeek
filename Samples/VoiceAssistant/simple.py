#                                                                     به نام خدا
# install: pip install --upgrade arabic-reshaper
import arabic_reshaper
#-------------------------------------------------------
# install: pip install python-bidi
from bidi.algorithm import get_display
#-------------------------------------------------------
# pip install speechrecognition pyaudio pyttsx3 
import speech_recognition as sr
import pyttsx3
import threading
import time

# تنظیمات نمایش فارسی
def fa(text):
    return get_display(arabic_reshaper.reshape(text))

class SimpleVoiceAssistant:
    def __init__(self):
        # تنظیمات تشخیص گفتار
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()
        
        # تنظیمات تبدیل متن به گفتار
        self.tts_engine = pyttsx3.init()
        self.tts_engine.setProperty('rate', 150)  # سرعت گفتار
        
        # تنظیم کیفیت صدا
        voices = self.tts_engine.getProperty('voices')
        if len(voices) > 1:
            self.tts_engine.setProperty('voice', voices[1].id)  # صدای زنانه
        
        print(fa("🎤 دستیار صوتی راه‌اندازی شد!"))
    
    def listen(self):
        """گوش دادن به صدا و تبدیل به متن"""
        try:
            with self.microphone as source:
                print(fa("🔊 در حال گوش دادن... صحبت کنید"))
                self.recognizer.adjust_for_ambient_noise(source)
                audio = self.recognizer.listen(source, timeout=10)
            
            print(fa("📝 در حال پردازش صدا..."))
            text = self.recognizer.recognize_google(audio, language='fa-IR')
            print(fa(f"👤 شما گفتید: {text}"))
            return text
            
        except sr.WaitTimeoutError:
            print(fa("⏰ زمان گوش دادن به پایان رسید"))
            return None
        except sr.UnknownValueError:
            print(fa("❌ متوجه نشدم. لطفا دوباره تلاش کنید"))
            return None
        except Exception as e:
            print(fa(f"❌ خطا در پردازش صدا: {e}"))
            return None
    
    def speak(self, text):
        """تبدیل متن به گفتار"""
        try:
            print(f"🤖 دستیار: {text}")
            self.tts_engine.say(text)
            self.tts_engine.runAndWait()
        except Exception as e:
            print(fa(f"❌ خطا در پخش صدا: {e}"))
    
    def start_conversation(self):
        """شروع گفتگو با دستیار"""
        self.speak(fa("سلام! من دستیار صوتی شما هستم. چگونه می‌توانم کمک کنم؟"))
        
        while True:
            user_speech = self.listen()
            
            if user_speech:
                # پردازش دستورات
                response = self.process_command(user_speech)
                self.speak(response)
                
                # خروج اگر کاربر بگوید "خداحافظ"
                if "خداحافظ" in user_speech or "بای" in user_speech:
                    self.speak(fa("خداحافظ! موفق باشید"))
                    break
    
    def process_command(self, command):
        """پردازش دستورات کاربر"""
        command = command.lower()
        
        if "سلام" in command:
            return fa("سلام! چطور میتونم کمک کنم؟")
        
        elif "حالت چطوره" in command or "چطوری" in command:
            return fa("من خوبم ممنون! شما چطورید؟")
        
        elif "ساعت" in command:
            current_time = time.strftime("%H:%M")
            return fa(f"ساعت {current_time} است")
        
        elif "تاریخ" in command:
            current_date = time.strftime("%Y/%m/%d")
            return fa(f"امروز {current_date} است")
        
        elif "اسم تو چیه" in command:
            return fa("من یک دستیار صوتی هوشمند هستم")
        
        elif "ممنون" in command or "مرسی" in command:
            return fa("خواهش می کنم! کاری دیگری هست؟")
        
        else:
            return fa("متوجه نشدم. می‌توانید سوال خود را تکرار کنید؟")

# استفاده از دستیار صوتی
if __name__ == "__main__":
    assistant = SimpleVoiceAssistant()
    assistant.start_conversation()