# به نام خدا                                                                              
# install: pip install --upgrade arabic-reshaper
import arabic_reshaper
#-------------------------------------------------------
# install: pip install python-bidi
from bidi.algorithm import get_display
#-------------------------------------------------------
from transformers import MarianMTModel, MarianTokenizer, MT5ForConditionalGeneration, MT5Tokenizer
import torch

def fa(text):
    return get_display(arabic_reshaper.reshape(text))

# ==================================================================================
#                        Translate With MarianMT If Dont Work Usr T5
# روش ترجمه فرانسوی به انگلیسی از مدل MarianMT استفاده می کنه
# ==================================================================================

class MyTranslator:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.en_fa_model = None
        self.en_fa_tokenizer = None
        self.en_fa_model_name = "در دسترس نیست"
        self.en_fa_method = "نامشخص"
        self.is_mt5 = False

        print(fa("در حال بارگذاری مدل انگلیسی → فارسی..."))

        # 1. اول سعی در لود مدل MarianMT واقعی برای فارسی
        marian_fa_models = ["Helsinki-NLP/opus-mt-en-fa"]
        for name in marian_fa_models:
            try:
                print(fa(f"در حال تست مدل MarianMT: {name}"))
                tokenizer = MarianTokenizer.from_pretrained(name)
                model = MarianMTModel.from_pretrained(name)
                
                # تست سریع برای اطمینان از فارسی بودن خروجی
                test = tokenizer("Hello", return_tensors="pt").to(self.device)
                with torch.no_grad():
                    out = model.generate(**test, max_length=20)
                sample = tokenizer.decode(out[0], skip_special_tokens=True)
                
                if any(0x0600 <= ord(c) <= 0x06FF for c in sample):  # حروف فارسی/عربی
                    self.en_fa_model = model.to(self.device)
                    self.en_fa_tokenizer = tokenizer
                    self.en_fa_model_name = name.split("/")[-1]
                    self.en_fa_method = "MarianMT"
                    print(fa(f"موفق! مدل فارسی با MarianMT بارگذاری شد"))
                    print(fa(f"تست نمونه: Hello → {sample}"))
                    break
            except Exception as e:
                print(fa(f"خطا در MarianMT: {e}"))
                continue

        # 2. اگر MarianMT نشد → mT5 (بهترین کیفیت فارسی)
        if self.en_fa_model is None:
            try:
                name = "persiannlp/mt5-small-parsinlu-translation_en_fa"
                print(fa(f"در حال بارگذاری mT5 قوی: {name}"))
                self.en_fa_tokenizer = MT5Tokenizer.from_pretrained(name)
                self.en_fa_model = MT5ForConditionalGeneration.from_pretrained(name).to(self.device)
                self.en_fa_model_name = "mt5-small-parsinlu"
                self.en_fa_method = "mT5"
                self.is_mt5 = True
                print(fa("موفق! مدل mT5 با کیفیت بالا بارگذاری شد"))
            except Exception as e:
                print(fa(f"خطا در mT5: {e}"))

        # 3. مدل انگلیسی → فرانسوی (همیشه MarianMT و همیشه کار می‌کنه)
        try:
            fr_name = "Helsinki-NLP/opus-mt-en-fr"
            self.en_fr_tokenizer = MarianTokenizer.from_pretrained(fr_name)
            self.en_fr_model = MarianMTModel.from_pretrained(fr_name).to(self.device)
            self.en_fr_method = "MarianMT"
            print(fa("مدل فرانسوی (MarianMT) با موفقیت بارگذاری شد"))
        except Exception as e:
            print(fa(f"خطا در مدل فرانسوی: {e}"))

    def translate_to_persian(self, text):
        if not self.en_fa_model:
            return "مدل فارسی در دسترس نیست"

        if self.is_mt5:
            input_text = f"translate English to Persian: {text}"
            inputs = self.en_fa_tokenizer(input_text, return_tensors="pt", truncation=True).to(self.device)
        else:
            inputs = self.en_fa_tokenizer(text, return_tensors="pt", padding=True, truncation=True).to(self.device)

        with torch.no_grad():
            outputs = self.en_fa_model.generate(
                **inputs,
                max_length=128,
                num_beams=5,
                early_stopping=True
            )
        translated = self.en_fa_tokenizer.decode(outputs[0], skip_special_tokens=True)
        if self.is_mt5:
            translated = translated.replace("translate English to Persian:", "").strip()
        return translated

    def translate_to_french(self, text):
        if not self.en_fr_model:
            return "مدل فرانسوی در دسترس نیست"
        inputs = self.en_fr_tokenizer(text, return_tensors="pt", padding=True, truncation=True).to(self.device)
        with torch.no_grad():
            outputs = self.en_fr_model.generate(**inputs, max_length=128, num_beams=5, early_stopping=True)
        return self.en_fr_tokenizer.decode(outputs[0], skip_special_tokens=True)


# ====================== اجرای تست ======================
if __name__ == "__main__":
    translator = MyTranslator()

    texts = [
        "Hello, how are you?",
        "Artificial intelligence is amazing.",
        "Thank you very much for your help.",
        "What is your name?",
        "Today is a beautiful day."
    ]

    print(fa("\n" + "="*70))
    print(fa("                  نتایج ترجمه و مقایسه روش‌ها"))
    print(fa("="*70))

    for text in texts:
        fa_text = translator.translate_to_persian(text)
        fr_text = translator.translate_to_french(text)

        print(f"EN → {text}")
        print(fa(f"FA → {fa_text}"))
        print(fa(f"     ├─ روش: {translator.en_fa_method} ({translator.en_fa_model_name})"))
        print(f"FR → {fr_text}")
        print(f"     └─ {fa("روش:")} MarianMT (opus-mt-en-fr)")
        print(fa("─" * 70))