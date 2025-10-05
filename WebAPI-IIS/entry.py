# entry.py - فایل ورودی جدید برای IIS
import os
import sys

# اضافه کردن مسیر فعلی به Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from Hugging_Face_Transformer_IIS import create_app
    application = create_app()
    
    # برای تست محلی
    if __name__ == '__main__':
        application.run(host='0.0.0.0', port=5000, debug=False)
        
except Exception as e:
    print(f"Error initializing application: {e}")
    import traceback
    traceback.print_exc()
    
    # Fallback application
    from flask import Flask
    application = Flask(__name__)
    
    @application.route('/')
    def home():
        return 'NLP API - Fallback Mode'
    
    @application.route('/health')
    def health():
        return {'status': 'fallback', 'error': str(e)}