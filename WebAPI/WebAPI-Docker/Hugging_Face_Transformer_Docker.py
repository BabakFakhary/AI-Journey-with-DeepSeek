#                                                                     به نام خدا   
# install: pip install --upgrade arabic-reshaper
import arabic_reshaper
#-------------------------------------------------------
# install: pip install python-bidi
from bidi.algorithm import get_display
#-------------------------------------------------------
# pip install transformers datasets accelerate flask
import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification,
    AutoModelForTokenClassification,
    AutoModelForQuestionAnswering,
    pipeline
)
import warnings
from flask import Flask, request, jsonify
import time 
import logging
from logging.handlers import RotatingFileHandler
import os

# ============================================================================================
#                             Multiple Tasks با Hugging Face Transformers
# ============================================================================================

# تنظیمات نمایش فارسی
def fa(text):
    return get_display(arabic_reshaper.reshape(text))

# -------------------------------------------------------

warnings.filterwarnings('ignore')

# تنظیمات logging برای Production
def setup_logging():
    log_dir = "logs"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            RotatingFileHandler(
                os.path.join(log_dir, 'nlp_api.log'), 
                maxBytes=10485760,  # 10MB
                backupCount=5
            ),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

logger = setup_logging()

# بررسی GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")

# =====================================================
#  ایجاد سیستم NLP چندمنظوره - Optimized for Production
# =====================================================

class SimpleNLPSystem:
    """سیستم NLP ساده برای وقتی که مدل‌های اصلی load نمی‌شوند"""
    def analyze_text(self, text):
        return {
            'sentiment': {'sentiment': 'Neutral', 'confidence': 0.5, 'probabilities': [0.5, 0.5]},
            'entities': [],
            'summary': text[:100] + '...' if len(text) > 100 else text,
            'note': 'Using simple NLP system - main models not loaded'
        }

class AdvancedNLPSystem:
    def __init__(self):
        self.models = {}
        self.tokenizers = {}
        self.logger = logging.getLogger(__name__)
        self._load_models()
    
    def _load_models(self):
        """بارگذاری بهینه مدل‌ها برای Production"""
        self.logger.info("Loading models for production...")
        
        try:
            # استفاده از مدل‌های کوچک‌تر برای جلوگیری از crash
            SMALL_MODELS = {
                'classification': 'distilbert-base-uncased',  # مدل کوچک‌تر
                'ner': 'dbmdz/bert-small-cased-finetuned-conll03-english',  # مدل کوچک‌تر
            }
            
            # مدل برای classification - با اندازه کوچک
            self.logger.info("Loading classification model...")
            self.models['classification'] = AutoModelForSequenceClassification.from_pretrained(
                SMALL_MODELS['classification'], 
                num_labels=2,
                cache_dir="./model_cache"
            )
            self.tokenizers['classification'] = AutoTokenizer.from_pretrained(
                SMALL_MODELS['classification'],
                cache_dir="./model_cache"
            )
            
            # مدل برای NER - با اندازه کوچک
            self.logger.info("Loading NER model...")
            self.models['ner'] = AutoModelForTokenClassification.from_pretrained(
                SMALL_MODELS['ner'],
                cache_dir="./model_cache"
            )
            self.tokenizers['ner'] = AutoTokenizer.from_pretrained(
                SMALL_MODELS['ner'],
                cache_dir="./model_cache"
            )
            
            # انتقال به GPU
            for name, model in self.models.items():
                model.to(device)
                model.eval()  # حالت evaluation برای production
            
            self.logger.info("All models loaded successfully!")
            
        except Exception as e:
            self.logger.error(f"Error loading models: {e}")
            # به جای raise، از مدل‌های ساده استفاده می‌کنیم
            self.models = {}
            self.tokenizers = {}
    
    def analyze_text(self, text):
        """آنالیز کامل متن با مدیریت خطا"""
        try:
            # اگر مدل‌ها load نشده‌اند، از سیستم ساده استفاده کن
            if not self.models:
                simple_system = SimpleNLPSystem()
                return simple_system.analyze_text(text)
                
            results = {}
            
            # Sentiment Analysis
            results['sentiment'] = self._analyze_sentiment(text)
            
            # Named Entity Recognition
            results['entities'] = self._extract_entities(text)
            
            # Text Summary
            results['summary'] = self._generate_summary(text)
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error in text analysis: {e}")
            return {'error': str(e)}
    
    def _analyze_sentiment(self, text):
        """تحلیل احساسات با بهینه‌سازی"""
        try:
            if 'classification' not in self.models:
                return {'sentiment': 'Neutral', 'confidence': 0.5, 'error': 'Model not loaded'}
                
            inputs = self.tokenizers['classification'](
                text, 
                return_tensors="pt", 
                truncation=True, 
                padding=True,
                max_length=128  # کاهش طول برای صرفه‌جویی در memory
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.models['classification'](**inputs)
            
            probs = torch.softmax(outputs.logits, dim=1)
            sentiment = "Positive" if torch.argmax(probs) == 1 else "Negative"
            
            confidence = float(probs[0][torch.argmax(probs)].item())
            probabilities = [float(prob) for prob in probs.cpu().numpy()[0]]
            
            return {
                'sentiment': sentiment,
                'confidence': confidence,
                'probabilities': probabilities
            }
            
        except Exception as e:
            self.logger.error(f"Sentiment analysis error: {e}")
            return {'sentiment': 'Unknown', 'confidence': 0.0, 'error': str(e)}
    
    def _extract_entities(self, text):
        """استخراج موجودیت‌ها با pipeline بهینه"""
        try:
            if 'ner' not in self.models:
                return []
                
            # استفاده از pipeline با تنظیمات بهینه
            ner_pipeline = pipeline(
                "token-classification",
                model=self.models['ner'],
                tokenizer=self.tokenizers['ner'],
                aggregation_strategy="simple",
                device=0 if torch.cuda.is_available() else -1
            )
            
            entities = ner_pipeline(text[:512])  # محدود کردن طول متن
            
            for entity in entities:
                entity['score'] = float(entity['score'])
            
            return entities
            
        except Exception as e:
            self.logger.error(f"NER extraction error: {e}")
            return []
    
    def _generate_summary(self, text):
        """ایجاد خلاصه متن"""
        try:
            sentences = [s.strip() for s in text.split('.') if s.strip()]
            if len(sentences) > 3:
                return '. '.join(sentences[:2]) + '.'
            return text
        except Exception as e:
            self.logger.error(f"Summary generation error: {e}")
            return text[:200] + "..." if len(text) > 200 else text

# =====================================================
#  Flask API Application - Optimized for IIS
# =====================================================

class NLPAPI:
    def __init__(self):
        self.app = Flask(__name__)
        self.app.config['JSON_SORT_KEYS'] = False
        self.app.config['JSONIFY_PRETTYPRINT_REGULAR'] = False
        self.logger = logging.getLogger(__name__)
        
        # Lazy loading - مدل‌ها هنگام اولین درخواست load می‌شوند
        self.nlp_system = None
        self._setup_routes()
        self._setup_error_handlers()
    
    def _get_nlp_system(self):
        """Lazy loading برای جلوگیری از خطا در startup"""
        if self.nlp_system is None:
            try:
                self.logger.info("Loading NLP system...")
                self.nlp_system = AdvancedNLPSystem()
                self.logger.info("NLP system loaded successfully")
            except Exception as e:
                self.logger.error(f"Failed to load NLP system: {e}")
                self.nlp_system = SimpleNLPSystem()
        return self.nlp_system
    
    def _setup_routes(self):
        """تنظیم routes با CORS support"""
        
        @self.app.route('/analyze', methods=['POST', 'OPTIONS'])
        def analyze_text():
            if request.method == 'OPTIONS':
                return self._build_cors_response()
                
            data = request.get_json(silent=True) or {}
            text = data.get('text', '')
            
            if not text:
                return jsonify({'error': 'No text provided'}), 400
            
            try:
                self.logger.info(f"Analyzing text: {text[:100]}...")
                nlp_system = self._get_nlp_system()  # Lazy loading
                results = nlp_system.analyze_text(text)
                response = jsonify(results)
                return self._add_cors_headers(response)
                
            except Exception as e:
                self.logger.error(f"API error: {e}")
                return jsonify({'error': 'Internal server error'}), 500
        
        @self.app.route('/health', methods=['GET'])
        def health_check():
            return jsonify({
                'status': 'healthy', 
                'device': str(device),
                'timestamp': time.time()
            })
        
        @self.app.route('/', methods=['GET'])
        def home():
            return jsonify({
                'message': 'NLP API Server is Running',
                'version': '1.0.0',
                'endpoints': {
                    'analyze': 'POST /analyze',
                    'health': 'GET /health'
                }
            })
    
    def _setup_error_handlers(self):
        """مدیریت خطاها"""
        
        @self.app.errorhandler(404)
        def not_found(error):
            return jsonify({'error': 'Endpoint not found'}), 404
        
        @self.app.errorhandler(500)
        def internal_error(error):
            self.logger.error(f"Internal server error: {error}")
            return jsonify({'error': 'Internal server error'}), 500
    
    def _build_cors_response(self):
        """پاسخ برای CORS preflight"""
        response = jsonify({'status': 'ok'})
        return self._add_cors_headers(response)
    
    def _add_cors_headers(self, response):
        """اضافه کردن headers برای CORS"""
        response.headers.add('Access-Control-Allow-Origin', '*')
        response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
        response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
        return response
    
    def run(self, host='0.0.0.0', port=5000, debug=False):
      # غیرفعال کردن debug در production
      self.app.run(
          host=host, 
          port=port, 
          debug=False,  # تغییر به False
          threaded=True)

# =====================================================
#  WSGI Configuration for IIS
# =====================================================

# ایجاد application instance برای IIS
def create_app():
    """ایجاد برنامه با تنظیمات سبک برای IIS"""
    api = NLPAPI()
    return api.app

# Global application instance - این متغیر باید وجود داشته باشد
application = create_app()

# =====================================================
#  Development Server
# =====================================================

if __name__ == '__main__':
    print("=" * 60)
    print("🤖 NLP API Server - Development Mode")
    print("=" * 60)
    
    api = NLPAPI()
    api.run(debug=True, port=5000)