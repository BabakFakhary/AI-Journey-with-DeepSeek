#                                                                             به نام خدا
# app/main.py
import os
import time
import logging
from logging.handlers import RotatingFileHandler
from flask import Flask, request, jsonify, render_template
import torch
from app.nlp_system import AdvancedNLPSystem, SimpleNLPSystem

# تنظیمات logging
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
                maxBytes=10485760,
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

app = Flask(__name__)
app.config['JSON_SORT_KEYS'] = False
app.config['JSONIFY_PRETTYPRINT_REGULAR'] = False

# Lazy loading برای مدل‌ها
nlp_system = None

def get_nlp_system():
    global nlp_system
    if nlp_system is None:
        try:
            logger.info("Loading NLP system...")
            nlp_system = AdvancedNLPSystem()
            logger.info("NLP system loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load NLP system: {e}")
            nlp_system = SimpleNLPSystem()
    return nlp_system

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/analyze', methods=['POST', 'OPTIONS'])
def analyze_text():
    if request.method == 'OPTIONS':
        response = jsonify({'status': 'ok'})
        response.headers.add('Access-Control-Allow-Origin', '*')
        response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
        response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
        return response
        
    data = request.get_json(silent=True) or {}
    text = data.get('text', '')
    
    if not text:
        return jsonify({'error': 'No text provided'}), 400
    
    try:
        logger.info(f"Analyzing text: {text[:100]}...")
        system = get_nlp_system()
        results = system.analyze_text(text)
        
        response = jsonify(results)
        response.headers.add('Access-Control-Allow-Origin', '*')
        return response
        
    except Exception as e:
        logger.error(f"API error: {e}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy',
        'device': str(device),
        'timestamp': time.time(),
        'models_loaded': nlp_system is not None and hasattr(nlp_system, 'models') and bool(nlp_system.models)
    })

@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    logger.error(f"Internal server error: {error}")
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    logger.info("Starting NLP API Server...")
    app.run(host='0.0.0.0', port=5000, debug=False)