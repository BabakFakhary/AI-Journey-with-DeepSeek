#                                                                             به نام خدا
# app/nlp_system.py
import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification,
    AutoModelForTokenClassification,
    pipeline
)
import logging

logger = logging.getLogger(__name__)

class SimpleNLPSystem:
    def analyze_text(self, text):
        return {
            'sentiment': {'sentiment': 'Neutral', 'confidence': 0.5},
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
        self.logger.info("Loading models for production...")
        
        try:
            SMALL_MODELS = {
                'classification': 'distilbert-base-uncased-finetuned-sst-2-english',
                'ner': 'dslim/bert-base-NER',
            }
            
            # مدل برای classification
            self.logger.info("Loading classification model...")
            self.models['classification'] = AutoModelForSequenceClassification.from_pretrained(
                SMALL_MODELS['classification'],
                cache_dir="./model_cache"
            )
            self.tokenizers['classification'] = AutoTokenizer.from_pretrained(
                SMALL_MODELS['classification'],
                cache_dir="./model_cache"
            )
            
            # مدل برای NER
            self.logger.info("Loading NER model...")
            self.models['ner'] = AutoModelForTokenClassification.from_pretrained(
                SMALL_MODELS['ner'],
                cache_dir="./model_cache"
            )
            self.tokenizers['ner'] = AutoTokenizer.from_pretrained(
                SMALL_MODELS['ner'],
                cache_dir="./model_cache"
            )
            
            # انتقال به GPU اگر موجود باشد
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            for name, model in self.models.items():
                model.to(device)
                model.eval()
            
            self.logger.info("All models loaded successfully!")
            
        except Exception as e:
            self.logger.error(f"Error loading models: {e}")
            self.models = {}
            self.tokenizers = {}
    
    def analyze_text(self, text):
        try:
            if not self.models:
                simple_system = SimpleNLPSystem()
                return simple_system.analyze_text(text)
                
            results = {}
            results['sentiment'] = self._analyze_sentiment(text)
            results['entities'] = self._extract_entities(text)
            results['summary'] = self._generate_summary(text)
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error in text analysis: {e}")
            return {'error': str(e)}
    
    def _analyze_sentiment(self, text):
        try:
            if 'classification' not in self.models:
                return {'sentiment': 'Neutral', 'confidence': 0.5}
                
            inputs = self.tokenizers['classification'](
                text, 
                return_tensors="pt", 
                truncation=True, 
                padding=True,
                max_length=512
            )
            
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.models['classification'](**inputs)
            
            probs = torch.softmax(outputs.logits, dim=1)
            sentiment_idx = torch.argmax(probs).item()
            sentiment = ["NEGATIVE", "POSITIVE"][sentiment_idx]
            confidence = float(probs[0][sentiment_idx].item())
            
            return {
                'sentiment': sentiment,
                'confidence': confidence,
                'probabilities': probs.cpu().numpy()[0].tolist()
            }
            
        except Exception as e:
            self.logger.error(f"Sentiment analysis error: {e}")
            return {'sentiment': 'Unknown', 'confidence': 0.0}
    
    def _extract_entities(self, text):
        try:
            if 'ner' not in self.models:
                return []
                
            ner_pipeline = pipeline(
                "ner",
                model=self.models['ner'],
                tokenizer=self.tokenizers['ner'],
                aggregation_strategy="simple",
                device=0 if torch.cuda.is_available() else -1
            )
            
            entities = ner_pipeline(text[:512])
            
            for entity in entities:
                entity['score'] = float(entity['score'])
            
            return entities
            
        except Exception as e:
            self.logger.error(f"NER extraction error: {e}")
            return []
    
    def _generate_summary(self, text):
        try:
            sentences = [s.strip() for s in text.split('.') if s.strip()]
            if len(sentences) > 3:
                return '. '.join(sentences[:2]) + '.'
            return text
        except Exception as e:
            self.logger.error(f"Summary generation error: {e}")
            return text[:200] + "..." if len(text) > 200 else text