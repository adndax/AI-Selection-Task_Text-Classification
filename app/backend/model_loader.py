import pickle
import numpy as np
import os
import sys
import re
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')

current_dir = os.path.dirname(os.path.abspath(__file__))
ml_scripts_path = os.path.join(current_dir, 'ml', 'scripts')
if os.path.exists(ml_scripts_path):
    sys.path.append(ml_scripts_path)

try:
    from preprocessing import TextPreprocessor
    PREPROCESSING_AVAILABLE = True
except ImportError:
    PREPROCESSING_AVAILABLE = False

class ModelPredictor:
    def __init__(self, models_dir='ml/models'):
        self.models_dir = os.path.abspath(models_dir)
        self.baseline_model = None
        self.label_encoder = None
        self.tfidf_vectorizer = None
        self.baseline_preprocessor = None
        
        self._initialize_preprocessors()
        self._load_models()
    
    def _initialize_preprocessors(self):
        if PREPROCESSING_AVAILABLE:
            kamus_path = os.path.join('ml', 'dataset', 'kamus_singkatan.csv')
            if os.path.exists(kamus_path):
                self.baseline_preprocessor = TextPreprocessor(
                    kamus_path=kamus_path,
                    use_stemming=True,
                    use_stopword_removal=True
                )
    
    def _load_models(self):
        if not os.path.exists(self.models_dir):
            raise FileNotFoundError(f"Models directory not found: {self.models_dir}")
        
        self._load_tfidf_vectorizer()
        self._load_baseline_model()
        self._load_label_encoder()
    
    def _load_tfidf_vectorizer(self):
        path = os.path.join(self.models_dir, 'tfidf_vectorizer.pkl')
        if os.path.exists(path):
            with open(path, 'rb') as f:
                self.tfidf_vectorizer = pickle.load(f)
    
    def _load_baseline_model(self):
        path = os.path.join(self.models_dir, 'best_baseline_model.pkl')
        if os.path.exists(path):
            with open(path, 'rb') as f:
                data = pickle.load(f)
                if isinstance(data, dict) and 'model' in data:
                    self.baseline_model = data['model']
                else:
                    self.baseline_model = data
    
    def _load_label_encoder(self):
        path = os.path.join(self.models_dir, 'label_encoder.pkl')
        if os.path.exists(path):
            with open(path, 'rb') as f:
                self.label_encoder = pickle.load(f)
    
    def _preprocess_text(self, text):
        if self.baseline_preprocessor:
            return self.baseline_preprocessor.preprocess_single(text)
        else:
            text = str(text).lower()
            text = re.sub(r'[^a-z\s]', '', text)
            text = re.sub(r'\s+', ' ', text).strip()
            return text
    
    def predict(self, text):
        if not self.baseline_model or not self.tfidf_vectorizer:
            raise ValueError("Baseline model or TF-IDF vectorizer not available")
        
        processed_text = self._preprocess_text(text)
        tfidf_features = self.tfidf_vectorizer.transform([processed_text])
        
        if self.label_encoder:
            prediction_encoded = self.baseline_model.predict(tfidf_features)[0]
            prediction = self.label_encoder.inverse_transform([prediction_encoded])[0]
        else:
            prediction = self.baseline_model.predict(tfidf_features)[0]
        
        probabilities = {}
        confidence = 0.0
        
        if hasattr(self.baseline_model, 'predict_proba'):
            proba = self.baseline_model.predict_proba(tfidf_features)[0]
            if self.label_encoder:
                classes = self.label_encoder.classes_
            elif hasattr(self.baseline_model, 'classes_'):
                classes = self.baseline_model.classes_
            else:
                classes = [f"class_{i}" for i in range(len(proba))]
            
            probabilities = dict(zip(classes, proba))
            confidence = float(max(proba))
        else:
            probabilities = {prediction: 1.0}
            confidence = 1.0
        
        return {
            "prediction": str(prediction),
            "confidence": confidence,
            "probabilities": {k: float(v) for k, v in probabilities.items()},
            "model_info": f"TF-IDF + {type(self.baseline_model).__name__}",
            "processed_text": processed_text
        }
    
    def is_ready(self):
        return self.baseline_model is not None and self.tfidf_vectorizer is not None
    
    def get_status(self):
        return {
            "model_available": self.baseline_model is not None and self.tfidf_vectorizer is not None,
            "baseline_model_loaded": self.baseline_model is not None,
            "tfidf_vectorizer_loaded": self.tfidf_vectorizer is not None,
            "label_encoder_loaded": self.label_encoder is not None,
            "preprocessing_available": PREPROCESSING_AVAILABLE,
            "baseline_preprocessor_ready": self.baseline_preprocessor is not None,
            "models_dir": self.models_dir,
            "models_dir_exists": os.path.exists(self.models_dir),
            "is_ready": self.is_ready(),
            "timestamp": datetime.now().isoformat()
        }