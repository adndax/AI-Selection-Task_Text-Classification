import numpy as np
import pandas as pd
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import pickle
import os
from datetime import datetime
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import warnings
warnings.filterwarnings('ignore')

MODELS_DIR = 'models'

class BaselineModelTrainer:
    def __init__(self):
        self.models = {
            'multinomial_nb': MultinomialNB(alpha=1.0),
            'svm_linear': SVC(kernel='linear', C=1.0, random_state=42),
            'logistic_regression': LogisticRegression(C=1.0, max_iter=1000, random_state=42)
        }
        self.trained_models = {}
        
    def train_model(self, model_name, X_train, y_train):
        model = self.models[model_name]
        model.fit(X_train, y_train)
        return model
    
    def train_all_models(self, tfidf_features):
        results = {}
        
        for model_name in self.models.keys():
            model = self.train_model(model_name, tfidf_features['X_train'], tfidf_features['y_train'])
            results[model_name] = model
            
            self.trained_models[f"tfidf_{model_name}"] = {
                'model': model,
                'feature_type': 'tfidf',
                'algorithm': model_name
            }
        
        return results

class DeepLearningModelTrainer:
    def __init__(self):
        self.label_encoder = LabelEncoder()
        self.trained_models = {}
    
    def create_feedforward_nn(self, input_dim, num_classes):
        model = Sequential([
            Dense(512, activation='relu', input_dim=input_dim),
            Dropout(0.5),
            Dense(256, activation='relu'),
            Dropout(0.3),
            Dense(128, activation='relu'),
            Dropout(0.2),
            Dense(num_classes, activation='softmax')
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def train_feedforward(self, X_train, y_train, X_val, y_val):
        y_train_encoded = self.label_encoder.fit_transform(y_train)
        y_val_encoded = self.label_encoder.transform(y_val)
        
        input_dim = X_train.shape[1]
        num_classes = len(np.unique(y_train_encoded))
        
        model = self.create_feedforward_nn(input_dim, num_classes)
        
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=15,
            restore_best_weights=True
        )
        
        history = model.fit(
            X_train, y_train_encoded,
            validation_data=(X_val, y_val_encoded),
            epochs=100,
            batch_size=32,
            callbacks=[early_stopping],
            verbose=0
        )
        
        return model, history
    
    def train_all_models(self, embedding_features):
        X_train = embedding_features['X_train']
        y_train = embedding_features['y_train']
        X_val = embedding_features['X_val']
        y_val = embedding_features['y_val']
        
        feedforward_model, ff_history = self.train_feedforward(X_train, y_train, X_val, y_val)
        
        self.trained_models[f"embedding_feedforward_nn"] = {
            'model': feedforward_model,
            'feature_type': 'embedding',
            'algorithm': 'feedforward_nn',
            'label_encoder': self.label_encoder
        }
        
        return {'feedforward_nn': feedforward_model}

class BaselineEvaluator:
    def __init__(self):
        self.evaluation_results = {}
    
    def evaluate_model(self, model, X_test, y_test):
        y_pred = model.predict(X_test)
        
        return {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
            'recall': recall_score(y_test, y_pred, average='weighted', zero_division=0),
            'f1_score': f1_score(y_test, y_pred, average='weighted', zero_division=0),
            'classification_report': classification_report(y_test, y_pred, zero_division=0),
            'confusion_matrix': confusion_matrix(y_test, y_pred),
            'predictions': y_pred,
            'true_labels': y_test
        }
    
    def evaluate_all_models(self, baseline_models, tfidf_features):
        results = {}
        
        for model_name, model in baseline_models.items():
            results[model_name] = self.evaluate_model(
                model, tfidf_features['X_test'], tfidf_features['y_test']
            )
        
        self.evaluation_results = results
        return results

class DeepLearningEvaluator:
    def __init__(self):
        self.evaluation_results = {}
    
    def evaluate_feedforward_model(self, model, X_test, y_test, label_encoder):
        y_test_encoded = label_encoder.transform(y_test)
        y_pred_proba = model.predict(X_test, verbose=0)
        y_pred_encoded = np.argmax(y_pred_proba, axis=1)
        y_pred = label_encoder.inverse_transform(y_pred_encoded)
        
        return {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
            'recall': recall_score(y_test, y_pred, average='weighted', zero_division=0),
            'f1_score': f1_score(y_test, y_pred, average='weighted', zero_division=0),
            'classification_report': classification_report(y_test, y_pred, zero_division=0),
            'confusion_matrix': confusion_matrix(y_test, y_pred),
            'predictions': y_pred,
            'true_labels': y_test
        }
    
    def evaluate_all_models(self, dl_models, embedding_features, dl_trainer):
        results = {}
        
        for model_name, model in dl_models.items():
            if model_name == 'feedforward_nn':
                results[model_name] = self.evaluate_feedforward_model(
                    model, embedding_features['X_test'], embedding_features['y_test'], 
                    dl_trainer.label_encoder
                )
        
        self.evaluation_results = results
        return results

def create_results_summary(baseline_results, dl_results):
    summary_data = []
    
    for model_name, metrics in baseline_results.items():
        summary_data.append({
            'Model_Type': 'baseline',
            'Algorithm': model_name,
            'Accuracy': metrics['accuracy'],
            'Precision': metrics['precision'],
            'Recall': metrics['recall'],
            'F1_Score': metrics['f1_score']
        })
    
    for model_name, metrics in dl_results.items():
        summary_data.append({
            'Model_Type': 'deep_learning',
            'Algorithm': model_name,
            'Accuracy': metrics['accuracy'],
            'Precision': metrics['precision'],
            'Recall': metrics['recall'],
            'F1_Score': metrics['f1_score']
        })
    
    return pd.DataFrame(summary_data).sort_values('F1_Score', ascending=False)

def create_results_summary(baseline_under, baseline_smote, dl_under, dl_smote):
    summary_data = []
    
    for model_name, metrics in baseline_under.items():
        summary_data.append({
            'Model_Type': 'baseline',
            'Algorithm': model_name,
            'Balance_Method': 'undersampling',
            'Accuracy': metrics['accuracy'],
            'Precision': metrics['precision'],
            'Recall': metrics['recall'],
            'F1_Score': metrics['f1_score']
        })
    
    for model_name, metrics in baseline_smote.items():
        summary_data.append({
            'Model_Type': 'baseline',
            'Algorithm': model_name,
            'Balance_Method': 'smote',
            'Accuracy': metrics['accuracy'],
            'Precision': metrics['precision'],
            'Recall': metrics['recall'],
            'F1_Score': metrics['f1_score']
        })
    
    for model_name, metrics in dl_under.items():
        summary_data.append({
            'Model_Type': 'deep_learning',
            'Algorithm': model_name,
            'Balance_Method': 'undersampling',
            'Accuracy': metrics['accuracy'],
            'Precision': metrics['precision'],
            'Recall': metrics['recall'],
            'F1_Score': metrics['f1_score']
        })
    
    for model_name, metrics in dl_smote.items():
        summary_data.append({
            'Model_Type': 'deep_learning',
            'Algorithm': model_name,
            'Balance_Method': 'smote',
            'Accuracy': metrics['accuracy'],
            'Precision': metrics['precision'],
            'Recall': metrics['recall'],
            'F1_Score': metrics['f1_score']
        })
    
    return pd.DataFrame(summary_data).sort_values('F1_Score', ascending=False)

def save_model(model, filename, feature_type, algorithm):
    os.makedirs(MODELS_DIR, exist_ok=True)
    
    if not filename.endswith('.pkl'):
        filename += '.pkl'
    
    filepath = os.path.join(MODELS_DIR, filename)
    
    model_data = {
        'model': model,
        'feature_type': feature_type,
        'algorithm': algorithm,
        'timestamp': datetime.now().strftime("%Y%m%d_%H%M%S")
    }
    
    with open(filepath, 'wb') as f:
        pickle.dump(model_data, f)
    
    return filepath

def train_baseline_models(tfidf_features):
    trainer = BaselineModelTrainer()
    models = trainer.train_all_models(tfidf_features)
    return models, trainer

def train_deep_learning_models(embedding_features):
    trainer = DeepLearningModelTrainer()
    models = trainer.train_all_models(embedding_features)
    return models, trainer

def evaluate_baseline_models(baseline_models, tfidf_features):
    evaluator = BaselineEvaluator()
    results = evaluator.evaluate_all_models(baseline_models, tfidf_features)
    return results

def evaluate_deep_learning_models(dl_models, embedding_features, dl_trainer):
    evaluator = DeepLearningEvaluator()
    results = evaluator.evaluate_all_models(dl_models, embedding_features, dl_trainer)
    return results