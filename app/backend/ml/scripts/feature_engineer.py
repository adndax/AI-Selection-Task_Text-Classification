import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from gensim.models import KeyedVectors
import fasttext
import fasttext.util
import pickle
import warnings
warnings.filterwarnings('ignore')

class TFIDFFeatureExtractor:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            max_features=10000,
            ngram_range=(1, 2),
            token_pattern=r'\b\w+\b'
        )
        
    def fit_transform(self, texts):
        return self.vectorizer.fit_transform(texts)
    
    def transform(self, texts):
        return self.vectorizer.transform(texts)

class WordEmbeddingExtractor:
    def __init__(self, embedding_dim=300):
        self.embedding_dim = embedding_dim
        self.word_vectors = None
        
    def load_embeddings(self, model_path=None):
        if model_path is None:
            fasttext.util.download_model('id', if_exists='ignore')
            self.word_vectors = fasttext.load_model('cc.id.300.bin')
        else:
            self.word_vectors = KeyedVectors.load_word2vec_format(model_path, binary=False)
    
    def _get_word_embedding(self, word):
        try:
            if hasattr(self.word_vectors, 'get_word_vector'):  # FastText API
                return self.word_vectors.get_word_vector(word)
            elif hasattr(self.word_vectors, 'get_vector'):     # Gensim API
                return self.word_vectors.get_vector(word)
            else:
                return self.word_vectors[word]
        except:
            return np.zeros(self.embedding_dim)
    
    def _pool_embeddings(self, embeddings):
        if len(embeddings) == 0:
            return np.zeros(self.embedding_dim)
        
        embeddings = np.array(embeddings)
        return np.mean(embeddings, axis=0)
    
    def transform_texts(self, texts):
        embeddings_list = []
        
        for text in texts:
            if pd.isna(text) or text == '':
                embedding = np.zeros(self.embedding_dim)
            else:
                words = text.split()
                word_embeddings = [self._get_word_embedding(word) for word in words]
                embedding = self._pool_embeddings(word_embeddings)
            
            embeddings_list.append(embedding)
        
        return np.array(embeddings_list)

def extract_tfidf_features(splits):
    extractor = TFIDFFeatureExtractor()
    
    train_texts = splits['train']['tweet'].tolist()
    val_texts = splits['validation']['tweet'].tolist()
    test_texts = splits['test']['tweet'].tolist()
    
    X_train = extractor.fit_transform(train_texts)
    X_val = extractor.transform(val_texts)
    X_test = extractor.transform(test_texts)
    
    y_train = splits['train']['label'].values
    y_val = splits['validation']['label'].values
    y_test = splits['test']['label'].values
    
    return {
        'X_train': X_train,
        'X_val': X_val,
        'X_test': X_test,
        'y_train': y_train,
        'y_val': y_val,
        'y_test': y_test
    }

def extract_embedding_features(splits):
    extractor = WordEmbeddingExtractor()
    extractor.load_embeddings()
    
    train_texts = splits['train']['tweet'].tolist()
    val_texts = splits['validation']['tweet'].tolist() 
    test_texts = splits['test']['tweet'].tolist()
    
    X_train = extractor.transform_texts(train_texts)
    X_val = extractor.transform_texts(val_texts)
    X_test = extractor.transform_texts(test_texts)
    
    y_train = splits['train']['label'].values
    y_val = splits['validation']['label'].values
    y_test = splits['test']['label'].values
    
    return {
        'X_train': X_train,
        'X_val': X_val,
        'X_test': X_test,
        'y_train': y_train,
        'y_val': y_val,
        'y_test': y_test
    }

def balance_data(features, strategy='smote'):
    if strategy == 'smote':
        sampler = SMOTE(random_state=42)
    elif strategy == 'undersampling':
        sampler = RandomUnderSampler(random_state=42)
    
    X_train_balanced, y_train_balanced = sampler.fit_resample(features['X_train'], features['y_train'])
    
    balanced_features = features.copy()
    balanced_features['X_train'] = X_train_balanced
    balanced_features['y_train'] = y_train_balanced
    
    return balanced_features