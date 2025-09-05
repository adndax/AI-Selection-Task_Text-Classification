import re
import pandas as pd
import nltk
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from multiprocessing import Pool, cpu_count

KAMUS_PATH = 'dataset/kamus_singkatan.csv'
TEST_SIZE = 0.15
VAL_SIZE = 0.15
RANDOM_STATE = 42
MULTIPROCESSING_THRESHOLD = 500 

def ensure_nltk_data():
    try:
        stopwords.words('indonesian')
    except LookupError:
        nltk.download('stopwords', quiet=True)

class TextPreprocessor:
    def __init__(self, kamus_path, use_stemming, use_stopword_removal):
        ensure_nltk_data()
        self.kamus_dictionary = self._load_kamus(kamus_path) 
        self.use_stemming = use_stemming
        self.use_stopword_removal = use_stopword_removal
        
        if self.use_stemming:
            self.stemmer = StemmerFactory().create_stemmer()
        else:
            self.stemmer = None
            
        if self.use_stopword_removal:
            self.stop_words = set(stopwords.words('indonesian'))
        else:
            self.stop_words = set()
            
        self._compile_regex()
    
    def _load_kamus(self, path):
        try:
            kamus = pd.read_csv(path, sep=';', header=None, names=['singkatan', 'hasil'])
            return dict(zip(kamus['singkatan'], kamus['hasil']))
        except FileNotFoundError:
            return {}
    
    def _compile_regex(self):
        self.cleanup_pattern = re.compile(
            r'(\[USERNAME\]|\[URL\])|'
            r'(@\w+)|'
            r'(https?://\S+|www\.\S+)|'
            r'(\d+)|'
            r'([^\w\s!?#])',
            re.IGNORECASE
        )
        
        if self.kamus_dictionary:
            sorted_keys = sorted(self.kamus_dictionary.keys(), key=len, reverse=True)
            pattern_parts = [re.escape(word) for word in sorted_keys]
            pattern_str = r'\b(' + '|'.join(pattern_parts) + r')\b'
            self.normalize_pattern = re.compile(pattern_str, re.IGNORECASE)
        else:
            self.normalize_pattern = None
    
    def _normalize_text(self, text):
        if self.normalize_pattern:
            return self.normalize_pattern.sub(
                lambda x: self.kamus_dictionary.get(x.group().lower(), x.group()), text)
        return text
    
    def preprocess_single(self, text):
        if pd.isna(text) or text == '':
            return ""
        
        original_text = str(text)
        
        text = original_text.lower()
        text = self.cleanup_pattern.sub(' ', text)
        text = self._normalize_text(text)
        text = re.sub(r'\s+', ' ', text).strip()
        
        if self.use_stemming and self.stemmer:
            text = self.stemmer.stem(text)
        
        words = text.split()
        
        if self.use_stopword_removal and self.stop_words:
            words = [word for word in words 
                    if word not in self.stop_words and len(word) > 1]
        else:
            words = [word for word in words if len(word) > 1]
            
        final_text = ' '.join(words)
        return final_text
    
    def preprocess_batch(self, texts, n_jobs=1, threshold=MULTIPROCESSING_THRESHOLD):
        if n_jobs == -1:
            n_jobs = min(cpu_count(), max(2, len(texts) // 500))
        
        use_multiprocessing = n_jobs > 1 and len(texts) >= threshold
        
        if not use_multiprocessing:
            return [self.preprocess_single(text) for text in texts]
        else:
            chunk_size = max(10, len(texts) // n_jobs)
            chunks = [texts[i:i + chunk_size] for i in range(0, len(texts), chunk_size)]
            
            with Pool(n_jobs) as pool:
                chunk_results = pool.map(self._process_chunk, chunks)
            
            return [item for sublist in chunk_results for item in sublist]
    
    def _process_chunk(self, chunk):
        return [self.preprocess_single(text) for text in chunk]

def split_dataset(df, test_size, val_size, random_state):
    X, y = df['tweet'], df['label']
    
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=test_size + val_size, stratify=y, random_state=random_state)
    
    val_ratio = val_size / (test_size + val_size)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=1-val_ratio, stratify=y_temp, random_state=random_state)
    
    train_df = pd.DataFrame({'tweet': X_train, 'label': y_train})
    val_df = pd.DataFrame({'tweet': X_val, 'label': y_val})
    test_df = pd.DataFrame({'tweet': X_test, 'label': y_test})
    
    return train_df, val_df, test_df

def preprocess_dataframe(df, text_column, kamus_path, n_jobs, use_stemming, use_stopword_removal):
    preprocessor = TextPreprocessor(kamus_path, use_stemming, use_stopword_removal)
    
    df_work = df.copy()
    
    original_texts = df_work[text_column].tolist()
    processed_texts = preprocessor.preprocess_batch(original_texts, n_jobs=n_jobs)
    
    df_work[text_column] = processed_texts
    df_work = df_work[df_work[text_column] != ''].reset_index(drop=True)
    
    return df_work

def preprocessing_pipeline(df, text_column, use_stemming, use_stopword_removal, n_jobs=4):
    df_processed = preprocess_dataframe(df, text_column, KAMUS_PATH, n_jobs, use_stemming, use_stopword_removal)
    
    result = {'full': df_processed}
    
    if 'label' in df_processed.columns:
        train_df, val_df, test_df = split_dataset(df_processed, TEST_SIZE, VAL_SIZE, RANDOM_STATE)
        
        result.update({
            'train': train_df,
            'validation': val_df,
            'test': test_df
        })
    
    return result

def preprocess_for_baseline_models(df):
    return preprocessing_pipeline(df=df, text_column='tweet', use_stemming=True, use_stopword_removal=True)

def preprocess_for_deep_learning(df):
    return preprocessing_pipeline(df=df, text_column='tweet', use_stemming=False, use_stopword_removal=False)