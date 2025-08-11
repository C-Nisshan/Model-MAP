import os
import re
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer

# Add DLL directory BEFORE importing lightgbm
if os.name == 'nt':  # Windows
    dll_path = os.path.abspath(os.path.join('venv', 'lib', 'site-packages', 'lightgbm', 'bin'))
    if os.path.exists(dll_path):
        os.add_dll_directory(dll_path)
        print(f"Added DLL path: {dll_path}")
    else:
        print(f"Warning: DLL path {dll_path} not found. Ensure LightGBM is installed correctly.")

import lightgbm as lgb  

# Paths and configuration from the main model
MODELS_DIR = 'models'
TRAIN_PATH = 'data/train.csv'
TEST_PATH = 'data/test.csv'

# Load data
try:
    df = pd.read_csv(TRAIN_PATH, keep_default_na=False)
    test_df = pd.read_csv(TEST_PATH, keep_default_na=False)
    print("Data loaded successfully")
except FileNotFoundError as e:
    print(f"Error: Ensure train.csv and test.csv are in the 'data/' directory. {e}")
    exit(1)

# Replicate preprocessing and feature engineering
def preprocess_text(text):
    text = str(text).lower()
    text = re.sub(r'[^\w\s]', '', text)
    return text

for d in (df, test_df):
    d['StudentExplanation'] = d['StudentExplanation'].apply(preprocess_text)
    d['QuestionText'] = d['QuestionText'].apply(preprocess_text)
    d['word_count'] = d['StudentExplanation'].apply(lambda x: len(str(x).split())).astype(np.int32)
    d['short_flag'] = (d['word_count'] < 5).astype(np.int8)
    d['contains_answer'] = d.apply(
        lambda r: str(r['MC_Answer']).lower() in r['StudentExplanation'], axis=1
    ).astype(np.int8)
print("Preprocessing completed")

# Load vectorizers
try:
    word_vect = joblib.load(os.path.join(MODELS_DIR, 'word_vect.pkl'))
    char_vect = joblib.load(os.path.join(MODELS_DIR, 'char_vect.pkl'))
    ques_vect = joblib.load(os.path.join(MODELS_DIR, 'ques_vect.pkl'))
    print("Vectorizers loaded successfully")
except FileNotFoundError as e:
    print(f"Error: Ensure vectorizer files (word_vect.pkl, char_vect.pkl, ques_vect.pkl) are in the 'models/' directory. {e}")
    exit(1)

# Generate feature names
word_features = word_vect.get_feature_names_out()
char_features = char_vect.get_feature_names_out()
ques_features = ques_vect.get_feature_names_out()
meta_features = ['word_count', 'short_flag', 'contains_answer']
feature_names = list(word_features) + list(char_features) + list(ques_features) + meta_features
print(f"Generated {len(feature_names)} feature names")

# Load a trained model (example: first fold of top-level classifier)
try:
    model = joblib.load(os.path.join(MODELS_DIR, 'is_true_model_fold0.pkl'))
    print("Model loaded successfully")
except FileNotFoundError as e:
    print(f"Error: Ensure model file 'is_true_model_fold0.pkl' is in the 'models/' directory. {e}")
    exit(1)

# Plot feature importance
plt.figure(figsize=(10, 6))
lgb.plot_importance(model, max_num_features=10, importance_type='gain', title='Top 10 Feature Importance')
plt.tight_layout()

# Save the plot
try:
    plt.savefig('feature_importance.png')
    print("Feature importance plot saved as 'feature_importance.png'")
except Exception as e:
    print(f"Error saving plot: {e}")
plt.close()
