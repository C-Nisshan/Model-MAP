import os
import re
import subprocess
import numpy as np
import pandas as pd
from scipy import sparse
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, classification_report
from joblib import Parallel, delayed
import lightgbm as lgb

# Optional: autocorrect
try:
    from autocorrect import Speller
    speller = Speller()
    def preprocess_text(text):
        text = str(text).lower()
        text = re.sub(r'[^\w\s]', '', text)
        return ' '.join(speller(word) for word in text.split())
except Exception:
    def preprocess_text(text):
        text = str(text).lower()
        return re.sub(r'[^\w\s]', '', text)

# Paths
TRAIN_PATH = '/kaggle/input/map-charting-student-math-misunderstandings/train.csv'
TEST_PATH  = '/kaggle/input/map-charting-student-math-misunderstandings/test.csv'
OUTPUT_PATH = '/kaggle/working/submission.csv'

# GPU detection
def has_gpu():
    try:
        res = subprocess.run(['nvidia-smi'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return res.returncode == 0
    except Exception:
        return False

USE_GPU = has_gpu()
print("GPU available:", USE_GPU)

# Load data
df = pd.read_csv(TRAIN_PATH, keep_default_na=False)
test_df = pd.read_csv(TEST_PATH, keep_default_na=False)

# Preprocess text + features
for d in (df, test_df):
    d['StudentExplanation'] = d['StudentExplanation'].apply(preprocess_text)
    d['word_count'] = d['StudentExplanation'].apply(lambda x: len(str(x).split()))
    d['short_flag'] = (d['word_count'] < 5).astype(np.int8)
    d['Topic'] = d['QuestionText'].apply(
        lambda x: 'Fractions' if 'fraction' in str(x).lower()
        else 'Probability' if 'probability' in str(x).lower()
        else 'Other'
    )

# TF-IDF vectorization
MAX_FEAT = 15000
vectorizer = TfidfVectorizer(max_features=MAX_FEAT, stop_words='english', ngram_range=(1, 2), min_df=2)
X_text = vectorizer.fit_transform(df['StudentExplanation'])
X_test_text = vectorizer.transform(test_df['StudentExplanation'])

# Extra features
topic_map = {'Fractions': 0, 'Probability': 1, 'Other': 2}
df['topic_id'] = df['Topic'].map(topic_map).astype(np.int32)
test_df['topic_id'] = test_df['Topic'].map(topic_map).astype(np.int32)

X_extra = sparse.csr_matrix(df[['word_count', 'short_flag', 'topic_id']].values.astype(np.float32))
X_test_extra = sparse.csr_matrix(test_df[['word_count', 'short_flag', 'topic_id']].values.astype(np.float32))

# Combine
X = sparse.hstack([X_text, X_extra], format='csr')
X_test = sparse.hstack([X_test_text, X_test_extra], format='csr')

print("Shape X:", X.shape, " Shape X_test:", X_test.shape)

# Encode targets
cat_le = LabelEncoder()
y_cat = cat_le.fit_transform(df['Category'])

# Misconception: keep only rows with non-empty values
mis_mask = df['Misconception'].str.strip() != ''
mis_df = df[mis_mask].copy()
y_mis_le = LabelEncoder()
y_mis = y_mis_le.fit_transform(mis_df['Misconception'])

print("Category classes:", list(cat_le.classes_))
print("Misconception classes:", len(y_mis_le.classes_), "classes (trained on", len(mis_df), "rows)")

# Import LightGBM/XGBoost
try:
    import xgboost as xgb
    HAVE_XGB = True
except:
    HAVE_XGB = False

try:
    import lightgbm as lgb
    HAVE_LGB = True
except:
    HAVE_LGB = False

if USE_GPU and HAVE_XGB:
    chosen_model = 'xgb'
    print("Using XGBoost GPU")
elif HAVE_LGB:
    chosen_model = 'lgb'
    print("Using LightGBM")
elif HAVE_XGB:
    chosen_model = 'xgb'
    print("Using XGBoost CPU")
else:
    raise RuntimeError("No LightGBM/XGBoost installed")

# Class weight helper
def get_class_weights(y):
    counts = Counter(y)
    total = sum(counts.values())
    n_classes = len(counts)
    return {cls: total / (n_classes * cnt) for cls, cnt in counts.items()}

# Training functions
def train_with_lgb(X, y, X_val=None, y_val=None, num_round=1000, early_stopping=50):
    params = {
        "objective": "multiclass",
        "metric": "multi_logloss",
        "num_class": len(np.unique(y)),
        "learning_rate": 0.05,
        "num_leaves": 31,
        "max_depth": -1,
        "verbosity": -1,
        "n_jobs": max(1, os.cpu_count() - 1),
        "seed": 42
    }
    if USE_GPU:
        params.update({"device": "gpu", "gpu_platform_id": 0, "gpu_device_id": 0})

    # Compute sample weights
    class_weights = get_class_weights(y)
    sample_weights = np.array([class_weights[cls] for cls in y])

    train_set = lgb.Dataset(X, label=y, weight=sample_weights)
    valid_sets = [train_set]
    valid_names = ['train']
    if X_val is not None and y_val is not None:
        # Compute sample weights for validation set
        val_sample_weights = np.array([class_weights[cls] for cls in y_val])
        valid_sets.append(lgb.Dataset(X_val, label=y_val, weight=val_sample_weights))
        valid_names.append('valid')

    model = lgb.train(
        params,
        train_set,
        num_boost_round=num_round,
        valid_sets=valid_sets,
        valid_names=valid_names,
        callbacks=[lgb.early_stopping(stopping_rounds=early_stopping, verbose=False)]
    )
    return model

def train_with_xgb(X, y, X_val=None, y_val=None, num_round=1000, early_stopping=50):
    dtrain = xgb.DMatrix(X, label=y)
    evals = [(dtrain, 'train')]
    params = {
        "objective": "multi:softprob",
        "eval_metric": "mlogloss",
        "num_class": len(np.unique(y)),
        "eta": 0.05,
        "max_depth": 6,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "seed": 42
    }
    if USE_GPU:
        params.update({"tree_method": "gpu_hist", "predictor": "gpu_predictor"})
    if X_val is not None:
        deval = xgb.DMatrix(X_val, label=y_val)
        evals.append((deval, 'valid'))
    model = xgb.train(params, dtrain, num_boost_round=num_round, evals=evals,
                      early_stopping_rounds=early_stopping, verbose_eval=False)
    return model

# CV training
def cv_train_target(X, y, target_name, model_type='lgb', n_splits=4):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    n_classes = len(np.unique(y))
    oof_proba = np.zeros((X.shape[0], n_classes))
    test_proba = np.zeros((X_test.shape[0], n_classes))
    models = []

    for fold, (tr_idx, val_idx) in enumerate(skf.split(X, y)):
        print(f"{target_name} Fold {fold+1}/{n_splits}")
        X_tr, X_val = X[tr_idx], X[val_idx]
        y_tr, y_val = y[tr_idx], y[val_idx]

        if model_type == 'lgb':
            model = train_with_lgb(X_tr, y_tr, X_val, y_val)
            oof_proba[val_idx] = model.predict(X_val)
            test_proba += model.predict(X_test) / n_splits
        else:
            model = train_with_xgb(X_tr, y_tr, X_val, y_val)
            dval = xgb.DMatrix(X_val)
            dtest = xgb.DMatrix(X_test)
            oof_proba[val_idx] = model.predict(dval)
            test_proba += model.predict(dtest) / n_splits
        models.append(model)

    return oof_proba, test_proba, models

# Train Category
cat_oof, cat_test_proba, _ = cv_train_target(X, y_cat, "Category", chosen_model)

# Train Misconception only on available rows
X_mis = X[mis_mask.values]
X_test_mis = X_test
mis_oof, mis_test_proba, _ = cv_train_target(X_mis, y_mis, "Misconception", chosen_model)

# Evaluate
cat_preds = np.argmax(cat_oof, axis=1)
print("\nCategory Classification Report:\n", classification_report(y_cat, cat_preds))

mis_preds = np.argmax(mis_oof, axis=1)
print("\nMisconception Classification Report (non-missing only):\n", classification_report(y_mis, mis_preds))

# Build final top-3 predictions
cat_labels = cat_le.inverse_transform(np.arange(cat_test_proba.shape[1]))
mis_labels = y_mis_le.inverse_transform(np.arange(mis_test_proba.shape[1]))

final_predictions = []
for i in range(X_test.shape[0]):
    ranked_pairs = []
    for cat_idx, cat_label in enumerate(cat_labels):
        for mis_idx, mis_label in enumerate(mis_labels):
            score = cat_test_proba[i, cat_idx] * mis_test_proba[i % mis_test_proba.shape[0], mis_idx]
            ranked_pairs.append((f"{cat_label}:{mis_label}", score))
    ranked_pairs.sort(key=lambda x: x[1], reverse=True)
    top_3 = [pair[0] for pair in ranked_pairs[:3]]
    final_predictions.append(" ".join(top_3))

submission = pd.DataFrame({
    'row_id': test_df['row_id'],
    'Category:Misconception': final_predictions
})
submission.to_csv(OUTPUT_PATH, index=False)
print(f"\nSaved submission to {OUTPUT_PATH}")