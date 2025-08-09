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
from sklearn.utils import resample
import lightgbm as lgb

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

# --- Preprocessing ---
def preprocess_text(text):
    text = str(text).lower()
    text = re.sub(r'[^\w\s]', '', text)
    return text

# Load data
df = pd.read_csv(TRAIN_PATH, keep_default_na=False)
test_df = pd.read_csv(TEST_PATH, keep_default_na=False)

# Clean text + basic numeric features
for d in (df, test_df):
    d['StudentExplanation'] = d['StudentExplanation'].apply(preprocess_text)
    d['QuestionText'] = d['QuestionText'].apply(preprocess_text)
    d['word_count'] = d['StudentExplanation'].apply(lambda x: len(str(x).split())).astype(np.int32)
    d['short_flag'] = (d['word_count'] < 5).astype(np.int8)
    d['contains_answer'] = d.apply(
        lambda r: str(r['MC_Answer']).lower() in r['StudentExplanation'],
        axis=1
    ).astype(np.int8)

# Encode target labels
cat_le = LabelEncoder()
df['Category_enc'] = cat_le.fit_transform(df['Category'])
df['is_true'] = df['Category'].apply(lambda x: 1 if x.startswith("True") else 0)

# Misconception encoding
df['Misconception_full'] = df['Misconception'].replace('', 'None')
mis_le = LabelEncoder()
df['Misconception_enc'] = mis_le.fit_transform(df['Misconception_full'])

# --- Fit TF-IDF once globally ---
print("Fitting TF-IDF vectorizers once...")
word_vect = TfidfVectorizer(max_features=10000, ngram_range=(1, 2), min_df=2, n_jobs=-1)
char_vect = TfidfVectorizer(max_features=6000, analyzer='char_wb', ngram_range=(3, 5), n_jobs=-1)
ques_vect = TfidfVectorizer(max_features=2000, ngram_range=(1, 2), min_df=1, n_jobs=-1)

combined_expl = pd.concat([df['StudentExplanation'], test_df['StudentExplanation']])
combined_qtxt = pd.concat([df['QuestionText'], test_df['QuestionText']])

word_vect.fit(combined_expl)
char_vect.fit(combined_expl)
ques_vect.fit(combined_qtxt)

# Transform train/test once
Xw = word_vect.transform(df['StudentExplanation'])
Xc = char_vect.transform(df['StudentExplanation'])
Xq = ques_vect.transform(df['QuestionText'])

Xt_w = word_vect.transform(test_df['StudentExplanation'])
Xt_c = char_vect.transform(test_df['StudentExplanation'])
Xt_q = ques_vect.transform(test_df['QuestionText'])

# Merge with numeric features
num_train = sparse.csr_matrix(df[['word_count', 'short_flag', 'contains_answer']].values.astype(np.float32))
num_test  = sparse.csr_matrix(test_df[['word_count', 'short_flag', 'contains_answer']].values.astype(np.float32))

X_train_full = sparse.hstack([Xw, Xc, Xq, num_train]).tocsr()
X_test_full  = sparse.hstack([Xt_w, Xt_c, Xt_q, num_test]).tocsr()

# --- LightGBM helper ---
def train_lgb(X, y, X_val, y_val):
    params = {
        "objective": "multiclass",
        "metric": "multi_logloss",
        "num_class": len(np.unique(y)),
        "learning_rate": 0.05,
        "num_leaves": 31,
        "max_depth": -1,
        "verbosity": -1,
        "n_jobs": max(1, os.cpu_count() - 1),
        "seed": 42,
        "class_weight": "balanced",
    }
    if USE_GPU:
        params.update({"device": "gpu", "gpu_platform_id": 0, "gpu_device_id": 0})

    train_set = lgb.Dataset(X, label=y, free_raw_data=False)
    val_set = lgb.Dataset(X_val, label=y_val, free_raw_data=False)

    model = lgb.train(
        params,
        train_set,
        num_boost_round=1000,
        valid_sets=[train_set, val_set],
        callbacks=[
            lgb.early_stopping(stopping_rounds=50),
            lgb.log_evaluation(period=0)
        ]
    )
    return model

def oversample(X, y, min_samples=200):
    idxs = []
    y = np.array(y)
    for cls in np.unique(y):
        cls_idx = np.where(y == cls)[0]
        if len(cls_idx) < min_samples:
            cls_idx = resample(cls_idx, replace=True, n_samples=min_samples, random_state=42)
        idxs.extend(cls_idx)
    idxs = np.array(idxs)
    return X[idxs], y[idxs]

# --- Hierarchical Prediction ---
def hierarchical_predict(train_df, test_df, X_train, X_test, topk=3):
    skf = StratifiedKFold(n_splits=4, shuffle=True, random_state=42)

    true_oof = np.zeros(len(train_df), dtype=int)
    true_test_preds = np.zeros((len(test_df), 2))

    # Step 1: Binary True/False
    for tr_idx, val_idx in skf.split(train_df, train_df['is_true']):
        X_tr, X_val = X_train[tr_idx], X_train[val_idx]
        y_tr, y_val = train_df['is_true'].values[tr_idx], train_df['is_true'].values[val_idx]

        model = train_lgb(X_tr, y_tr, X_val, y_val)
        true_oof[val_idx] = model.predict(X_val).argmax(axis=1)
        true_test_preds += model.predict(X_test)

    true_test_preds /= skf.n_splits
    true_test = true_test_preds.argmax(axis=1)

    # Step 2: Category within each branch
    sub_oof = np.zeros(len(train_df), dtype=int)
    sub_test_preds = np.zeros((len(test_df), len(cat_le.classes_)))

    for branch in [0, 1]:
        branch_idx = np.where(train_df['is_true'].values == branch)[0]
        branch_df = train_df.iloc[branch_idx]
        if len(branch_df) == 0:
            continue

        branch_le = LabelEncoder()
        branch_y = branch_le.fit_transform(branch_df['Category'].values)

        branch_test_idx = np.where(true_test == branch)[0]
        if len(np.unique(branch_y)) <= 1:
            if len(branch_test_idx) > 0:
                global_cat_idx = cat_le.transform(branch_le.classes_)[0]
                sub_test_preds[branch_test_idx, global_cat_idx] = 1.0
            continue

        for tr_idx, val_idx in StratifiedKFold(n_splits=4, shuffle=True, random_state=42).split(branch_df, branch_y):
            tr_idx_full = branch_idx[tr_idx]
            val_idx_full = branch_idx[val_idx]

            X_tr, y_tr = X_train[tr_idx_full], branch_y[tr_idx]
            X_val, y_val = X_train[val_idx_full], branch_y[val_idx]

            X_tr_os, y_tr_os = oversample(X_tr, y_tr)
            model = train_lgb(X_tr_os, y_tr_os, X_val, y_val)

            val_preds = model.predict(X_val)
            val_pred_labels = val_preds.argmax(axis=1)
            sub_oof[val_idx_full] = cat_le.transform(branch_le.inverse_transform(val_pred_labels))

            if len(branch_test_idx) > 0:
                X_test_branch = X_test[branch_test_idx]
                test_preds = model.predict(X_test_branch)
                global_test_preds = np.zeros((len(branch_test_idx), len(cat_le.classes_)))
                for local_i, cls_name in enumerate(branch_le.classes_):
                    global_i = cat_le.transform([cls_name])[0]
                    global_test_preds[:, global_i] = test_preds[:, local_i]
                sub_test_preds[branch_test_idx] += global_test_preds / 4

    top3_cat_idxs = np.argsort(-sub_test_preds, axis=1)[:, :topk]
    return top3_cat_idxs

# --- Misconception Prediction ---
def misconception_predict(train_df, test_df, X_train, X_test, topk=3):
    skf = StratifiedKFold(n_splits=4, shuffle=True, random_state=42)
    mis_test_preds = np.zeros((len(test_df), len(mis_le.classes_)))

    for tr_idx, val_idx in skf.split(train_df, train_df['Misconception_enc']):
        X_tr, X_val = X_train[tr_idx], X_train[val_idx]
        y_tr, y_val = train_df['Misconception_enc'].values[tr_idx], train_df['Misconception_enc'].values[val_idx]

        X_tr_os, y_tr_os = oversample(X_tr, y_tr)
        model = train_lgb(X_tr_os, y_tr_os, X_val, y_val)

        preds = model.predict(X_test)
        mis_test_preds += preds / skf.n_splits

    top3_mis_idxs = np.argsort(-mis_test_preds, axis=1)[:, :topk]
    return top3_mis_idxs

# --- Run Predictions ---
topk = 3
category_top3 = hierarchical_predict(df, test_df, X_train_full, X_test_full, topk=topk)
misconception_top3 = misconception_predict(df, test_df, X_train_full, X_test_full, topk=topk)

# --- Format Submission ---
final_preds = []
for cat_idxs, mis_idxs in zip(category_top3, misconception_top3):
    preds = []
    for c_idx, m_idx in zip(cat_idxs, mis_idxs):
        cat_name = cat_le.inverse_transform([c_idx])[0]
        mis_name = mis_le.inverse_transform([m_idx])[0]
        preds.append(f"{cat_name}:{mis_name}")
    final_preds.append(" ".join(preds))

submission = pd.DataFrame({
    'row_id': test_df['row_id'],
    'Category:Misconception': final_preds
})

submission.to_csv(OUTPUT_PATH, index=False)
print(f"Saved submission to {OUTPUT_PATH}")
