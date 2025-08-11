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

# Paths (adjust these if needed)
TRAIN_PATH = '/kaggle/input/map-charting-student-math-misunderstandings/train.csv'
TEST_PATH  = '/kaggle/input/map-charting-student-math-misunderstandings/test.csv'
OUTPUT_PATH = '/kaggle/working/submission.csv'

# GPU detection (optional)
def has_gpu():
    try:
        res = subprocess.run(['nvidia-smi'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return res.returncode == 0
    except Exception:
        return False
USE_GPU = has_gpu()
print("GPU available:", USE_GPU)

# Text preprocessing
def preprocess_text(text):
    text = str(text).lower()
    text = re.sub(r'[^\w\s]', '', text)
    return text

# Load data
df = pd.read_csv(TRAIN_PATH, keep_default_na=False)
test_df = pd.read_csv(TEST_PATH, keep_default_na=False)

# Feature engineering: clean text, word count, flags
for d in (df, test_df):
    d['StudentExplanation'] = d['StudentExplanation'].apply(preprocess_text)
    d['QuestionText'] = d['QuestionText'].apply(preprocess_text)
    d['word_count'] = d['StudentExplanation'].apply(lambda x: len(str(x).split())).astype(np.int32)
    d['short_flag'] = (d['word_count'] < 5).astype(np.int8)
    d['contains_answer'] = d.apply(
        lambda r: str(r['MC_Answer']).lower() in r['StudentExplanation'],
        axis=1
    ).astype(np.int8)

# Encode target category globally
cat_le = LabelEncoder()
df['Category_enc'] = cat_le.fit_transform(df['Category'])

# Binary True/False top-level label for hierarchy step 1
df['is_true'] = df['Category'].apply(lambda x: 1 if x.startswith("True") else 0)

# Misconception encoding (fill missing with 'None')
df['Misconception_full'] = df['Misconception'].replace('', 'None')
mis_le = LabelEncoder()
df['Misconception_enc'] = mis_le.fit_transform(df['Misconception_full'])

# TF-IDF vectorizers (reuse in folds)
def build_vectorizers(train_texts, train_qtexts):
    word_vect = TfidfVectorizer(max_features=12000, ngram_range=(1, 2), min_df=2)
    char_vect = TfidfVectorizer(max_features=8000, analyzer='char_wb', ngram_range=(3, 5))
    ques_vect = TfidfVectorizer(max_features=3000, ngram_range=(1, 2), min_df=1)

    X_word = word_vect.fit_transform(train_texts)
    X_char = char_vect.fit_transform(train_texts)
    X_ques = ques_vect.fit_transform(train_qtexts)

    return word_vect, char_vect, ques_vect, X_word, X_char, X_ques

# LightGBM training helper with class_weight to handle imbalance
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
        "class_weight": "balanced",  # Important for imbalance
    }
    if USE_GPU:
        params.update({"device": "gpu", "gpu_platform_id": 0, "gpu_device_id": 0})

    train_set = lgb.Dataset(X, label=y)
    val_set = lgb.Dataset(X_val, label=y_val)

    model = lgb.train(
        params,
        train_set,
        num_boost_round=1000,
        valid_sets=[train_set, val_set],
        callbacks=[
            lgb.early_stopping(stopping_rounds=50),
            lgb.log_evaluation(period=0)  # disable logs
        ]
    )
    return model

# Oversample minority classes to at least min_samples=200
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

# Predict top-level True/False (Step 1) and then Category subclasses (Step 2)
def hierarchical_predict(train_df, test_df, topk=3):
    skf = StratifiedKFold(n_splits=4, shuffle=True, random_state=42)

    # Step 1: Predict is_true (binary)
    true_oof = np.zeros(len(train_df), dtype=int)
    true_test_preds = np.zeros((len(test_df), 2))

    for tr_idx, val_idx in skf.split(train_df, train_df['is_true']):
        tr, val = train_df.iloc[tr_idx], train_df.iloc[val_idx]
        wv, cv, qv, Xw_tr, Xc_tr, Xq_tr = build_vectorizers(tr['StudentExplanation'], tr['QuestionText'])

        X_tr = sparse.hstack([
            Xw_tr, Xc_tr, Xq_tr,
            sparse.csr_matrix(tr[['word_count', 'short_flag', 'contains_answer']].values.astype(np.float32))
        ])
        X_val = sparse.hstack([
            wv.transform(val['StudentExplanation']),
            cv.transform(val['StudentExplanation']),
            qv.transform(val['QuestionText']),
            sparse.csr_matrix(val[['word_count', 'short_flag', 'contains_answer']].values.astype(np.float32))
        ])
        X_test = sparse.hstack([
            wv.transform(test_df['StudentExplanation']),
            cv.transform(test_df['StudentExplanation']),
            qv.transform(test_df['QuestionText']),
            sparse.csr_matrix(test_df[['word_count', 'short_flag', 'contains_answer']].values.astype(np.float32))
        ])

        model = train_lgb(X_tr, tr['is_true'].values, X_val, val['is_true'].values)
        true_oof[val_idx] = model.predict(X_val).argmax(axis=1)
        true_test_preds += model.predict(X_test)

    true_test_preds /= skf.n_splits
    true_test = true_test_preds.argmax(axis=1)

    # Step 2: Predict Category subclasses within each True/False branch
    sub_oof = np.zeros(len(train_df), dtype=int)
    sub_test_preds = np.zeros((len(test_df), len(cat_le.classes_)))

    for branch in [0, 1]:
        branch_idx = train_df.index[train_df['is_true'] == branch]
        branch_df = train_df.loc[branch_idx]
        if len(branch_df) == 0:
            continue

        # Local encoding for this branch's categories
        branch_cat_labels = branch_df['Category'].values
        branch_le = LabelEncoder()
        branch_y = branch_le.fit_transform(branch_cat_labels)

        # Indices in test where predicted top-level == branch
        branch_test_idx = test_df.index[true_test == branch]
        branch_test_df = test_df.loc[branch_test_idx]

        if len(np.unique(branch_y)) <= 1:
            # Only one category in branch, fill predictions directly
            if len(branch_test_df) > 0:
                # Map local to global index
                global_cat_idx = cat_le.transform(branch_le.classes_)[0]
                sub_test_preds[branch_test_idx, global_cat_idx] = 1.0
            continue

        for tr_idx, val_idx in StratifiedKFold(n_splits=4, shuffle=True, random_state=42).split(branch_df, branch_y):
            tr, val = branch_df.iloc[tr_idx], branch_df.iloc[val_idx]
            wv, cv, qv, Xw_tr, Xc_tr, Xq_tr = build_vectorizers(tr['StudentExplanation'], tr['QuestionText'])

            X_tr = sparse.hstack([
                Xw_tr, Xc_tr, Xq_tr,
                sparse.csr_matrix(tr[['word_count', 'short_flag', 'contains_answer']].values.astype(np.float32))
            ])
            X_val = sparse.hstack([
                wv.transform(val['StudentExplanation']),
                cv.transform(val['StudentExplanation']),
                qv.transform(val['QuestionText']),
                sparse.csr_matrix(val[['word_count', 'short_flag', 'contains_answer']].values.astype(np.float32))
            ])
            X_test = sparse.hstack([
                wv.transform(branch_test_df['StudentExplanation']),
                cv.transform(branch_test_df['StudentExplanation']),
                qv.transform(branch_test_df['QuestionText']),
                sparse.csr_matrix(branch_test_df[['word_count', 'short_flag', 'contains_answer']].values.astype(np.float32))
            ])

            # Oversample minority classes inside branch
            X_tr_os, y_tr_os = oversample(X_tr, branch_y[tr_idx])
            model = train_lgb(X_tr_os, y_tr_os, X_val, branch_y[val_idx])

            # Validation predictions
            val_preds = model.predict(X_val)
            val_pred_labels = val_preds.argmax(axis=1)
            # Map back local labels to global category indexes
            global_val_pred_labels = cat_le.transform(branch_le.inverse_transform(val_pred_labels))
            sub_oof[val.index] = global_val_pred_labels

            # Test predictions (average over folds)
            if len(branch_test_df) > 0:
                test_preds = model.predict(X_test)
                # Map local class probabilities to global space
                global_test_preds = np.zeros((len(branch_test_df), len(cat_le.classes_)))
                for local_i, cls_name in enumerate(branch_le.classes_):
                    global_i = cat_le.transform([cls_name])[0]
                    global_test_preds[:, global_i] = test_preds[:, local_i]
                sub_test_preds[branch_test_idx] += global_test_preds / 4  # 4 folds

    # For any train samples not covered (e.g. edge cases), fill with most common category
    sub_oof[sub_oof == 0] = cat_le.transform(['True_Correct'])[0]

    # Get top 3 predicted categories per test row by probability
    top3_cat_idxs = np.argsort(-sub_test_preds, axis=1)[:, :topk]

    return top3_cat_idxs

# Misconception prediction with top 3 outputs
def misconception_predict(train_df, test_df, topk=3):
    skf = StratifiedKFold(n_splits=4, shuffle=True, random_state=42)
    mis_test_preds = np.zeros((len(test_df), len(mis_le.classes_)))

    for tr_idx, val_idx in skf.split(train_df, train_df['Misconception_enc']):
        tr, val = train_df.iloc[tr_idx], train_df.iloc[val_idx]
        wv, cv, qv, Xw_tr, Xc_tr, Xq_tr = build_vectorizers(tr['StudentExplanation'], tr['QuestionText'])

        X_tr = sparse.hstack([
            Xw_tr, Xc_tr, Xq_tr,
            sparse.csr_matrix(tr[['word_count', 'short_flag', 'contains_answer']].values.astype(np.float32))
        ])
        X_val = sparse.hstack([
            wv.transform(val['StudentExplanation']),
            cv.transform(val['StudentExplanation']),
            qv.transform(val['QuestionText']),
            sparse.csr_matrix(val[['word_count', 'short_flag', 'contains_answer']].values.astype(np.float32))
        ])
        X_test = sparse.hstack([
            wv.transform(test_df['StudentExplanation']),
            cv.transform(test_df['StudentExplanation']),
            qv.transform(test_df['QuestionText']),
            sparse.csr_matrix(test_df[['word_count', 'short_flag', 'contains_answer']].values.astype(np.float32))
        ])

        X_tr_os, y_tr_os = oversample(X_tr, tr['Misconception_enc'].values)
        model = train_lgb(X_tr_os, y_tr_os, X_val, val['Misconception_enc'].values)

        preds = model.predict(X_test)
        mis_test_preds += preds / skf.n_splits

    # top 3 misconception indexes per sample
    top3_mis_idxs = np.argsort(-mis_test_preds, axis=1)[:, :topk]

    return top3_mis_idxs

# --- Run hierarchical category and misconception prediction ---
topk = 3
category_top3 = hierarchical_predict(df, test_df, topk=topk)
misconception_top3 = misconception_predict(df, test_df, topk=topk)

# Format submission with top 3 predictions space separated (Category:Misconception)
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
