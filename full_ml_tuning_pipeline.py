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
import optuna

# =========================
# Paths (adjust if needed)
# =========================
TRAIN_PATH = '/kaggle/input/map-charting-student-math-misunderstandings/train.csv'
TEST_PATH  = '/kaggle/input/map-charting-student-math-misunderstandings/test.csv'
OUTPUT_PATH = '/kaggle/working/submission.csv'

# =========================
# GPU detection
# =========================
def has_gpu():
    try:
        res = subprocess.run(['nvidia-smi'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return res.returncode == 0
    except Exception:
        return False
USE_GPU = has_gpu()
print("GPU available:", USE_GPU)

# =========================
# Text preprocessing
# =========================
def preprocess_text(text):
    text = str(text).lower()
    text = re.sub(r'[^\w\s]', '', text)
    return text

# =========================
# Load data
# =========================
df = pd.read_csv(TRAIN_PATH, keep_default_na=False)
test_df = pd.read_csv(TEST_PATH, keep_default_na=False)

# =========================
# Feature engineering
# =========================
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

# =========================
# Vectorizer builder
# =========================
def build_vectorizers(train_texts, train_qtexts):
    word_vect = TfidfVectorizer(max_features=12000, ngram_range=(1, 2), min_df=2)
    char_vect = TfidfVectorizer(max_features=8000, analyzer='char_wb', ngram_range=(3, 5))
    ques_vect = TfidfVectorizer(max_features=3000, ngram_range=(1, 2), min_df=1)

    X_word = word_vect.fit_transform(train_texts)
    X_char = char_vect.fit_transform(train_texts)
    X_ques = ques_vect.fit_transform(train_qtexts)

    return word_vect, char_vect, ques_vect, X_word, X_char, X_ques

# =========================
# Oversampling helper
# =========================
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

# =========================
# Optuna tuning
# =========================
def objective(trial, X, y, num_class):
    params = {
        "objective": "multiclass",
        "metric": "multi_logloss",
        "num_class": num_class,
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2),
        "num_leaves": trial.suggest_int("num_leaves", 15, 255),
        "max_depth": trial.suggest_int("max_depth", -1, 20),
        "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 10, 200),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.6, 1.0),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.6, 1.0),
        "bagging_freq": trial.suggest_int("bagging_freq", 1, 10),
        "lambda_l1": trial.suggest_float("lambda_l1", 0.0, 5.0),
        "lambda_l2": trial.suggest_float("lambda_l2", 0.0, 5.0),
        "verbosity": -1,
        "seed": 42,
        "class_weight": "balanced"
    }
    if USE_GPU:
        params.update({"device": "gpu", "gpu_platform_id": 0, "gpu_device_id": 0})

    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    scores = []
    for tr_idx, val_idx in skf.split(X, y):
        X_tr, X_val = X[tr_idx], X[val_idx]
        y_tr, y_val = y[tr_idx], y[val_idx]
        lgb_train = lgb.Dataset(X_tr, label=y_tr)
        lgb_val = lgb.Dataset(X_val, label=y_val)
        model = lgb.train(
            params,
            lgb_train,
            num_boost_round=500,
            valid_sets=[lgb_val],
            early_stopping_rounds=30,
            verbose_eval=False
        )
        preds = model.predict(X_val)
        logloss = -np.mean([np.log(p[y_true] + 1e-15) for p, y_true in zip(preds, y_val)])
        scores.append(logloss)
    return np.mean(scores)

# =========================
# Train LightGBM final
# =========================
def train_lgb(X, y, X_val, y_val, params):
    params = params.copy()
    params["num_class"] = len(np.unique(y))
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
            lgb.log_evaluation(period=0)
        ]
    )
    return model

# =========================
# Hierarchical + tuning
# =========================
def hierarchical_predict_with_tuning(train_df, test_df, topk=3):
    skf = StratifiedKFold(n_splits=4, shuffle=True, random_state=42)

    # Step 1: Predict is_true
    true_oof = np.zeros(len(train_df), dtype=int)
    true_test_preds = np.zeros((len(test_df), 2))

    # Prepare features
    wv, cv, qv, Xw_full, Xc_full, Xq_full = build_vectorizers(train_df['StudentExplanation'], train_df['QuestionText'])
    X_full = sparse.hstack([
        Xw_full, Xc_full, Xq_full,
        sparse.csr_matrix(train_df[['word_count', 'short_flag', 'contains_answer']].values.astype(np.float32))
    ])
    X_full = X_full.tocsr()

    # Tuning for step 1
    print("Tuning Step 1 (is_true)...")
    study1 = optuna.create_study(direction="minimize")
    study1.optimize(lambda trial: objective(trial, X_full, train_df['is_true'].values, 2), n_trials=30)
    best_params_step1 = study1.best_params
    best_params_step1.update({"objective": "multiclass", "metric": "multi_logloss", "seed": 42, "class_weight": "balanced"})

    # CV training
    for tr_idx, val_idx in skf.split(train_df, train_df['is_true']):
        X_tr, X_val = X_full[tr_idx], X_full[val_idx]
        y_tr, y_val = train_df['is_true'].values[tr_idx], train_df['is_true'].values[val_idx]
        model = train_lgb(X_tr, y_tr, X_val, y_val, best_params_step1)
        true_oof[val_idx] = model.predict(X_val).argmax(axis=1)
        X_test = sparse.hstack([
            wv.transform(test_df['StudentExplanation']),
            cv.transform(test_df['StudentExplanation']),
            qv.transform(test_df['QuestionText']),
            sparse.csr_matrix(test_df[['word_count', 'short_flag', 'contains_answer']].values.astype(np.float32))
        ]).tocsr()
        true_test_preds += model.predict(X_test)

    true_test_preds /= skf.n_splits
    true_test = true_test_preds.argmax(axis=1)

    # Step 2: Predict Category subclasses per branch
    sub_test_preds = np.zeros((len(test_df), len(cat_le.classes_)))
    for branch in [0, 1]:
        branch_idx = train_df.index[train_df['is_true'] == branch]
        branch_df = train_df.loc[branch_idx]
        if len(branch_df) == 0:
            continue
        branch_le = LabelEncoder()
        branch_y = branch_le.fit_transform(branch_df['Category'].values)

        # Branch tuning
        print(f"Tuning Step 2 (branch {branch})...")
        wv_b, cv_b, qv_b, Xw_b, Xc_b, Xq_b = build_vectorizers(branch_df['StudentExplanation'], branch_df['QuestionText'])
        X_branch = sparse.hstack([
            Xw_b, Xc_b, Xq_b,
            sparse.csr_matrix(branch_df[['word_count', 'short_flag', 'contains_answer']].values.astype(np.float32))
        ]).tocsr()

        study_b = optuna.create_study(direction="minimize")
        study_b.optimize(lambda trial: objective(trial, X_branch, branch_y, len(branch_le.classes_)), n_trials=30)
        best_params_branch = study_b.best_params
        best_params_branch.update({"objective": "multiclass", "metric": "multi_logloss", "seed": 42, "class_weight": "balanced"})

        # Train & predict
        branch_test_idx = test_df.index[true_test == branch]
        branch_test_df = test_df.loc[branch_test_idx]
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
            X_test_b = sparse.hstack([
                wv.transform(branch_test_df['StudentExplanation']),
                cv.transform(branch_test_df['StudentExplanation']),
                qv.transform(branch_test_df['QuestionText']),
                sparse.csr_matrix(branch_test_df[['word_count', 'short_flag', 'contains_answer']].values.astype(np.float32))
            ])
            X_tr_os, y_tr_os = oversample(X_tr, branch_y[tr_idx])
            model = train_lgb(X_tr_os, y_tr_os, X_val, branch_y[val_idx], best_params_branch)

            if len(branch_test_df) > 0:
                test_preds = model.predict(X_test_b)
                global_test_preds = np.zeros((len(branch_test_df), len(cat_le.classes_)))
                for local_i, cls_name in enumerate(branch_le.classes_):
                    global_i = cat_le.transform([cls_name])[0]
                    global_test_preds[:, global_i] = test_preds[:, local_i]
                sub_test_preds[branch_test_idx] += global_test_preds / 4

    top3_cat_idxs = np.argsort(-sub_test_preds, axis=1)[:, :topk]
    return top3_cat_idxs

# =========================
# Misconception prediction
# =========================
def misconception_predict_with_tuning(train_df, test_df, topk=3):
    skf = StratifiedKFold(n_splits=4, shuffle=True, random_state=42)
    wv, cv, qv, Xw_full, Xc_full, Xq_full = build_vectorizers(train_df['StudentExplanation'], train_df['QuestionText'])
    X_full = sparse.hstack([
        Xw_full, Xc_full, Xq_full,
        sparse.csr_matrix(train_df[['word_count', 'short_flag', 'contains_answer']].values.astype(np.float32))
    ]).tocsr()

    print("Tuning Misconception model...")
    study = optuna.create_study(direction="minimize")
    study.optimize(lambda trial: objective(trial, X_full, train_df['Misconception_enc'].values, len(mis_le.classes_)), n_trials=30)
    best_params_mis = study.best_params
    best_params_mis.update({"objective": "multiclass", "metric": "multi_logloss", "seed": 42, "class_weight": "balanced"})

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
        model = train_lgb(X_tr_os, y_tr_os, X_val, val['Misconception_enc'].values, best_params_mis)
        mis_test_preds += model.predict(X_test) / skf.n_splits

    top3_mis_idxs = np.argsort(-mis_test_preds, axis=1)[:, :topk]
    return top3_mis_idxs

# =========================
# Run tuned pipeline
# =========================
topk = 3
category_top3 = hierarchical_predict_with_tuning(df, test_df, topk=topk)
misconception_top3 = misconception_predict_with_tuning(df, test_df, topk=topk)

# Format submission
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
