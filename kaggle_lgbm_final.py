import os
import re
import subprocess
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import resample
import lightgbm as lgb

# =========================================================
# CONFIG
# =========================================================
TRAIN_PATH = '/kaggle/input/map-charting-student-math-misunderstandings/train.csv'
TEST_PATH  = '/kaggle/input/map-charting-student-math-misunderstandings/test.csv'
OUTPUT_PATH = '/kaggle/working/submission.csv'
topk = 3  # top K predictions for MAP@3

# =========================================================
# GPU Detection
# =========================================================
def has_gpu():
    try:
        res = subprocess.run(['nvidia-smi'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return res.returncode == 0
    except Exception:
        return False

USE_GPU = False  # Force CPU mode
print("GPU available:", USE_GPU)

# =========================================================
# MAP@K metric
# =========================================================
def mapk(actual, predicted, k=3):
    def apk(a, p, k):
        if len(p) > k:
            p = p[:k]
        score = 0.0
        num_hits = 0.0
        for i, pred in enumerate(p):
            if pred in a and pred not in p[:i]:
                num_hits += 1.0
                score += num_hits / (i + 1.0)
        return score / min(len(a), k)
    return np.mean([apk(a, p, k) for a, p in zip(actual, predicted)])

# =========================================================
# Preprocessing
# =========================================================
def preprocess_text(text):
    text = str(text).lower()
    text = re.sub(r'[^\w\s]', '', text)
    return text

# Load data
df = pd.read_csv(TRAIN_PATH, keep_default_na=False)
test_df = pd.read_csv(TEST_PATH, keep_default_na=False)

# Feature engineering
for d in (df, test_df):
    d['StudentExplanation'] = d['StudentExplanation'].apply(preprocess_text)
    d['QuestionText'] = d['QuestionText'].apply(preprocess_text)
    d['word_count'] = d['StudentExplanation'].apply(lambda x: len(str(x).split())).astype(np.int32)
    d['short_flag'] = (d['word_count'] < 5).astype(np.int8)
    d['contains_answer'] = d.apply(
        lambda r: str(r['MC_Answer']).lower() in r['StudentExplanation'],
        axis=1
    ).astype(np.int8)

# Encode category + binary top level
cat_le = LabelEncoder()
df['Category_enc'] = cat_le.fit_transform(df['Category'])
df['is_true'] = df['Category'].apply(lambda x: 1 if x.startswith("True") else 0)

# Misconception encoding
df['Misconception_full'] = df['Misconception'].replace('', 'None')
mis_le = LabelEncoder()
df['Misconception_enc'] = mis_le.fit_transform(df['Misconception_full'])

# =========================================================
# Vectorizers - Fit ONCE globally and reuse
# =========================================================
print("Fitting vectorizers globally...")
word_vect = TfidfVectorizer(max_features=8000, ngram_range=(1, 2), min_df=2)
char_vect = TfidfVectorizer(max_features=5000, analyzer='char_wb', ngram_range=(3, 5))
ques_vect = TfidfVectorizer(max_features=2000, ngram_range=(1, 2), min_df=1)

X_word = word_vect.fit_transform(df['StudentExplanation'])
X_char = char_vect.fit_transform(df['StudentExplanation'])
X_ques = ques_vect.fit_transform(df['QuestionText'])

X_test_word = word_vect.transform(test_df['StudentExplanation'])
X_test_char = char_vect.transform(test_df['StudentExplanation'])
X_test_ques = ques_vect.transform(test_df['QuestionText'])

X_meta = sparse.csr_matrix(df[['word_count', 'short_flag', 'contains_answer']].values.astype(np.float32))
X_test_meta = sparse.csr_matrix(test_df[['word_count', 'short_flag', 'contains_answer']].values.astype(np.float32))

X_all = sparse.hstack([X_word, X_char, X_ques, X_meta]).tocsr()
X_test_all = sparse.hstack([X_test_word, X_test_char, X_test_ques, X_test_meta]).tocsr()

# =========================================================
# LightGBM training
# =========================================================
def train_lgb(X, y, X_val, y_val, num_class):
    params = {
        "objective": "multiclass",
        "metric": "multi_logloss",
        "num_class": num_class,
        "learning_rate": 0.05,
        "num_leaves": 63,
        "max_depth": -1,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 2,
        "min_data_in_leaf": 20,
        "verbosity": -1,
        "num_threads": 2,
        "seed": 42,
        "class_weight": "balanced",
    }
    if USE_GPU:
        params.update({"device": "gpu", "gpu_platform_id": 0, "gpu_device_id": 0})

    train_set = lgb.Dataset(X, label=y)
    val_set = lgb.Dataset(X_val, label=y_val)

    model = lgb.train(
        params,
        train_set,
        num_boost_round=1500,
        valid_sets=[train_set, val_set],
        callbacks=[
            lgb.early_stopping(stopping_rounds=50),
            lgb.log_evaluation(period=50)
        ]
    )
    return model

# =========================================================
# Oversample helper
# =========================================================
def oversample(X, y, min_samples=200):
    y = np.array(y)
    idxs = []
    for cls in np.unique(y):
        cls_idx = np.where(y == cls)[0]
        if len(cls_idx) < min_samples:
            cls_idx = resample(cls_idx, replace=True, n_samples=min_samples, random_state=42)
        idxs.extend(cls_idx)
    idxs = np.array(idxs)
    return X[idxs], y[idxs]

# =========================================================
# Hierarchical prediction
# =========================================================
def hierarchical_predict(train_df, test_df, topk=3):
    skf = StratifiedKFold(n_splits=4, shuffle=True, random_state=42)
    true_test_preds = np.zeros((len(test_df), 2))
    oof_preds = np.zeros((len(train_df), len(cat_le.classes_)))

    for tr_idx, val_idx in skf.split(train_df, train_df['is_true']):
        tr, val = train_df.iloc[tr_idx], train_df.iloc[val_idx]
        X_tr = X_all[tr_idx]
        X_val = X_all[val_idx]
        X_test = X_test_all

        model = train_lgb(X_tr, tr['is_true'].values, X_val, val['is_true'].values, num_class=2)
        true_test_preds += model.predict(X_test)
        val_pred_top = model.predict(X_val)

        for i, idx in enumerate(val_idx):
            oof_preds[idx, 0:2] = val_pred_top[i]  # only binary here

    true_test_preds /= skf.n_splits
    true_test = true_test_preds.argmax(axis=1)

    sub_test_preds = np.zeros((len(test_df), len(cat_le.classes_)))
    for branch in [0, 1]:
        branch_idx = train_df.index[train_df['is_true'] == branch]
        branch_df = train_df.loc[branch_idx]
        branch_le = LabelEncoder()
        branch_y = branch_le.fit_transform(branch_df['Category'])

        branch_test_idx = test_df.index[true_test == branch]
        for tr_idx, val_idx in StratifiedKFold(n_splits=4, shuffle=True, random_state=42).split(branch_df, branch_y):
            tr, val = branch_df.iloc[tr_idx], branch_df.iloc[val_idx]
            X_tr = X_all[branch_idx][tr_idx]
            X_val = X_all[branch_idx][val_idx]
            X_test = X_test_all[branch_test_idx]

            X_tr_os, y_tr_os = oversample(X_tr, branch_y[tr_idx])
            model = train_lgb(X_tr_os, y_tr_os, X_val, branch_y[val_idx], num_class=len(np.unique(branch_y)))

            val_preds = model.predict(X_val)
            for i, idx in enumerate(val.index):
                global_idx = cat_le.transform([branch_le.inverse_transform([val_preds[i].argmax()])[0]])[0]
                oof_preds[idx, global_idx] = 1

            if len(branch_test_idx) > 0:
                test_preds = model.predict(X_test)
                for local_i, cls_name in enumerate(branch_le.classes_):
                    global_i = cat_le.transform([cls_name])[0]
                    sub_test_preds[branch_test_idx, global_i] += test_preds[:, local_i] / 4

    top3_cat_idxs = np.argsort(-sub_test_preds, axis=1)[:, :topk]

    y_true = train_df['Category_enc'].values
    y_pred_top1 = oof_preds.argmax(axis=1)
    acc = (y_pred_top1 == y_true).mean()
    map3 = mapk([[y] for y in y_true], [list(np.argsort(-row)[:3]) for row in oof_preds], k=3)
    print(f"Category Accuracy: {acc:.4f} | Category MAP@3: {map3:.4f}")

    return top3_cat_idxs, oof_preds

# =========================================================
# Misconception prediction
# =========================================================
def misconception_predict(train_df, test_df, topk=3):
    skf = StratifiedKFold(n_splits=4, shuffle=True, random_state=42)
    mis_test_preds = np.zeros((len(test_df), len(mis_le.classes_)))
    oof_preds = np.zeros((len(train_df), len(mis_le.classes_)))

    for tr_idx, val_idx in skf.split(train_df, train_df['Misconception_enc']):
        tr, val = train_df.iloc[tr_idx], train_df.iloc[val_idx]
        X_tr = X_all[tr_idx]
        X_val = X_all[val_idx]
        X_test = X_test_all

        X_tr_os, y_tr_os = oversample(X_tr, tr['Misconception_enc'].values)
        model = train_lgb(X_tr_os, y_tr_os, X_val, val['Misconception_enc'].values, num_class=len(mis_le.classes_))
        mis_test_preds += model.predict(X_test) / skf.n_splits

        val_preds = model.predict(X_val)
        for i, idx in enumerate(val.index):
            oof_preds[idx] = val_preds[i]

    top3_mis_idxs = np.argsort(-mis_test_preds, axis=1)[:, :topk]

    y_true = train_df['Misconception_enc'].values
    y_pred_top1 = oof_preds.argmax(axis=1)
    acc = (y_pred_top1 == y_true).mean()
    map3 = mapk([[y] for y in y_true], [list(np.argsort(-row)[:3]) for row in oof_preds], k=3)
    print(f"Misconception Accuracy: {acc:.4f} | Misconception MAP@3: {map3:.4f}")

    return top3_mis_idxs, oof_preds

# =========================================================
# Combined MAP@3 + Accuracy
# =========================================================
def combined_map3(train_df, category_oof_probs, mis_oof_probs, topk=3):
    pair_labels = [f"{c}:{m}" for c, m in zip(train_df['Category'], train_df['Misconception_full'])]
    pair_le = LabelEncoder()
    y_true = pair_le.fit_transform(pair_labels)

    combined_probs = np.zeros((len(train_df), len(pair_le.classes_)))
    for i in range(len(train_df)):
        for ci in range(len(cat_le.classes_)):
            for mi in range(len(mis_le.classes_)):
                pair = f"{cat_le.inverse_transform([ci])[0]}:{mis_le.inverse_transform([mi])[0]}"
                if pair in pair_le.classes_:
                    pair_idx = pair_le.transform([pair])[0]
                    combined_probs[i, pair_idx] = category_oof_probs[i, ci] * mis_oof_probs[i, mi]

    topk_preds = np.argsort(-combined_probs, axis=1)[:, :topk]

    acc = (topk_preds[:, 0] == y_true).mean()
    map3_score = mapk([[y] for y in y_true], [list(row) for row in topk_preds], k=topk)
    print(f"OVERALL Accuracy: {acc:.4f} | OVERALL MAP@3: {map3_score:.4f}")
    return acc, map3_score

# =========================================================
# RUN PIPELINE
# =========================================================
category_top3, cat_oof_probs = hierarchical_predict(df, test_df, topk=topk)
misconception_top3, mis_oof_probs = misconception_predict(df, test_df, topk=topk)

combined_map3(df, cat_oof_probs, mis_oof_probs, topk=topk)

# Final submission formatting
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
