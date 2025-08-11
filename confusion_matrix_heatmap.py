import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
import re
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
import lightgbm as lgb
from sklearn.utils import resample

# Assuming the same imports and preprocessing as in your original script
# For brevity, only the necessary parts are included here

# Paths and configurations (from your script)
TRAIN_PATH = 'data/train.csv'
MODELS_DIR = 'models'
os.makedirs(MODELS_DIR, exist_ok=True)

# Load data
df = pd.read_csv(TRAIN_PATH, keep_default_na=False)

# Preprocessing (same as your script)
def preprocess_text(text):
    text = str(text).lower()
    text = re.sub(r'[^\w\s]', '', text)
    return text

df['StudentExplanation'] = df['StudentExplanation'].apply(preprocess_text)
df['QuestionText'] = df['QuestionText'].apply(preprocess_text)
df['word_count'] = df['StudentExplanation'].apply(lambda x: len(str(x).split())).astype(np.int32)
df['short_flag'] = (df['word_count'] < 5).astype(np.int8)
df['contains_answer'] = df.apply(
    lambda r: str(r['MC_Answer']).lower() in r['StudentExplanation'],
    axis=1
).astype(np.int8)

# Encode category
cat_le = LabelEncoder()
df['Category_enc'] = cat_le.fit_transform(df['Category'])
df['is_true'] = df['Category'].apply(lambda x: 1 if x.startswith("True") else 0)

# Vectorizers (fit globally as in your script)
word_vect = TfidfVectorizer(max_features=8000, ngram_range=(1, 2), min_df=2)
char_vect = TfidfVectorizer(max_features=5000, analyzer='char_wb', ngram_range=(3, 5))
ques_vect = TfidfVectorizer(max_features=2000, ngram_range=(1, 2), min_df=1)

X_word = word_vect.fit_transform(df['StudentExplanation'])
X_char = char_vect.fit_transform(df['StudentExplanation'])
X_ques = ques_vect.fit_transform(df['QuestionText'])
X_meta = sparse.csr_matrix(df[['word_count', 'short_flag', 'contains_answer']].values.astype(np.float32))
X_all = sparse.hstack([X_word, X_char, X_ques, X_meta]).tocsr()

# Modified hierarchical_predict to collect OOF predictions
def hierarchical_predict_with_oof(train_df, X_all, topk=3):
    skf = StratifiedKFold(n_splits=4, shuffle=True, random_state=42)
    oof_preds = np.zeros((len(train_df), len(cat_le.classes_)))
    y_true = train_df['Category_enc'].values

    for fold, (tr_idx, val_idx) in enumerate(skf.split(train_df, train_df['is_true'])):
        tr, val = train_df.iloc[tr_idx], train_df.iloc[val_idx]
        X_tr = X_all[tr_idx]
        X_val = X_all[val_idx]

        # Train top-level model (is_true)
        params = {
            "objective": "multiclass",
            "metric": "multi_logloss",
            "num_class": 2,
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
        train_set = lgb.Dataset(X_tr, label=tr['is_true'].values)
        val_set = lgb.Dataset(X_val, label=val['is_true'].values)
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

        # Predict top-level (is_true)
        val_pred_top = model.predict(X_val)
        true_pred = val_pred_top.argmax(axis=1)

        # Store top-level OOF predictions
        for i, idx in enumerate(val_idx):
            oof_preds[idx, 0:2] = val_pred_top[i]  # Store probabilities

        # Branch-level predictions
        for branch in [0, 1]:
            branch_idx = train_df.index[train_df['is_true'] == branch]
            branch_df = train_df.loc[branch_idx]
            branch_le = LabelEncoder()
            branch_y = branch_le.fit_transform(branch_df['Category'])

            for fold_branch, (tr_idx_b, val_idx_b) in enumerate(
                StratifiedKFold(n_splits=4, shuffle=True, random_state=42).split(branch_df, branch_y)
            ):
                tr_b, val_b = branch_df.iloc[tr_idx_b], branch_df.iloc[val_idx_b]
                X_tr_b = X_all[branch_idx][tr_idx_b]
                X_val_b = X_all[branch_idx][val_idx_b]

                # Oversample
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

                X_tr_os, y_tr_os = oversample(X_tr_b, branch_y[tr_idx_b])
                model_b = lgb.train(
                    {**params, "num_class": len(np.unique(branch_y))},
                    lgb.Dataset(X_tr_os, label=y_tr_os),
                    num_boost_round=1500,
                    valid_sets=[lgb.Dataset(X_val_b, label=branch_y[val_idx_b])],
                    callbacks=[
                        lgb.early_stopping(stopping_rounds=50),
                        lgb.log_evaluation(period=50)
                    ]
                )

                # Predict branch-level and map to global category indices
                val_preds = model_b.predict(X_val_b)
                for i, idx in enumerate(val_b.index):
                    global_idx = cat_le.transform([branch_le.inverse_transform([val_preds[i].argmax()])[0]])[0]
                    oof_preds[idx, global_idx] = 1  # Store as one-hot for simplicity

    return oof_preds, y_true

# Generate OOF predictions
oof_preds, y_true = hierarchical_predict_with_oof(df, X_all)

# Compute confusion matrix
cm = confusion_matrix(y_true, oof_preds.argmax(axis=1), normalize='true')

# Create heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(
    cm * 100,  # Convert to percentages
    annot=True, fmt='.1f', cmap='Blues',  # Show percentages with one decimal
    xticklabels=cat_le.classes_, yticklabels=cat_le.classes_,
    cbar_kws={'label': 'Percentage (%)'},
)
plt.title('Normalized Confusion Matrix for Category Predictions (Percentages)')
plt.xlabel('Predicted Category')
plt.ylabel('True Category')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()

# Save the figure
plt.savefig('figure_4_confusion_matrix.png', dpi=300, bbox_inches='tight')
plt.close()

print("Figure 4 saved as 'figure_4_confusion_matrix.png'")