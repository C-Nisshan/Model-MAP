import pandas as pd
import numpy as np
import re
import autocorrect
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

# ---------------------------
# Preprocessing
# ---------------------------
def preprocess_text(text):
    text = str(text).lower()
    text = re.sub(r'[^\w\s]', '', text)
    speller = autocorrect.Speller()
    return ' '.join(speller(word) for word in text.split())

# ---------------------------
# Load & preprocess train
# ---------------------------
df = pd.read_csv('data/train.csv', keep_default_na=False)
df['StudentExplanation'] = df['StudentExplanation'].apply(preprocess_text)
df['word_count'] = df['StudentExplanation'].apply(lambda x: len(str(x).split()))
df['Topic'] = df['QuestionText'].apply(
    lambda x: 'Fractions' if 'fraction' in str(x).lower()
    else 'Probability' if 'probability' in str(x).lower()
    else 'Other'
)

# TF-IDF
vectorizer = TfidfVectorizer(max_features=2000, stop_words='english')  # more features
X_text = vectorizer.fit_transform(df['StudentExplanation']).toarray()

# Extra features
X_extra = df[['word_count', 'Topic']].copy()
X_extra['Topic'] = X_extra['Topic'].map({'Fractions': 0, 'Probability': 1, 'Other': 2})
X = np.hstack([X_text, X_extra])

# Targets
y_category = df['Category']
y_misconception = df['Misconception']

# Train-test split
X_train, X_val, y_cat_train, y_cat_val, y_mis_train, y_mis_val = train_test_split(
    X, y_category, y_misconception, test_size=0.2, random_state=42
)

# ---------------------------
# Optimized RF models
# ---------------------------
rf_params = {
    "n_estimators": 400,
    "max_depth": None,           # grow trees fully
    "max_features": "sqrt",      # better generalization
    "min_samples_split": 5,
    "min_samples_leaf": 2,
    "bootstrap": True,
    "n_jobs": -1,
    "random_state": 42,
    "class_weight": "balanced"
}

model_cat = RandomForestClassifier(**rf_params)
model_cat.fit(X_train, y_cat_train)

model_mis = RandomForestClassifier(**rf_params)
model_mis.fit(X_train, y_mis_train)

# ---------------------------
# Evaluate (local check)
# ---------------------------
cat_pred = model_cat.predict(X_val)
mis_pred = model_mis.predict(X_val)

cat_acc = accuracy_score(y_cat_val, cat_pred)
cat_f1 = f1_score(y_cat_val, cat_pred, average='weighted')
mis_acc = accuracy_score(y_mis_val, mis_pred)
mis_f1 = f1_score(y_mis_val, mis_pred, average='weighted')

print(f"Category - Accuracy: {cat_acc:.4f}, F1: {cat_f1:.4f}")
print(f"Misconception - Accuracy: {mis_acc:.4f}, F1: {mis_f1:.4f}")

# ---------------------------
# Prepare test data
# ---------------------------
test_df = pd.read_csv('data/test.csv', keep_default_na=False)
test_df['StudentExplanation'] = test_df['StudentExplanation'].apply(preprocess_text)
test_df['word_count'] = test_df['StudentExplanation'].apply(lambda x: len(str(x).split()))
test_df['Topic'] = test_df['QuestionText'].apply(
    lambda x: 'Fractions' if 'fraction' in str(x).lower()
    else 'Probability' if 'probability' in str(x).lower()
    else 'Other'
)

X_test_text = vectorizer.transform(test_df['StudentExplanation']).toarray()
X_test_extra = test_df[['word_count', 'Topic']].copy()
X_test_extra['Topic'] = X_test_extra['Topic'].map({'Fractions': 0, 'Probability': 1, 'Other': 2})
X_test = np.hstack([X_test_text, X_test_extra])

# ---------------------------
# Get probabilities
# ---------------------------
cat_proba = model_cat.predict_proba(X_test)
mis_proba = model_mis.predict_proba(X_test)

cat_labels = model_cat.classes_
mis_labels = model_mis.classes_

# ---------------------------
# Build ranked top-25 predictions
# ---------------------------
final_predictions = []

for i in range(len(test_df)):
    ranked_pairs = []
    for cat_idx, cat_label in enumerate(cat_labels):
        for mis_idx, mis_label in enumerate(mis_labels):
            score = cat_proba[i][cat_idx] * mis_proba[i][mis_idx]
            ranked_pairs.append((f"{cat_label}:{mis_label}", score))

    ranked_pairs.sort(key=lambda x: x[1], reverse=True)
    top_25 = [pair[0] for pair in ranked_pairs[:25]]
    final_predictions.append(" ".join(top_25))

# ---------------------------
# Save Kaggle submission
# ---------------------------
submission = pd.DataFrame({
    'row_id': test_df['row_id'],
    'Category:Misconception': final_predictions
})
submission.to_csv('submission.csv', index=False)

print("Here 'submission.csv' saved")
