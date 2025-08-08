import pandas as pd
import numpy as np
import re

# If autocorrect missing, fallback to simple cleaning.
try:
    from autocorrect import Speller
    speller = Speller()
    def preprocess_text(text):
        text = str(text).lower()
        # Remove punctuation
        text = re.sub(r'[^\w\s]', '', text)
        # autocorrect each word
        return ' '.join(speller(word) for word in text.split())
except ImportError:
    # fallback if autocorrect not installed
    def preprocess_text(text):
        text = str(text).lower()
        # Remove punctuation only
        text = re.sub(r'[^\w\s]', '', text)
        return text

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

# File paths for Kaggle competition dataset
TRAIN_PATH = '/kaggle/input/map-charting-student-math-misunderstandings/train.csv'
TEST_PATH  = '/kaggle/input/map-charting-student-math-misunderstandings/test.csv'
OUTPUT_PATH = '/kaggle/working/submission.csv'

# ---------------------------
# Load & preprocess train
# ---------------------------
df = pd.read_csv(TRAIN_PATH, keep_default_na=False)
df['StudentExplanation'] = df['StudentExplanation'].apply(preprocess_text)
df['word_count'] = df['StudentExplanation'].apply(lambda x: len(str(x).split()))

# Simple heuristic Topic feature
def topic_from_text(text):
    text = str(text).lower()
    if 'fraction' in text:
        return 'Fractions'
    elif 'probability' in text:
        return 'Probability'
    else:
        return 'Other'

df['Topic'] = df['QuestionText'].apply(topic_from_text)

# TF-IDF vectorizer (max 2000 features, English stopwords)
vectorizer = TfidfVectorizer(max_features=2000, stop_words='english')
X_text = vectorizer.fit_transform(df['StudentExplanation']).toarray()

# Encode Topic to int
topic_map = {'Fractions': 0, 'Probability': 1, 'Other': 2}
X_extra = df[['word_count', 'Topic']].copy()
X_extra['Topic'] = X_extra['Topic'].map(topic_map)

# Combine TF-IDF features + extra features
X = np.hstack([X_text, X_extra])

# Targets
y_category = df['Category']
y_misconception = df['Misconception']

# Train-test split (20% val)
X_train, X_val, y_cat_train, y_cat_val, y_mis_train, y_mis_val = train_test_split(
    X, y_category, y_misconception, test_size=0.2, random_state=42
)

# ---------------------------
# Random Forest parameters
# ---------------------------
rf_params = {
    "n_estimators": 400,
    "max_depth": None,
    "max_features": "sqrt",
    "min_samples_split": 5,
    "min_samples_leaf": 2,
    "bootstrap": True,
    "n_jobs": -1,
    "random_state": 42,
    "class_weight": "balanced"
}

# Train models
model_cat = RandomForestClassifier(**rf_params)
model_cat.fit(X_train, y_cat_train)

model_mis = RandomForestClassifier(**rf_params)
model_mis.fit(X_train, y_mis_train)

# ---------------------------
# Validation evaluation (optional, print to console)
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
test_df = pd.read_csv(TEST_PATH, keep_default_na=False)
test_df['StudentExplanation'] = test_df['StudentExplanation'].apply(preprocess_text)
test_df['word_count'] = test_df['StudentExplanation'].apply(lambda x: len(str(x).split()))
test_df['Topic'] = test_df['QuestionText'].apply(topic_from_text)

X_test_text = vectorizer.transform(test_df['StudentExplanation']).toarray()
X_test_extra = test_df[['word_count', 'Topic']].copy()
X_test_extra['Topic'] = X_test_extra['Topic'].map(topic_map)

X_test = np.hstack([X_test_text, X_test_extra])

# ---------------------------
# Predict probabilities on test set
# ---------------------------
cat_proba = model_cat.predict_proba(X_test)
mis_proba = model_mis.predict_proba(X_test)

cat_labels = model_cat.classes_
mis_labels = model_mis.classes_

# ---------------------------
# Combine probabilities to rank top-3 predictions per row
# ---------------------------
final_predictions = []

for i in range(len(test_df)):
    ranked_pairs = []
    for cat_idx, cat_label in enumerate(cat_labels):
        for mis_idx, mis_label in enumerate(mis_labels):
            score = cat_proba[i][cat_idx] * mis_proba[i][mis_idx]
            ranked_pairs.append((f"{cat_label}:{mis_label}", score))
    ranked_pairs.sort(key=lambda x: x[1], reverse=True)
    top_3 = [pair[0] for pair in ranked_pairs[:3]]
    final_predictions.append(" ".join(top_3))

# ---------------------------
# Prepare submission DataFrame
# ---------------------------
submission = pd.DataFrame({
    'row_id': test_df['row_id'],
    'Category:Misconception': final_predictions
})

submission.to_csv(OUTPUT_PATH, index=False)
print(f"Saved submission file to {OUTPUT_PATH}")
