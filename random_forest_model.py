import pandas as pd
import numpy as np
import re
import autocorrect
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

# ---------------------------
# Preprocessing function
# ---------------------------
def preprocess_text(text):
    text = str(text).lower()
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    speller = autocorrect.Speller()
    text = ' '.join(speller(word) for word in text.split())  # Spell correction
    return text

# ---------------------------
# Load training data (keep_default_na=False so "NA" stays as a string)
# ---------------------------
df = pd.read_csv('data/train.csv', keep_default_na=False)

# Preprocess text
df['StudentExplanation'] = df['StudentExplanation'].apply(preprocess_text)

# Create word_count feature
df['word_count'] = df['StudentExplanation'].apply(lambda x: len(str(x).split()))

# Create Topic feature
df['Topic'] = df['QuestionText'].apply(
    lambda x: 'Fractions' if 'fraction' in str(x).lower()
    else 'Probability' if 'probability' in str(x).lower()
    else 'Other'
)

# ---------------------------
# TF-IDF Vectorization
# ---------------------------
vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
X_text = vectorizer.fit_transform(df['StudentExplanation']).toarray()

# Extra features
X_extra = df[['word_count', 'Topic']].copy()
X_extra['Topic'] = X_extra['Topic'].map({'Fractions': 0, 'Probability': 1, 'Other': 2})
X = np.hstack([X_text, X_extra])

# ---------------------------
# Separate targets
# ---------------------------
y_category = df['Category']
y_misconception = df['Misconception']

# ---------------------------
# Train-test split
# ---------------------------
X_train, X_val, y_cat_train, y_cat_val, y_mis_train, y_mis_val = train_test_split(
    X, y_category, y_misconception, test_size=0.2, random_state=42
)

# ---------------------------
# Train two Random Forest models
# ---------------------------
model_cat = RandomForestClassifier(
    n_estimators=200, random_state=42, class_weight='balanced'
)
model_cat.fit(X_train, y_cat_train)

model_mis = RandomForestClassifier(
    n_estimators=200, random_state=42, class_weight='balanced'
)
model_mis.fit(X_train, y_mis_train)

# ---------------------------
# Evaluate
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
# Predict on test set
# ---------------------------
test_df = pd.read_csv('data/test.csv', keep_default_na=False)

# Preprocess
test_df['StudentExplanation'] = test_df['StudentExplanation'].apply(preprocess_text)
test_df['word_count'] = test_df['StudentExplanation'].apply(lambda x: len(str(x).split()))
test_df['Topic'] = test_df['QuestionText'].apply(
    lambda x: 'Fractions' if 'fraction' in str(x).lower()
    else 'Probability' if 'probability' in str(x).lower()
    else 'Other'
)

# Vectorize
X_test_text = vectorizer.transform(test_df['StudentExplanation']).toarray()
X_test_extra = test_df[['word_count', 'Topic']].copy()
X_test_extra['Topic'] = X_test_extra['Topic'].map({'Fractions': 0, 'Probability': 1, 'Other': 2})
X_test = np.hstack([X_test_text, X_test_extra])

# Predictions
cat_predictions = model_cat.predict(X_test)
mis_predictions = model_mis.predict(X_test)

# ---------------------------
# Build Kaggle output format
# ---------------------------
# For each test sample, we create the three pairs in order:
# True_Correct:<mis> False_Neither:<mis> False_Misconception:<mis>
final_predictions = []
for cat, mis in zip(cat_predictions, mis_predictions):
    # Build each label part
    pred_str = f"True_Correct:{mis if cat == 'True_Correct' else 'NA'} " \
               f"False_Neither:{mis if cat == 'True_Neither' else 'NA'} " \
               f"False_Misconception:{mis if cat == 'False_Misconception' else 'NA'}"
    final_predictions.append(pred_str)

# Save submission
submission = pd.DataFrame({
    'row_id': test_df['row_id'],
    'Category:Misconception': final_predictions
})
submission.to_csv('submission.csv', index=False)

print("Submission file saved as 'submission.csv' in correct Kaggle format")
