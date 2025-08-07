import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from imblearn.over_sampling import SMOTE
import re
import autocorrect

# Load dataset
df = pd.read_csv('data/train.csv')

# Preprocessing function
def preprocess_text(text):
    text = str(text).lower()
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    speller = autocorrect.Speller()
    text = ' '.join(speller(word) for word in text.split())  # Spell correction
    return text

# Apply preprocessing
df['StudentExplanation'] = df['StudentExplanation'].apply(preprocess_text)
df['Topic'] = df['QuestionText'].apply(lambda x: 'Fractions' if 'fraction' in str(x).lower() else 'Probability' if 'probability' in x.lower() else 'Other')

# TF-IDF Vectorization
vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
X_text = vectorizer.fit_transform(df['StudentExplanation']).toarray()

# Add features
X_extra = df[['word_count', 'Topic']].copy()
X_extra['Topic'] = X_extra['Topic'].map({'Fractions': 0, 'Probability': 1, 'Other': 2})
X = np.hstack([X_text, X_extra])

# Target
y = df['Category']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Handle imbalance with SMOTE
smote = SMOTE(random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)

# Train Random Forest
model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
model.fit(X_train_balanced, y_train_balanced)

# Evaluate
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average='weighted')
print(f"Accuracy: {accuracy:.4f}")
print(f"F1 Score: {f1:.4f}")

# Save predictions for Kaggle
test_df = pd.read_csv('data/test.csv')
test_df['StudentExplanation'] = test_df['StudentExplanation'].apply(preprocess_text)
test_df['Topic'] = test_df['QuestionText'].apply(lambda x: 'Fractions' if 'fraction' in str(x).lower() else 'Probability' if 'probability' in x.lower() else 'Other')
X_test_text = vectorizer.transform(test_df['StudentExplanation']).toarray()
X_test_extra = test_df[['word_count', 'Topic']].copy()
X_test_extra['Topic'] = X_test_extra['Topic'].map({'Fractions': 0, 'Probability': 1, 'Other': 2})
X_test = np.hstack([X_test_text, X_test_extra])
predictions = model.predict(X_test)
submission = pd.DataFrame({'row_id': test_df['row_id'], 'Category': predictions})
submission.to_csv('submission.csv', index=False)
