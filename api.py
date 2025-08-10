# api.py
import uvicorn
import joblib
import numpy as np
import pandas as pd
import re
from scipy import sparse
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List

app = FastAPI()

# Load your saved vectorizers and encoders once on startup
word_vect = joblib.load('models/word_vect.pkl')
char_vect = joblib.load('models/char_vect.pkl')
ques_vect = joblib.load('models/ques_vect.pkl')
cat_le = joblib.load('models/cat_le.pkl')
mis_le = joblib.load('models/mis_le.pkl')

# Load one or more models (for example, one fold)
is_true_model = joblib.load('models/is_true_model_fold0.pkl')
branch_0_model = joblib.load('models/branch_0_model_fold0.pkl')
branch_1_model = joblib.load('models/branch_1_model_fold0.pkl')
misconception_model = joblib.load('models/misconception_model_fold0.pkl')

# Utility: preprocess text
def preprocess_text(text):
    text = str(text).lower()
    text = re.sub(r'[^\w\s]', '', text)
    return text

# Input data schema
class InputSample(BaseModel):
    row_id: int
    StudentExplanation: str
    QuestionText: str
    MC_Answer: str

class PredictRequest(BaseModel):
    samples: List[InputSample]

# Helper function to prepare features for new data
def prepare_features(df):
    df['StudentExplanation'] = df['StudentExplanation'].apply(preprocess_text)
    df['QuestionText'] = df['QuestionText'].apply(preprocess_text)
    df['word_count'] = df['StudentExplanation'].apply(lambda x: len(str(x).split())).astype(np.int32)
    df['short_flag'] = (df['word_count'] < 5).astype(np.int8)
    df['contains_answer'] = df.apply(
        lambda r: str(r['MC_Answer']).lower() in r['StudentExplanation'],
        axis=1
    ).astype(np.int8)

    X_word = word_vect.transform(df['StudentExplanation'])
    X_char = char_vect.transform(df['StudentExplanation'])
    X_ques = ques_vect.transform(df['QuestionText'])
    X_meta = sparse.csr_matrix(df[['word_count', 'short_flag', 'contains_answer']].values.astype(np.float32))
    X_all = sparse.hstack([X_word, X_char, X_ques, X_meta]).tocsr()
    return X_all

@app.post("/predict")
def predict(request: PredictRequest):
    df = pd.DataFrame([s.dict() for s in request.samples])
    X_all = prepare_features(df)

    # 1. Predict is_true branch probabilities
    is_true_preds = is_true_model.predict(X_all)
    is_true_class = np.argmax(is_true_preds, axis=1)

    # Prepare results
    results = []

    for i in range(len(df)):
        branch = is_true_class[i]

        # Select branch model
        if branch == 0:
            branch_model = branch_0_model
        else:
            branch_model = branch_1_model

        # Predict category probabilities
        cat_probs = branch_model.predict(X_all[i])

        # Predict misconception probabilities
        mis_probs = misconception_model.predict(X_all[i])

        # Get topk category indexes and names
        topk_cat_idx = np.argsort(-cat_probs)[0][:3]
        topk_cat_names = cat_le.inverse_transform(topk_cat_idx)

        # Get topk misconception indexes and names
        topk_mis_idx = np.argsort(-mis_probs)[0][:3]
        topk_mis_names = mis_le.inverse_transform(topk_mis_idx)

        # Combine results "Category:Misconception"
        combined_preds = [f"{c}:{m}" for c, m in zip(topk_cat_names, topk_mis_names)]

        results.append({
            "row_id": int(df.loc[i, 'row_id']),
            "predictions": combined_preds
        })

    return {"results": results}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
