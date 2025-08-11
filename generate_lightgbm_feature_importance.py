import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib

# ===========================
# FIX: Add LightGBM DLL path (Windows only, before importing lightgbm)
# ===========================
if os.name == 'nt':  # Windows
    dll_path = os.path.abspath(os.path.join('venv', 'lib', 'site-packages', 'lightgbm', 'bin'))
    if os.path.exists(dll_path):
        os.add_dll_directory(dll_path)
        print(f"Added DLL path: {dll_path}")
    else:
        print(f"Warning: DLL path not found: {dll_path}. "
              f"Ensure LightGBM is installed correctly.")

import lightgbm as lgb

# ===========================
# CONFIGURATION
# ===========================
MODELS_DIR = os.path.abspath('models')

# Load vectorizers to get feature names
try:
    word_vect = joblib.load(os.path.join(MODELS_DIR, 'word_vect.pkl'))
    char_vect = joblib.load(os.path.join(MODELS_DIR, 'char_vect.pkl'))
    ques_vect = joblib.load(os.path.join(MODELS_DIR, 'ques_vect.pkl'))
    print("Vectorizers loaded successfully")
except FileNotFoundError as e:
    print(f"Error: Missing vectorizer file. {e}")
    sys.exit(1)

feature_names = (
    list(word_vect.get_feature_names_out()) +
    list(char_vect.get_feature_names_out()) +
    list(ques_vect.get_feature_names_out()) +
    ['word_count', 'short_flag', 'contains_answer']
)

# Collect gain importances from all models across folds
all_importances = []

# Load is_true models (4 folds)
for fold in range(4):
    model_path = os.path.join(MODELS_DIR, f'is_true_model_fold{fold}.pkl')
    if os.path.exists(model_path):
        model = joblib.load(model_path)
        imp = model.feature_importance(importance_type='gain')
        all_importances.append(imp)

# Load branch models (branch 0 and 1, each 4 folds)
for branch in [0, 1]:
    for fold in range(4):
        model_path = os.path.join(MODELS_DIR, f'branch_{branch}_model_fold{fold}.pkl')
        if os.path.exists(model_path):
            model = joblib.load(model_path)
            imp = model.feature_importance(importance_type='gain')
            all_importances.append(imp)

# Load misconception models (4 folds)
for fold in range(4):
    model_path = os.path.join(MODELS_DIR, f'misconception_model_fold{fold}.pkl')
    if os.path.exists(model_path):
        model = joblib.load(model_path)
        imp = model.feature_importance(importance_type='gain')
        all_importances.append(imp)

# Compute average importance if any models were loaded
if all_importances:
    avg_imp = np.mean(all_importances, axis=0)

    # Normalize to percentages
    total_imp = np.sum(avg_imp)
    avg_imp_perc = (avg_imp / total_imp) * 100 if total_imp > 0 else avg_imp

    # Create DataFrame for sorting
    imp_df = pd.DataFrame({'feature': feature_names, 'importance': avg_imp_perc})
    imp_df = imp_df.sort_values('importance', ascending=False).head(20)

    # Generate bar plot
    plt.figure(figsize=(12, 10))
    plt.barh(imp_df['feature'], imp_df['importance'], color='skyblue')
    plt.xlabel('Importance (%)')
    plt.title('Top 20 Feature Importances (Gain) from All LightGBM Models')
    plt.gca().invert_yaxis()  # Highest importance at top
    plt.tight_layout()

    # Save the figure
    plt.savefig('figure_5_feature_importance.png', dpi=300, bbox_inches='tight')
    plt.close()

    print("Figure 5 saved as 'figure_5_feature_importance.png'")
else:
    print("No models found in the specified directory.")
