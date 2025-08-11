import pandas as pd
import re
from collections import Counter
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import seaborn as sns
import os
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
import numpy as np

# ========== Output Directory ==========
output_dir = 'output/eda'
os.makedirs(output_dir, exist_ok=True)

# ========== Load Dataset ==========
df = pd.read_csv('data/train.csv')

# 1. Dataset Overview
print("Dataset Overview:")
print(f"Rows: {len(df)}")
print(f"Columns: {df.columns.tolist()}")
print(f"Unique Questions: {df['QuestionId'].nunique()}")

# 2. Category Distribution
category_counts = df['Category'].value_counts()
category_percentages = df['Category'].value_counts(normalize=True) * 100
gini = 1 - sum((category_percentages / 100) ** 2)
print("\nCategory Distribution:")
print(category_counts)
print(category_percentages)
print(f"Gini Coefficient (Imbalance Measure): {gini:.4f}")

# Bar chart with percentage labels
plt.figure(figsize=(8, 5))
ax = sns.barplot(x=category_counts.index, y=category_counts.values)
plt.title('Category Distribution')
plt.ylabel('Count')
plt.xticks(rotation=45)
for i, v in enumerate(category_counts.values):
    ax.text(i, v, f'{category_percentages.iloc[i]:.1f}%', ha='center', va='bottom')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'category_distribution.png'))
plt.close()

# 3. Word Count Analysis
df['word_count'] = df['StudentExplanation'].apply(lambda x: len(str(x).split()))
word_count_stats = df.groupby('Category')['word_count'].mean()
print("\nAverage Word Count by Category:")
print(word_count_stats)

# Word count histogram
plt.figure(figsize=(10, 6))
for category in df['Category'].unique():
    sns.histplot(df[df['Category'] == category]['word_count'], label=category, bins=30, alpha=0.5)
plt.title('Word Count Distribution by Category')
plt.xlabel('Word Count')
plt.ylabel('Frequency')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'word_count_histogram.png'))
plt.close()

# 4. Frequent Words Analysis
stop_words = set(ENGLISH_STOP_WORDS)

def get_frequent_words(text_series, n=5):
    all_words = []
    for text in text_series.dropna():
        words = re.findall(r'\b[a-zA-Z]+\b', str(text).lower())
        words = [w for w in words if w not in stop_words]
        all_words.extend(words)
    return Counter(all_words).most_common(n)

false_misconception_texts = df[df['Category'] == 'False_Misconception']['StudentExplanation']
true_correct_texts = df[df['Category'] == 'True_Correct']['StudentExplanation']

false_words = get_frequent_words(false_misconception_texts, n=5)
true_words = get_frequent_words(true_correct_texts, n=5)
print("\nTop 5 Words in False_Misconception:")
print(false_words)
print("\nTop 5 Words in True_Correct:")
print(true_words)

# Frequent words bar chart
words_df = pd.DataFrame({
    'Word': [w[0] for w in false_words] + [w[0] for w in true_words],
    'Count': [w[1] for w in false_words] + [w[1] for w in true_words],
    'Category': ['False_Misconception'] * 5 + ['True_Correct'] * 5
})
plt.figure(figsize=(12, 6))
sns.barplot(x='Count', y='Word', hue='Category', data=words_df)
plt.title('Top 5 Words in False_Misconception vs. True_Correct')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'frequent_words_comparison.png'))
plt.close()

# 5. Word Cloud
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(' '.join(false_misconception_texts.dropna().astype(str)))
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Word Cloud for False_Misconception Explanations')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'wordcloud_misconception.png'))
plt.close()

# 6. Misconception Types
misconception_counts = df[df['Category'].isin(['False_Misconception', 
    'True_Misconception'])]['Misconception'].value_counts()
print("\nMisconception Types (False_Misconception and True_Misconception):")
print(misconception_counts)

# Bar chart for top 5 misconceptions
top_misconceptions = misconception_counts.head(5)
plt.figure(figsize=(8, 5))
sns.barplot(x=top_misconceptions.values, y=top_misconceptions.index)
plt.title('Top 5 Misconceptions in False_Misconception and True_Misconception')
plt.xlabel('Count')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'top_misconceptions.png'))
plt.close()

# 7. Question Topic Categorization
def categorize_question(text):
    text = str(text).lower()
    if 'fraction' in text:
        return 'Fractions'
    elif 'decimal' in text:
        return 'Decimals'
    elif 'probability' in text:
        return 'Probability'
    else:
        return 'Other'

df['Topic'] = df['QuestionText'].apply(categorize_question)
topic_counts = df['Topic'].value_counts(normalize=True) * 100
print("\nTopic Distribution:")
print(topic_counts)

# Cross-tabulation of Category vs. Topic
cross_tab = pd.crosstab(df['Category'], df['Topic'], normalize='columns') * 100
print("\nCategory Distribution by Topic (%):")
print(cross_tab)

# Stacked bar chart for Category vs. Topic
cross_tab.plot(kind='bar', stacked=True, figsize=(10, 6))
plt.title('Category Distribution by Topic')
plt.ylabel('Percentage (%)')
plt.xticks(rotation=45)
plt.legend(title='Topic')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'category_by_topic.png'))
plt.close()

# 8. Data Quality Checks
print("\nData Quality:")
print(f"Missing Values:\n{df.isnull().sum()}")
short_explanations = df[df['word_count'] < 5]['StudentExplanation'].count()
print(f"Short Explanations (<5 words): {short_explanations}")

# 9. Save Summary
with open(os.path.join(output_dir, 'eda_summary.txt'), 'w') as f:
    f.write("EDA Summary\n")
    f.write(f"Rows: {len(df)}\n")
    f.write(f"Unique Questions: {df['QuestionId'].nunique()}\n")
    f.write(f"\nCategory Distribution:\n{category_counts}\n{category_percentages}\n")
    f.write(f"\nGini Coefficient (Imbalance Measure): {gini:.4f}\n")
    f.write(f"\nAverage Word Count:\n{word_count_stats}\n")
    f.write(f"\nTop 5 Words in False_Misconception:\n{false_words}\n")
    f.write(f"\nTop 5 Words in True_Correct:\n{true_words}\n")
    f.write(f"\nMisconception Types (False_Misconception and True_Misconception):\n{misconception_counts}\n")
    f.write(f"\nTopic Distribution:\n{topic_counts}\n")
    f.write(f"\nCategory Distribution by Topic (%):\n{cross_tab}\n")
    f.write(f"\nMissing Values:\n{df.isnull().sum()}\n")
    f.write(f"Short Explanations (<5 words): {short_explanations}\n")