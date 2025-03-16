import pandas as pd
import re
from transformers import XLMRobertaTokenizer

tokenizer = XLMRobertaTokenizer.from_pretrained('xlm-roberta-base')

# Load raw data
df = pd.read_csv('data/multilingual_sample.csv')

# Cleaning function
def clean_text(text):
    if pd.isna(text):
        return ""
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)  # Remove URLs
    text = re.sub(r'[^A-Za-z0-9\s.,!?]', '', text)  # Keep alphanumeric and basic punctuation
    return text.strip()

# Apply cleaning to all language columns
for col in ['text', 'text_es', 'text_hi']:
    df[col] = df[col].apply(clean_text)

# Drop rows with empty text in all columns
df = df[df[['text', 'text_es', 'text_hi']].ne('').any(axis=1)]
df.to_csv('data/cleaned_multilingual.csv', index=False)
print("Data cleaned and saved!")

# Tokenize each language column
def tokenize_text(text):
    return tokenizer(text, padding=True, truncation=True, max_length=128, return_tensors='pt')

# Apply tokenization (store tokenized IDs for later use if needed)
df['tokens_en'] = df['text'].apply(lambda x: tokenize_text(x)['input_ids'].tolist())
df['tokens_es'] = df['text_es'].apply(lambda x: tokenize_text(x)['input_ids'].tolist())
df['tokens_hi'] = df['text_hi'].apply(lambda x: tokenize_text(x)['input_ids'].tolist())
df.to_csv('data/tokenized_multilingual.csv', index=False)
print("Data tokenized and saved!")
