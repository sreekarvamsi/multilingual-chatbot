import pandas as pd
from googletrans import Translator
import time

# Load dataset with UTF-8 encoding
df = pd.read_csv('data/sample_customer_support.csv', nrows=100, encoding='utf-8')
translator = Translator()

# Translate 'text' column to Spanish and Hindi
df['text_es'] = df['text'].apply(lambda x: translator.translate(x, dest='es').text)
time.sleep(1)  # Avoid rate limiting
df['text_hi'] = df['text'].apply(lambda x: translator.translate(x, dest='hi').text)

# Save with UTF-8 encoding
df.to_csv('data/multilingual_sample.csv', index=False, encoding='utf-8-sig')
print("Sample multilingual dataset created!")
