import pandas as pd

# Load and display the first few rows
df = pd.read_csv('data/multilingual_sample.csv')
print(df[['text', 'text_es', 'text_hi']].head())
