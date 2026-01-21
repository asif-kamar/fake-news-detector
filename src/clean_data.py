import pandas as pd
import re
import string

# Load
true_df = pd.read_csv('data/True.csv')
fake_df = pd.read_csv('data/Fake.csv')

# Label and Combine
true_df['label'] = 1
fake_df['label'] = 0
df = pd.concat([true_df, fake_df])

def clean_text(text):
    text = text.lower()
    text = re.sub('\[.*?\]', '', text) # Remove brackets
    text = re.sub("\\W"," ",text)     # Remove special chars
    return text

df['text'] = df['text'].apply(clean_text)
df.to_csv('data/cleaned_news.csv', index=False)
print("Data cleaned and saved to data/cleaned_news.csv")