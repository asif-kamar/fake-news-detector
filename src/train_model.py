import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
import joblib

df = pd.read_csv('data/cleaned_news.csv')
x_train, x_test, y_train, y_test = train_test_split(df['text'], df['label'], test_size=0.2)

# Convert text to numbers
tfidf = TfidfVectorizer(stop_words='english', max_df=0.7)
x_train_tfidf = tfidf.fit_transform(x_train)

# Train
model = PassiveAggressiveClassifier(max_iter=50)
model.fit(x_train_tfidf, y_train)

# Save the model and vectorizer
joblib.dump(model, 'models/news_model.pkl')
joblib.dump(tfidf, 'models/tfidf_vectorizer.pkl')
print("Model trained and saved in /models/")