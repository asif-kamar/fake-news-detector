import joblib
import re

# 1. Load the model and the vectorizer
model = joblib.load('models/news_model.pkl')
tfidf = joblib.load('models/tfidf_vectorizer.pkl')

def clean_text(text):
    text = text.lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub("\\W"," ",text)
    return text

def predict_news():
    print("\n--- Fake News Detector ---")
    user_input = input("Enter the news headline or text to check: ")
    
    # 2. Preprocess the input
    cleaned_input = clean_text(user_input)
    
    # 3. Transform the text using the loaded TF-IDF vectorizer
    # Note: we wrap input in a list [cleaned_input] because the vectorizer expects a collection
    vectorized_input = tfidf.transform([cleaned_input])
    
    # 4. Make the prediction
    prediction = model.predict(vectorized_input)
    
    # 5. Output the result
    if prediction[0] == 1:
        print("\nRESULT: This news looks REAL. ✅")
    else:
        print("\nRESULT: This news looks FAKE. ❌")

if __name__ == "__main__":
    while True:
        predict_news()
        cont = input("\nCheck another? (y/n): ")
        if cont.lower() != 'y':
            break