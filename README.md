📰 Fake News Detector AI
An end-to-end machine learning system that classifies news articles as Real or Fake. This project features a high-performance Passive Aggressive Classifier backend and an interactive Streamlit web dashboard for real-time verification.

🚀 Key Features
Interactive Web UI: Built with Streamlit for a seamless, user-friendly experience.

Real-time Inference: Instantly analyzes news headlines or full articles.

Optimized NLP Pipeline: Uses TF-IDF Vectorization to handle high-velocity text data.

High Accuracy: Achieved a validation accuracy of 97.2% using the ISOT Research Dataset.

🛠️ Installation & Setup
1. Clone the Repository
Bash
git clone https://github.com/asif-kamar/fake-news-detector.git
cd fake-news-detector
2. Install Dependencies
Bash
pip install -r requirements.txt
3. Data Preparation
Note: Large datasets and models are excluded from the repository to maintain a lean structure and avoid GitHub file size limits.

Create the necessary local directories: mkdir data models

Download the ISOT Fake News Dataset from Kaggle.

Place True.csv and Fake.csv into the data/ folder.

📊 How to Run
Step 1: Train the Model
Before running the web app, you must generate the trained model and vectorizer locally:

Bash
python src/clean_data.py
python src/train_model.py
This will process the raw data and save news_model.pkl and tfidf_vectorizer.pkl into the models/ directory.

Step 2: Launch the Streamlit App
Start the interactive web interface:

Bash
streamlit run app.py
The app will automatically open in your default browser at http://localhost:8501.

🧠 Algorithm & Logic
Preprocessing: Raw text is cleaned by removing punctuation, special characters, and converting to lowercase to reduce noise.

Vectorization: TF-IDF (Term Frequency-Inverse Document Frequency) converts text into weighted numerical information, highlighting unique keywords that distinguish fake news from factual reporting.

Classifier: The Passive Aggressive Classifier (PAC) is used for its efficiency in high-speed text classification and its ability to handle large-scale data with minimal memory overhead.

📂 Project Structure
app.py: The main Streamlit web application.

src/: Core logic for data cleaning and model training.

data/: (Local only) Raw and cleaned news datasets.

models/: (Local only) Serialized machine learning models (.pkl).

tests/: Sample headlines for batch testing.

📝 Author
Asif Kamar BTech in Artificial Intelligence and Machine Learning

Vidya Academy of Science and Technology

📜 License
This project is open-source and available under the MIT License.