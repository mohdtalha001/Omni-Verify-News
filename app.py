from flask import Flask, render_template, request
import pickle
import re
import os
from nltk.corpus import stopwords
from mtranslate import translate
import requests

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Model aur Vectorizer load karein
model = pickle.load(open('fake_news_model.pkl', 'rb'))
vectorizer = pickle.load(open('tfidf_vectorizer.pkl', 'rb'))
stop_words = set(stopwords.words('english'))

def clean_input(text):
    text = text.lower()
    text = re.sub(r'\W', ' ', text)
    words = text.split()
    cleaned_words = [word for word in words if word not in stop_words]
    return " ".join(cleaned_words)

@app.route('/')
def home():
    return render_template('index.html')

# 1. TEXT DETECTION (HINDI/ENGLISH)
@app.route('/predict_text', methods=['POST'])
def predict_text():
    if request.method == 'POST':
        news_text = request.form['news']
        try:
            translated_text = translate(news_text, 'en')
        except Exception:
            translated_text = news_text
            
        cleaned_news = clean_input(translated_text)
        vectorized_text = vectorizer.transform([cleaned_news])
        prediction = model.predict(vectorized_text)
        
        result = "REAL NEWS" if prediction[0] == 1 else "FAKE NEWS"
        return render_template('index.html', prediction_text=result)

# 2. VIDEO FILE DETECTION
@app.route('/predict_video', methods=['POST'])
def predict_video():
    if request.method == 'POST':
        if 'video' not in request.files:
            return render_template('index.html', prediction_text="No video file uploaded")
        file = request.files['video']
        if file.filename == '':
            return render_template('index.html', prediction_text="No selected file")
            
        if file:
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)
            
            # Temporary logic for testing
            if 'fake' in file.filename.lower() or 'deepfake' in file.filename.lower():
                result = "DEEPFAKE / FAKE VIDEO DETECTED"
            else:
                result = "REAL VIDEO"
                
            if os.path.exists(filepath):
                os.remove(filepath)
            return render_template('index.html', prediction_text=result)

# 3. VIDEO LINK DETECTION (YOUTUBE / TWITTER LINKS)
@app.route('/predict_link', methods=['POST'])
def predict_link():
    if request.method == 'POST':
        url = request.form['video_url']
        
        # URL validate karne ki basic check
        if not url.startswith("http://") and not url.startswith("https://"):
            return render_template('index.html', prediction_text="INVALID URL FORMAT")
            
        try:
            # Yahan backend mein API ya custom verification logic video download/process karegi.
            # Abhi testing ke liye agar URL mein 'fake' word hai toh fake batayega:
            if 'fake' in url.lower() or 'mock' in url.lower():
                result = "DEEPFAKE / FAKE VIDEO (FROM LINK)"
            else:
                result = "REAL VIDEO (AUTHENTIC LINK)"
        except Exception as e:
            result = "Error reading video link"
            
        return render_template('index.html', prediction_text=result)

if __name__ == '__main__':
    app.run(debug=True)