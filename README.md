# 📰 Fake News Detector

This is a simple **Fake News Detection App** built using **Python**, **Streamlit**, and **Machine Learning**.

It uses **NLP** and a **Naive Bayes Classifier** trained to detect whether a news article is **Fake** or **Real**.

---

## 🚀 How It Works

- Upload or type news text
- The app processes the text with a trained model (`fakenews_nb_model.pkl`) and TF-IDF vectorizer
- It predicts if the news is fake or real

---

## 🛠️ Tech Stack

- Python
- Streamlit
- Scikit-learn
- NLP (Natural Language Processing)

---

## 📂 How to Run Locally

You have **two ways** to run this project:

---

### ✅ Option 1: Run with `.bat` file (Windows only)

```bash
# Just double-click the `run_fake_news_app.bat` file in the project folder.
# Clone this repo
git clone https://github.com/ompisal63/fake-news-detector.git

# Change directory
cd fake-news-detector

# (Optional) Activate virtual environment and install requirements
pip install -r requirements.txt

# Run the Streamlit app
streamlit run app.py
