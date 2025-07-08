import streamlit as st
import joblib
import requests
from streamlit_lottie import st_lottie

model = joblib.load("fakenews_nb_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

st.set_page_config(
    page_title="Fake News Detector",
    page_icon="📰",
    layout="centered"
)

st.markdown(
    """
    <style>
    body {
        background-color: #000000;
        color: #ffffff;
    }
    .stApp {
        background-color: #000000;
        color: #ffffff;
    }
    h1, h2, h3, h4, h5, h6, p {
        color: #ffffff;
    }
    .big-title {
        font-size: 48px;
        font-weight: 700;
        text-align: center;
        color: #ffffff;
    }
    .sub-text {
        font-size: 18px;
        text-align: center;
        color: #cccccc;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown('<h1 class="big-title">📰 Fake News Detector</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-text">An elegant way to detect True or Fake News</p>', unsafe_allow_html=True)


def load_lottie_url(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

lottie_url = "https://assets10.lottiefiles.com/packages/lf20_WpDGFs.json"  # Simple news search animation
lottie_json = load_lottie_url(lottie_url)
if lottie_json:
    st_lottie(lottie_json, height=200)

st.write("")
st.write("")

st.subheader("📝 Enter News Text")
news_input = st.text_area("", height=200, placeholder="Type or paste the news content here...")

if st.button("Detect"):
    if news_input.strip() == "":
        st.warning("Please enter some news text.")
    else:
        news_vect = vectorizer.transform([news_input])
        pred = model.predict(news_vect)

        if pred[0] == 1:
            st.success("This is likely **True News** ✅")
        else:
            st.error("This is likely **Fake News** ⚠️")

with st.sidebar:
    st.title("About")
    st.write(
        """
        **Fake News Detector**  
        - Uses a trained **Naive Bayes** model  
        - Text processed with **TF-IDF**  
        - Built with **Streamlit** """
    )



