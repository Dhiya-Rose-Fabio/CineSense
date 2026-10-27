import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

st.set_page_config(
    page_title="Movie Sentiment Analyzer",
    page_icon="🎬",
    layout="centered"
)

# Custom styling
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #0f1026, #211438, #101326);
    color: white;
}

.block-container {
    max-width: 800px;
    padding-top: 3rem;
}

h1 {
    text-align: center;
    font-size: 3rem !important;
    background: linear-gradient(90deg, #ff7eb3, #a78bfa);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.subtitle {
    text-align: center;
    color: #b9b6d3;
    font-size: 1.1rem;
    margin-bottom: 2rem;
}

.stTextArea textarea {
    background-color: #211d3b !important;
    color: white !important;
    border: 1px solid #65528e !important;
    border-radius: 15px !important;
    padding: 15px !important;
}

.stButton > button {
    width: 100%;
    background: linear-gradient(90deg, #ec4899, #8b5cf6);
    color: white;
    border: none;
    border-radius: 12px;
    padding: 12px;
    font-size: 17px;
    font-weight: bold;
}

.stButton > button:hover {
    color: white;
    border: none;
    background: linear-gradient(90deg, #db2777, #7c3aed);
}

.result-card {
    background: rgba(255,255,255,0.07);
    padding: 25px;
    border-radius: 18px;
    text-align: center;
    margin-top: 20px;
    border: 1px solid rgba(255,255,255,0.12);
}

.footer {
    text-align: center;
    color: #8f8ba9;
    margin-top: 60px;
    font-size: 13px;
}
</style>
""", unsafe_allow_html=True)


# Load trained model
@st.cache_resource
def load_model():
    path = "dhiyarosefabio/cinesense-distilbert"
    tokenizer = AutoTokenizer.from_pretrained(path)
    model = AutoModelForSequenceClassification.from_pretrained(path)
    model.eval()
    return tokenizer, model

tokenizer, model = load_model()


# Heading
st.title("🎬 Movie Sentiment Analyzer")

st.markdown(
    '<p class="subtitle">Discover the emotion behind every movie review ✨</p>',
    unsafe_allow_html=True
)

st.markdown("### 🍿 Your Movie Review")

review = st.text_area(
    "Movie Review",
    placeholder="Was the movie a masterpiece or a disappointment? Tell us what you think...",
    height=180,
    label_visibility="collapsed"
)

# Analyze button
if st.button("✨ Analyze Sentiment"):

    if review.strip() == "":
        st.warning("Please enter a movie review!")

    else:
        with st.spinner("Analyzing your review..."):

            inputs = tokenizer(
                review,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=128,
                return_token_type_ids=False
            )

            with torch.no_grad():
                outputs = model(**inputs)

            probabilities = torch.softmax(outputs.logits, dim=1)[0]

            negative = probabilities[0].item()
            positive = probabilities[1].item()

        if positive >= negative:
            st.markdown("""
            <div class="result-card">
                <h2 style="color:#4ade80;">😊 Positive Review!</h2>
                <p>The audience seems to love this movie!</p>
            </div>
            """, unsafe_allow_html=True)

        else:
            st.markdown("""
            <div class="result-card">
                <h2 style="color:#fb7185;">😞 Negative Review!</h2>
                <p>Looks like this movie didn't impress!</p>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("### 📊 Sentiment Confidence")

        st.write(f"💚 Positive — {positive * 100:.2f}%")
        st.progress(positive)

        st.write(f"💗 Negative — {negative * 100:.2f}%")
        st.progress(negative)


st.markdown(
    '<p class="footer">🎞️ Powered by DistilBERT | Made with Streamlit 💜</p>',
    unsafe_allow_html=True
)
