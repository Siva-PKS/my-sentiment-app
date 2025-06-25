import streamlit as st
import pandas as pd
import plotly.express as px
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForSeq2SeqLM
import torch
import os
import warnings

# Fix torch Streamlit bug
try:
    del torch._classes
except AttributeError:
    pass

warnings.filterwarnings("ignore", category=FutureWarning, module="huggingface_hub.file_download")

st.set_page_config(page_title="Sentiment Analyzer & Auto-Responder", layout="wide")
st.title("\U0001F310 Multilingual Sentiment Analyzer & Auto-Responder")

# Session state setup
for key in ["processed", "last_uploaded_filename"]:
    if key not in st.session_state:
        st.session_state[key] = None if key == "last_uploaded_filename" else False

uploaded_file = st.file_uploader("\U0001F4C1 Upload CSV with 'Review_text' column", type="csv")
sample_data_path = "sample_data.csv"

# Handle data input
if uploaded_file:
    if st.session_state.last_uploaded_filename != uploaded_file.name:
        st.session_state.last_uploaded_filename = uploaded_file.name
        st.session_state.processed = False
    df = pd.read_csv(uploaded_file)
else:
    if os.path.exists(sample_data_path):
        st.success("✅ Using 'sample_data.csv' from directory.")
        df = pd.read_csv(sample_data_path)
    else:
        st.error("❌ 'sample_data.csv' not found. Please upload a CSV.")
        st.stop()

# Validation
if df.empty or "Review_text" not in df.columns:
    st.error("❌ Missing or empty 'Review_text' column.")
    st.stop()

# Drop NA and limit rows for performance
MAX_ROWS = 10  # Keep small for Streamlit Cloud
df = df.dropna(subset=["Review_text"]).head(MAX_ROWS)

# Load multilingual sentiment model
@st.cache_resource
def load_sentiment_model():
    model_name = "cardiffnlp/twitter-xlm-roberta-base-sentiment"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    return tokenizer, model

# Load instruction-tuned multilingual LLM (mt0-small)
@st.cache_resource
def load_llm_model():
    tokenizer = AutoTokenizer.from_pretrained("bigscience/mt0-small", use_fast=False, legacy=True)
    model = AutoModelForSeq2SeqLM.from_pretrained("bigscience/mt0-small")
    return tokenizer, model

sentiment_tokenizer, sentiment_model = load_sentiment_model()
llm_tokenizer, llm_model = load_llm_model()

# Sentiment label map
label_map = {
    0: "Negative",
    1: "Neutral",
    2: "Positive"
}

# Batch sentiment analysis
def analyze_all_sentiments(texts, batch_size=8):
    sentiment_model.eval()
    results = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            inputs = sentiment_tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=512)
            outputs = sentiment_model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
            labels = torch.argmax(probs, dim=1).tolist()
            results.extend([label_map.get(l, "Unknown") for l in labels])
    return results

# Generate response
@st.cache_data(show_spinner=False)
def generate_response_cached(sentiment, review):
    if sentiment != "Negative":
        return "No response needed."
    prompt = f"Respond politely to this negative review: {review}"
    inputs = llm_tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512, padding=True)
    output = llm_model.generate(**inputs, max_new_tokens=80)  # Limit tokens for speed
    reply = llm_tokenizer.decode(output[0], skip_special_tokens=True).strip()
    return f"Thank you for your review. We will look into the issue. {reply.rstrip('.!?')}."

# Process data
if not st.session_state.processed:
    try:
        with st.spinner("Analyzing sentiments and generating responses..."):
            df["Sentiment"] = analyze_all_sentiments(df["Review_text"].tolist())

            responses = []
            for i, (sentiment, review) in enumerate(zip(df["Sentiment"], df["Review_text"])):
                response = generate_response_cached(sentiment, review)
                responses.append(response)

            df["Response"] = responses
            st.session_state.df_processed = df.copy()
            st.session_state.processed = True
    except Exception as e:
        st.error(f"❌ An error occurred: {str(e)}")
        st.stop()

# Use cached processed data
df = st.session_state.df_processed

st.success("✅ Processing complete!")
st.subheader("\U0001F4CB Preview")
cols_to_show = [col for col in ["Unique_ID", "Category", "Review_text", "Sentiment", "Response"] if col in df.columns]
st.dataframe(df[cols_to_show], use_container_width=True)

# Sentiment breakdown chart
st.subheader("\U0001F4CA Sentiment Breakdown")
chart_data = df["Sentiment"].value_counts().reset_index()
chart_data.columns = ["Sentiment", "Count"]
fig = px.bar(chart_data, x="Sentiment", y="Count", color="Sentiment",
             color_discrete_map={"Positive": "green", "Neutral": "gray", "Negative": "red"})
st.plotly_chart(fig, use_container_width=True)

# Download button
st.download_button(
    label="⬇️ Download CSV",
    data=df.to_csv(index=False).encode("utf-8"),
    file_name="sentiment_responses.csv",
    mime="text/csv"
)
