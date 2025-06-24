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
st.title("🌐 Multilingual Sentiment Analyzer & Auto-Responder")

# Session state setup
for key in ["processed", "last_uploaded_filename"]:
    if key not in st.session_state:
        st.session_state[key] = None if key == "last_uploaded_filename" else False

uploaded_file = st.file_uploader("📁 Upload CSV with 'Review_text' column", type="csv")
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

# Limit rows for demo speed
MAX_ROWS = 100
if len(df) > MAX_ROWS:
    st.warning(f"⚠️ Limiting to first {MAX_ROWS} rows for demo.")
    df = df.head(MAX_ROWS)

# Load multilingual sentiment model
@st.cache_resource
def load_sentiment_model():
    model_name = "cardiffnlp/twitter-xlm-roberta-base-sentiment"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    return tokenizer, model

# Load LLM model for auto-responses
@st.cache_resource
def load_llm_model():
    tokenizer = AutoTokenizer.from_pretrained("google/mt5-small", use_fast=False)
    model = AutoModelForSeq2SeqLM.from_pretrained("google/mt5-small")
    return tokenizer, model

sentiment_tokenizer, sentiment_model = load_sentiment_model()
llm_tokenizer, llm_model = load_llm_model()

# Mapping from model config
label_map = {
    0: "Negative",
    1: "Neutral",
    2: "Positive"
}

def analyze_all_sentiments(texts):
    results = []
    sentiment_model.eval()
    with torch.no_grad():
        for text in texts:
            inputs = sentiment_tokenizer(text[:512], return_tensors="pt", truncation=True, padding=True)
            outputs = sentiment_model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
            label_id = torch.argmax(probs, dim=1).item()
            results.append(label_map.get(label_id, "Unknown"))
    return results

def generate_response(sentiment, review):
    if sentiment != "Negative":
        return "No response needed."

    prompt = f"Generate a polite customer support reply to this negative review: {review}"

    inputs = llm_tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    output = llm_model.generate(**inputs, max_new_tokens=150)

    llm_reply = llm_tokenizer.decode(output[0], skip_special_tokens=True).strip()
    llm_reply = llm_reply.replace("<extra_id_0>", "").strip()

    return f"Thank you for your review. We will look into the issue. {llm_reply.rstrip('.!?')}."


# Process data if not already done
if not st.session_state.processed:
    progress_bar = st.progress(0)
    df["Sentiment"] = analyze_all_sentiments(df["Review_text"].tolist())
    responses = []
    for i, row in df.iterrows():
        sentiment = row["Sentiment"]
        review = row["Review_text"]
        responses.append(generate_response(sentiment, review))
        progress_bar.progress((i + 1) / len(df))
    df["Response"] = responses
    st.session_state.df_processed = df.copy()
    st.session_state.processed = True

# Use cached processed data
df = st.session_state.df_processed

st.success("✅ Processing complete!")
st.subheader("📋 Preview")
cols_to_show = [col for col in ["Unique_ID", "Category", "Review_text", "Sentiment", "Response"] if col in df.columns]
st.dataframe(df[cols_to_show], use_container_width=True)

# Sentiment breakdown
st.subheader("📊 Sentiment Breakdown")
chart_data = df["Sentiment"].value_counts().reset_index()
chart_data.columns = ["Sentiment", "Count"]
fig = px.bar(chart_data, x="Sentiment", y="Count", color="Sentiment",
             color_discrete_map={"Positive": "green", "Neutral": "gray", "Negative": "red"})
st.plotly_chart(fig, use_container_width=True)

# CSV Download
st.download_button(
    label="⬇️ Download CSV",
    data=df.to_csv(index=False).encode("utf-8"),
    file_name="sentiment_responses.csv",
    mime="text/csv"
)
